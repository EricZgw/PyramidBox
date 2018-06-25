
import tensorflow as tf

import tensorflow.contrib.slim as slim

import numpy as np

import math
from numpy import newaxis
from nets import custom_layers
import tf_extended as tfe
from nets import ssd_common
from tensorflow.python.ops import array_ops


class PyramidBoxModel():
    """Implementation of the PyramidBox vgg-based 640 network.

    The default features layers with 640x640 image input are:
      block3 ==> 160 x 160
      block4 ==> 80 x 80
      block5 ==> 40 x 40
      block7 ==> 20 x 20
      block8 ==> 10 x 10
      block9 ==> 5 x 5
    The default image size used to train this network is 640x640.
    """
    def __init__(self):
        
        self.img_shape=(640, 640)
        self.num_classes=2
        self.no_annotation_label=2
        self.feat_layers=['block3', 'block4', 'block5', 'block7', 'block8', 'block9']
        self.feat_shapes=[(160, 160), (80, 80), (40, 40), (20, 20), (10, 10), (5, 5)]
        self.layer_shape=[]
        self.anchor_sizes=[16.,32.,64.,128.,256.,512.]
        self.anchor_ratios=1. 
        self.anchor_steps=[4, 8, 16, 32, 64, 128] 
        self.anchor_offset=0.5
        #Scaling of encoded coordinates.
        #For the scaling, the idea is try to scale such that all error terms (classification + position + size) 
        #have roughly the same scaling. Otherwise, the training would tend to over-optimise one component and not the others.
        self.prior_scaling=[0.1, 0.1, 0.2, 0.2] 
        
        #normalization for conv4 3
        self.normalizations=[1, 1, 1, -1, -1, -1]
        
        #thresholding for ignoring "no annotation label"
        self.ignore_threshold = 0.5 

        self.np__anchors = None

        self.np_anchors_minmax = None
        self.model_name = 'ssd_300_vgg'
        
        #post processing
        self.select_threshold = 0.01
        self.nms_threshold = 0.1
        self.select_top_k = 600
        self.keep_top_k = 400
        
        return
    def __dropout(self,net):
        net_shape = net.get_shape().as_list() 
        noise_shape = [net_shape[0],1,1,net_shape[-1]]
        return slim.dropout(net, noise_shape=noise_shape)
    def __additional_ssd_block(self, end_points,channels, net, is_training=False):
        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        #net = slim.batch_norm(net)
        #net = self.__dropout(net)
        end_points['block6'] = net
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        #net = slim.batch_norm(net)
        #net = self.__dropout(net)
        end_points['block7'] = net
        channels['block7']=1024
        self.layer_shape.append(tfe.get_shape(net)[1:3])

        # Block 8/9: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            #net = slim.batch_norm(net)
            #net = self.__dropout(net)
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            #net = slim.batch_norm(net)
            #net = self.__dropout(net)
        end_points[end_point] = net
        channels[end_point]=512
        self.layer_shape.append(tfe.get_shape(net)[1:3])
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            #net = slim.batch_norm(net)
            #net = self.__dropout(net)
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            #net = slim.batch_norm(net)
            #net = self.__dropout(net)
        end_points[end_point] = net
        channels[end_point]=256
        self.layer_shape.append(tfe.get_shape(net)[1:3])

        # Prediction and localisations layers.
        predictions = []

        logits, localisations = self.ssd_multibox_layer(end_points,
                                  channels,
                                  self.feat_layers,
                                  self.normalizations,
                                  is_training=is_training)
        
        if is_training==True:
            return localisations, logits, end_points
        else:
            predictions = []
            for l in range(len(logits[0])):
                predictions.append(slim.softmax(logits[0][l]))
            return predictions, localisations, logits, end_points
       
    
    def __arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Defines the VGG arg scope.
    
        Args:
          weight_decay: The l2 regularization coefficient.
    
        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format):
                with slim.arg_scope([custom_layers.pad2d,
                                     custom_layers.l2_normalization,
                                     custom_layers.channel_to_last],
                                    data_format=data_format) as sc:
                    return sc

    
    def get_model(self,inputs, weight_decay=0.0005,is_training=False):
        # End_points collect relevant activations for external use.
        arg_scope = self.__arg_scope(weight_decay=weight_decay)
        self.img_shape=tfe.get_shape(inputs)[1:3]
        with slim.arg_scope(arg_scope):
            end_points = {}
            channels={}
            with tf.variable_scope('vgg_16', [inputs]):
                # Original VGG-16 blocks.
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                end_points['block1'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                # Block 2.
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                end_points['block2'] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                # Block 3.
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                end_points['block3'] = net
                channels['block3']=256
                self.layer_shape.append(tfe.get_shape(net)[1:3])
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                # Block 4.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                end_points['block4'] = net
                channels['block4']=512
                self.layer_shape.append(tfe.get_shape(net)[1:3])
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                # Block 5.
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                end_points['block5'] = net
                channels['block5']=512
                self.layer_shape.append(tfe.get_shape(net)[1:3])
                net = slim.max_pool2d(net, [2, 2],  scope='pool5')
        
            # Additional SSD blocks.
            #with slim.arg_scope([slim.conv2d],
                            #activation_fn=None):
                #with slim.arg_scope([slim.batch_norm],
                            #activation_fn=tf.nn.relu, is_training=is_training,updates_collections=None):
                    #with slim.arg_scope([slim.dropout],
                            #is_training=is_training,keep_prob=0.8):
            with tf.variable_scope(self.model_name):
                return self.__additional_ssd_block(end_points, channels,net,is_training=is_training)
                        
    def cpm(self,inputs):
        residual1=slim.conv2d(inputs, 256, [1, 1], activation_fn=None,scope='cpm_conv1x1_branch1')
        #residual1=slim.batch_norm(residual1)
        inputs1=slim.conv2d(inputs, 1024, [1, 1], scope='cpm_conv_1x1_branch1a')
        #inputs1=slim.batch_norm(inputs1)
        inputs1=slim.conv2d(inputs1, 256, [1, 1], scope='cpm_conv_1x1_branch1b')
        #inputs1=slim.batch_norm(inputs1)
        inputs1=slim.conv2d(inputs1, 256, [1, 1],activation_fn=None, scope='cpm_conv_1x1_branch1c')
        #inputs1=slim.batch_norm(inputs1)
        branch1 = tf.add(inputs1, residual1, name='cpm_addition_branch1')
        branch1=tf.nn.relu(branch1)        

        residual2=slim.conv2d(inputs, 256, [1, 1], activation_fn=None,scope='cpm_conv1x1_branch2')
        #residual2=slim.batch_norm(residual2)
        inputs2=slim.conv2d(inputs, 1024, [1, 1], scope='cpm_conv_1x1_branch2a')
        #inputs2=slim.batch_norm(inputs2)
        inputs2=slim.conv2d(inputs2, 256, [1, 1], scope='cpm_conv_1x1_branch2b')
        #inputs2=slim.batch_norm(inputs2)
        inputs2=slim.conv2d(inputs2, 256, [1, 1],activation_fn=None, scope='cpm_conv_1x1_branch2c')
        #inputs2=slim.batch_norm(inputs2)
        branch2 = tf.add(inputs2, residual2, name='cpm_addition_branch2')
        branch2=tf.nn.relu(branch2) 
        
        residual2_1=slim.conv2d(branch2, 128, [1, 1], activation_fn=None,scope='cpm_conv1x1_branch2_1')
        #residual2_1=slim.batch_norm(residual2_1)
        inputs2_1=slim.conv2d(branch2, 1024, [1, 1], scope='cpm_conv_1x1_branch2_1a')
        #inputs2_1=slim.batch_norm(inputs2_1)
        inputs2_1=slim.conv2d(inputs2_1, 256, [1, 1], scope='cpm_conv_1x1_branch2_1b')
        #inputs2_1=slim.batch_norm(inputs2_1)
        inputs2_1=slim.conv2d(inputs2_1, 128, [1, 1],activation_fn=None, scope='cpm_conv_1x1_branch2_1c')
        #inputs2_1=slim.batch_norm(inputs2_1)
        branch2_1= tf.add(inputs2_1, residual2_1, name='cpm_addition_branch2_1')
        branch2_1=tf.nn.relu(branch2_1) 
        
        residual2_2=slim.conv2d(branch2, 128, [1, 1], activation_fn=None,scope='cpm_conv1x1_branch2_2')
        #residual2_2=slim.batch_norm(residual2_2)
        inputs2_2=slim.conv2d(branch2, 1024, [1, 1], scope='cpm_conv_1x1_branch2_2a')
        #inputs2_2=slim.batch_norm(inputs2_2)
        inputs2_2=slim.conv2d(inputs2_2, 256, [1, 1], scope='cpm_conv_1x1_branch2_2b')
        #inputs2_2=slim.batch_norm(inputs2_2)
        inputs2_2=slim.conv2d(inputs2_2, 128, [1, 1], activation_fn=None,scope='cpm_conv_1x1_branch2_2c')
        #inputs2_2=slim.batch_norm(inputs2_2)
        branch2_2= tf.add(inputs2_2, residual2_2, name='cpm_addition_branch2_2')
        branch2_2=tf.nn.relu(branch2_2)
        
        residual2_2_1=slim.conv2d(branch2_2, 128, [1, 1], activation_fn=None,scope='cpm_conv1x1_branch2_2_1')
        #residual2_2_1=slim.batch_norm(residual2_2_1)
        inputs2_2_1=slim.conv2d(branch2_2, 1024, [1, 1], scope='cpm_conv_1x1_branch2_2_1a')
        #inputs2_2_1=slim.batch_norm(inputs2_2_1)
        inputs2_2_1=slim.conv2d(inputs2_2_1, 256, [1, 1], scope='cpm_conv_1x1_branch2_2_1b')
        #inputs2_2_1=slim.batch_norm(inputs2_2_1)
        inputs2_2_1=slim.conv2d(inputs2_2_1, 128, [1, 1], activation_fn=None,scope='cpm_conv_1x1_branch2_2_1c')
        #inputs2_2_1=slim.batch_norm(inputs2_2_1)
        branch2_2_1= tf.add(inputs2_2_1, residual2_2_1, name='cpm_addition_branch2_2_1')
        branch2_2_1=tf.nn.relu(branch2_2_1)
	
        return tf.concat(values=[branch1, branch2_1,branch2_2_1], axis=3)
    def ssd_multibox_layer(self, end_points,
                       channels,
                       feat_layers,
                       normalization,
                       bn_normalization=False,
                       is_training=True):
        """Construct a multibox layer, return a class and localization predictions.
        """
        face_logits = []
        face_localisations = []
        head_logits = []
        head_localisations = []
        body_logits = []
        body_localisations = []
        pyramid_layer=end_points.copy()
        feat_layers.reverse()
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                if int(layer[-1])<7:
                        u=pyramid_layer[layer]
                        sz = tf.shape(u)
                        d=pyramid_layer[feat_layers[i-1]]
                        d_=slim.conv2d(d, channels[layer], [1, 1], scope='conv_1x1')
                        #d_=slim.batch_norm(d_)
                        d_=tf.image.resize_bilinear(d_,(sz[1],sz[2]),name='2xup')
                        u_ = tf.add(d_, u, name='addition')
                        pyramid_layer[layer]=u_
                        net_=u_
                else:
                        net=pyramid_layer[layer]
                if normalization[5-i] > 0:
                    net = custom_layers.l2_normalization(net_, scaling=True,scope='L2Norm%d'%(5-i))		
                net=self.cpm(net)
                # Location.
                num_loc_pred =  4
                loc_pred_face = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                                       scope='conv_loc_face')
                face_localisations.append(loc_pred_face)
                
                loc_pred_head = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                                       scope='conv_loc_head')
                head_localisations.append(loc_pred_head)
                
                loc_pred_body = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                                       scope='conv_loc_body')
                body_localisations.append(loc_pred_body)	
                
                # Class prediction.
                num_cls_pred =  4
                cls_pred_face = slim.conv2d(net, num_cls_pred, [1, 1], activation_fn=None,
                                       scope='conv_cls_face')
                if i==5:
                    cn1,cn2,cn3,cp=tf.split(cls_pred_face,4,3)
                    cn=tf.maximum(cn1,tf.maximum(cn2,cn3))
                else:
                    cn,cp1,cp2,cp3=tf.split(cls_pred_face,4,3)
                    cp=tf.maximum(cp1,tf.maximum(cp2,cp3))
                cls_pred_face=tf.concat([cn,cp],3)
                face_logits.append(cls_pred_face)
                
                num_cls_pred =  2					  
                cls_pred_head = slim.conv2d(net, num_cls_pred, [1, 1], activation_fn=None,
                                       scope='conv_cls_head')
                head_logits.append(cls_pred_head)	
                
                num_cls_pred =  2					  
                cls_pred_body = slim.conv2d(net, num_cls_pred, [1, 1], activation_fn=None,
                                       scope='conv_cls_body')			  
                body_logits.append(cls_pred_body)
        face_logits.reverse()
        head_logits.reverse()
        body_logits.reverse()
        face_localisations.reverse()
        head_localisations.reverse()
        body_localisations.reverse()
        feat_layers.reverse()
        if is_training==True:
            logits=[face_logits,head_logits,body_logits]
            localisations=[face_localisations,head_localisations,body_localisations]
        else:
            logits=[face_logits]
            localisations=[face_localisations]
        return logits,localisations
    
    
    def tensor_shape(self, x, rank=3):
        """Returns the dimensions of a tensor.
        Args:
          image: A N-D Tensor of shape.
        Returns:
          A list of dimensions. Dimensions that are statically known are python
            integers,otherwise they are integer scalar tensors.
        """
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(x), rank)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]
                
    def get_allanchors(self, minmaxformat=False,layer_shape=None,img_shape=None):
        
        if self.np__anchors is None:
            
            anchors = self.ssd_anchors_all_layers(feat_shapes=layer_shape,img_shape=img_shape)
            self.np_anchors_minmax_face =[]
            self.np_anchors_minmax_head =[]
            self.np_anchors_minmax_body =[]
            self.np__anchors_face = []
            self.np__anchors_head = []
            self.np__anchors_body = []
            for i, anchors_layer in enumerate(anchors):
                
                yref, xref, href, wref = anchors_layer
                href=href
                wref=wref
                ymin = yref - href / 2.
                xmin = xref - wref / 2.
                ymax = yref + href / 2
                xmax = xref + wref / 2.
                temp_achors = np.concatenate([ymin[...,np.newaxis],xmin[...,np.newaxis],ymax[...,np.newaxis],xmax[...,np.newaxis]], axis = -1)
                self.np_anchors_minmax_face.append(temp_achors)
                cy = (ymax + ymin) / 2.
                cx = (xmax + xmin) / 2.
                h = ymax - ymin
                w = xmax - xmin
                temp_achors = np.concatenate([cx[...,np.newaxis],cy[...,np.newaxis],w[...,np.newaxis],h[...,np.newaxis]], axis = -1)
                self.np__anchors_face.append(temp_achors)
                if i>0:
                    yref, xref, href_src, wref_src = anchors_layer
                    href=href_src/2
                    wref=wref_src/2
                    ymin = yref - href / 2.
                    xmin = xref - wref / 2.
                    ymax = yref + href / 2
                    xmax = xref + wref / 2.
                    temp_achors = np.concatenate([ymin[...,np.newaxis],xmin[...,np.newaxis],ymax[...,np.newaxis],xmax[...,np.newaxis]], axis = -1)
                    self.np_anchors_minmax_head.append(temp_achors)
                    cy = (ymax + ymin) / 2.
                    cx = (xmax + xmin) / 2.
                    h = ymax - ymin
                    w = xmax - xmin
                    temp_achors = np.concatenate([cx[...,np.newaxis],cy[...,np.newaxis],w[...,np.newaxis],h[...,np.newaxis]], axis = -1)
                    self.np__anchors_head.append(temp_achors)
                if i>1:
                    yref, xref, href_src, wref_src = anchors_layer
                    href=href_src/4
                    wref=wref_src/4
                    ymin = yref - href / 2.
                    xmin = xref - wref / 2.
                    ymax = yref + href / 2
                    xmax = xref + wref / 2.
                    temp_achors = np.concatenate([ymin[...,np.newaxis],xmin[...,np.newaxis],ymax[...,np.newaxis],xmax[...,np.newaxis]], axis = -1)
                    self.np_anchors_minmax_body.append(temp_achors)
                    cy = (ymax + ymin) / 2.
                    cx = (xmax + xmin) / 2.
                    h = ymax - ymin
                    w = xmax - xmin
                    temp_achors = np.concatenate([cx[...,np.newaxis],cy[...,np.newaxis],w[...,np.newaxis],h[...,np.newaxis]], axis = -1)
                    self.np__anchors_body.append(temp_achors)
            self.np_anchors_minmax=[self.np_anchors_minmax_face,self.np_anchors_minmax_head,self.np_anchors_minmax_body]
            self.np__anchors=[self.np__anchors_face,self.np__anchors_head,self.np__anchors_body]
        if  minmaxformat:
            return self.np_anchors_minmax
        else:
            return self.np__anchors
        
    def detected_bboxes(self, predictions, localisations,
                        clipping_bbox=None):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=self.select_threshold,
                                            num_classes=self.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=self.select_top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=self.nms_threshold,
                                 keep_top_k=self.keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes
    def decode_bboxes_all_ayers_tf(self, feat_localizations):
        """convert ssd boxes from relative to input image anchors to relative to input width/height
    
        Return:
          numpy array NlayersxNx4: ymin, xmin, ymax, xmax
        """
        anchors = self.ssd_anchors_all_layers(feat_shapes=self.layer_shape,img_shape=self.img_shape)
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.prior_scaling)
    def compute_jaccard(self, gt_bboxes, anchors):
        with tf.device('/cpu:0'):
            gt_bboxes = tf.reshape(gt_bboxes, (-1,1,4))
            anchors = tf.reshape(anchors, (1,-1,4))
            
            inter_ymin = tf.maximum(gt_bboxes[:,:,0], anchors[:,:,0])
            inter_xmin = tf.maximum(gt_bboxes[:,:,1], anchors[:,:,1])
            inter_ymax = tf.minimum(gt_bboxes[:,:,2], anchors[:,:,2])
            inter_xmax = tf.minimum(gt_bboxes[:,:,3], anchors[:,:,3])
            
            h = tf.maximum(inter_ymax - inter_ymin, 0.)
            w = tf.maximum(inter_xmax - inter_xmin, 0.)
            
            inter_area = h * w
            anchors_area = (anchors[:,:,3] - anchors[:,:,1]) * (anchors[:,:,2] - anchors[:,:,0])
            gt_bboxes_area = (gt_bboxes[:,:,3] - gt_bboxes[:,:,1]) * (gt_bboxes[:,:,2] - gt_bboxes[:,:,0])
            union_area = anchors_area - inter_area + gt_bboxes_area
            jaccard = inter_area/union_area
            
            return jaccard
    
    def __match_no_miss(self,gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores,jaccard,gt_labels,gt_bboxes, num_anchors):
        #make sure every ground truth box can be matched to at least one anchor box
        max_inds = tf.cast(tf.argmax(jaccard, axis=1),tf.int32)
        def cond(i,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores):
            r = tf.less(i, tf.shape(gt_labels)[0])
            return r
        def body(i,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores):
            
            #upate gt_anchors_labels
            updates = tf.reshape(gt_labels[i], [-1])
            indices = tf.reshape(max_inds[i],[1,-1])
            shape = tf.reshape(num_anchors,[-1])
            
            
            new_labels = tf.scatter_nd(indices, updates, shape)
            new_mask = tf.cast(new_labels, tf.bool)
            gt_anchors_labels = tf.where(new_mask, new_labels, gt_anchors_labels)
            
            #update gt_anchors_bboxes
            updates = tf.reshape(gt_bboxes[i], [1,-1])
            indices = tf.reshape(max_inds[i],[1,-1])
            shape = tf.shape(gt_anchors_bboxes)
            new_bboxes = tf.scatter_nd(indices, updates, shape)
            gt_anchors_bboxes = tf.where(new_mask, new_bboxes, gt_anchors_bboxes)
            
            #update gt_anchors_scores
            updates = tf.reshape(jaccard[i, max_inds[i]], [-1])
            indices = tf.reshape(max_inds[i],[1,-1])
            shape = tf.reshape(num_anchors,[-1])
            new_scores = tf.scatter_nd(indices, updates, shape)
            gt_anchors_scores = tf.where(new_mask, new_scores, gt_anchors_scores)
            
    
            
            return [i+1,gt_anchors_labels,gt_anchors_bboxes,gt_anchors_scores]
        
        
        i = 0
        [i,gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores] = tf.while_loop(cond, body,[i,gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores])
        
        return gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores
    
    def __match_no_labels(self,gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores,jaccard,matching_threshold,gt_labels,gt_bboxes,num_anchors):
        #For images without labels, just return all zero tensors
        
        return gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores
    def __match_with_labels(self,u,gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores,jaccard,matching_threshold,gt_labels,gt_bboxes,num_anchors):
       
        mask = tf.reduce_max (jaccard, axis = 0) > matching_threshold
        mask_inds = tf.argmax(jaccard, axis = 0)
        matched_labels = tf.gather(gt_labels, mask_inds)
        gt_anchor_labels = tf.where(mask, matched_labels, gt_anchor_labels)
        gt_anchor_bboxes = tf.where(mask, tf.gather(gt_bboxes, mask_inds),gt_anchor_bboxes)
        gt_anchor_scores = tf.reduce_max(jaccard, axis= 0)
    
        #matching each ground truth box to the default box with the best jaccard overlap
        if u==0:
            use_no_miss = True
        else:
            use_no_miss = False
        if use_no_miss:
            gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores = self.__match_no_miss(gt_anchor_labels, \
                                                                                      gt_anchor_bboxes, gt_anchor_scores, jaccard, \
                                                                                      gt_labels, gt_bboxes, num_anchors)
        
        return gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores
    
    def match_achors(self, gt_labels, gt_bboxes, matching_threshold = 0.35,layer_shape=None,img_shape=None):
        
        anchors_src = self.get_allanchors(minmaxformat=True,layer_shape=layer_shape,img_shape=img_shape)
        #flattent the anchors
        gt_anchor_bboxes_list=[]
        gt_anchor_labels_list=[]
        gt_anchor_scores_list=[]
        for u in range(3):
            temp_anchors = []
            for i in range(len(anchors_src[u])):
                temp_anchors.append(tf.reshape(anchors_src[u][i], [-1, 4]))
            anchors = tf.concat(temp_anchors, axis=0)
            
            jaccard = self.compute_jaccard(gt_bboxes, anchors)
            num_anchors= jaccard.shape[1]
            
            #initialize output
            gt_anchor_labels = tf.zeros(num_anchors, dtype=tf.int64)
            gt_anchor_scores = tf.zeros(num_anchors, dtype=tf.float32)
            gt_anchor_ymins = tf.zeros(num_anchors)
            gt_anchor_xmins = tf.zeros(num_anchors)
            gt_anchor_ymaxs = tf.ones(num_anchors)
            gt_anchor_xmaxs = tf.ones(num_anchors)
            gt_anchor_bboxes = tf.stack([gt_anchor_ymins,gt_anchor_xmins,gt_anchor_ymaxs,gt_anchor_xmaxs], axis=-1)
            
            n__glabels = tf.size(gt_labels)
            gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores = tf.cond(tf.equal(n__glabels, 0), \
                                                                         lambda: self.__match_no_labels(gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores,jaccard,matching_threshold,gt_labels,gt_bboxes,num_anchors), \
                                                                         lambda: self.__match_with_labels(u,gt_anchor_labels,gt_anchor_bboxes,gt_anchor_scores,jaccard,matching_threshold,gt_labels,gt_bboxes,num_anchors))
            
            
            
            # Transform to center / size.
            feat_cx = (gt_anchor_bboxes[:,3] + gt_anchor_bboxes[:,1]) / 2.
            feat_cy = (gt_anchor_bboxes[:,2] + gt_anchor_bboxes[:,0]) / 2.
            feat_w = gt_anchor_bboxes[:,3] - gt_anchor_bboxes[:,1]
            feat_h = gt_anchor_bboxes[:,2] - gt_anchor_bboxes[:,0]
            
            xref = (anchors[:,3] + anchors[:,1]) / 2.
            yref = (anchors[:,2] + anchors[:,0]) / 2.
            wref = anchors[:,3] - anchors[:,1]
            href = anchors[:,2] - anchors[:,0]
            
            
            # Encode features, convert ground truth bboxes to  shape offset relative to default boxes 
            feat_cx = (feat_cx - xref) / wref
            feat_cy = (feat_cy - yref) / href 
            feat_w = tf.log(feat_w / wref) 
            feat_h = tf.log(feat_h / href) 
            if u!=0:
                feat_cy=feat_cy-(1-2**u)/2.0*feat_h
                feat_cx=feat_cx-(1-2**u)/2.0*feat_w
                feat_h=(2**u)*feat_h
                feat_w=(2**u)*feat_w	
            feat_cy/=self.prior_scaling[0]
            feat_cx/=self.prior_scaling[1]
            feat_h/=self.prior_scaling[2]
            feat_w/=self.prior_scaling[3]
            
            gt_anchor_bboxes = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
            
            gt_anchor_bboxes_list.append(gt_anchor_bboxes)
            gt_anchor_labels_list.append(gt_anchor_labels)
            gt_anchor_scores_list.append(gt_anchor_scores)
        gt_anchor_labels_list, gt_anchor_bboxes_list,gt_anchor_scores_list = self.__convert2layers(gt_anchor_labels_list, gt_anchor_bboxes_list,gt_anchor_scores_list)

        return gt_anchor_labels_list, gt_anchor_bboxes_list, gt_anchor_scores_list
    def __convert2layers(self,gclasses, glocalisations, gscores):
        gt_anchor_labels_list = []
        gt_anchor_bboxes_list = []
        gt_anchor_scores_list = []
        
        anchors = self.get_allanchors(minmaxformat = False)
        for u in range(3):
            start = 0
            end = 0
            gt_anchor_labels = []
            gt_anchor_bboxes = []
            gt_anchor_scores = []
            for i in range(len(anchors[u])):
                anchor_shape = anchors[u][i].shape[:-1]
                anchor_shape = list(anchor_shape)
                anchor_num = np.array(anchor_shape).prod()
                start = end
                end = start + anchor_num

                gt_anchor_labels.append(tf.reshape(gclasses[u][start:end],anchor_shape))
                gt_anchor_scores.append(tf.reshape(gscores[u][start:end],anchor_shape))
                gt_anchor_bboxes.append(tf.reshape(glocalisations[u][start:end],anchor_shape + [4]))
            
            gt_anchor_labels_list.append(gt_anchor_labels)
            gt_anchor_scores_list.append(gt_anchor_scores)
            gt_anchor_bboxes_list.append(gt_anchor_bboxes)
            
        return gt_anchor_labels_list, gt_anchor_bboxes_list,gt_anchor_scores_list
   
    def ssd_anchors_all_layers(self,
                           dtype=np.float32,feat_shapes=None,img_shape=None):
        """Compute anchor boxes for all feature layers.
        """
        layers_anchors = []
        if feat_shapes==None and img_shape==None:
            for i, s in enumerate(self.feat_shapes):
                anchor_bboxes = self.__ssd_anchor_one_layer(self.img_shape, s,
                                                     self.anchor_sizes[i],        
                                                     self.anchor_steps[i],
                                                     offset=self.anchor_offset, dtype=dtype)
                layers_anchors.append(anchor_bboxes)
        else:
            for i, s in enumerate(feat_shapes):
                anchor_bboxes = self.__ssd_anchor_one_layer(img_shape, s,
                                                     self.anchor_sizes[i],        
                                                     self.anchor_steps[i],
                                                     offset=self.anchor_offset, dtype=dtype)
                layers_anchors.append(anchor_bboxes)
        return layers_anchors
    def __ssd_anchor_one_layer(self,img_shape,
                         feat_shape,
                         sizes,
                         step,
                         offset=0.5,
                         dtype=np.float32):
        """Computer SSD default anchor boxes for one feature layer.
    
        Determine the relative position grid of the centers, and the relative
        width and height.
    
        Arguments:
          feat_shape: Feature shape, used for computing relative position grids;
          size: Absolute reference sizes;
          ratios: Ratios to use on these features;
          img_shape: Image shape, used for computing height, width relatively to the
            former;
          offset: Grid offset.
    
        Return:
          y, x, h, w: Relative x and y grids, and height and width.
        """
        # Weird SSD-Caffe computation using steps values...
        y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
        y = (y.astype(dtype) + offset) * step / img_shape[0]
        x = (x.astype(dtype) + offset) * step / img_shape[1]
    
        # Expand dims to support easy broadcasting.
        y = np.expand_dims(y, axis=-1)
        x = np.expand_dims(x, axis=-1)
    
        # Compute relative height and width.
        # Tries to follow the original implementation of SSD for the order.

        h = np.zeros((1, ), dtype=dtype)
        w = np.zeros((1, ), dtype=dtype)
        # Add first anchor boxes with ratio=1.
        h[0] = sizes / img_shape[0]
        w[0] = sizes / img_shape[1]
        
        return y, x, h, w
   
    
    def get_losses(self, logits3, localisations3,
               gclasses3, glocalisations3, gscores3,
               match_threshold=0.5,
               negative_ratio=2.,
               alpha=1.,
               label_smoothing=0.,
               scope=None):
        """Loss functions for training the SSD 300 VGG network.
    
        This function defines the different loss components of the SSD, and
        adds them to the TF loss collection.
    
        Arguments:
          logits: (list of) predictions logits Tensors;
          localisations: (list of) localisations Tensors;
          gclasses: (list of) groundtruth labels Tensors;
          glocalisations: (list of) groundtruth localisations Tensors;
          gscores: (list of) groundtruth score Tensors;
        """
        with tf.name_scope(scope, 'ssd_losses'):
            train_or_eval_test=len(logits3)
            all_pmask=[]
            apmask=[]
            for u in range(train_or_eval_test):
                gclasses=gclasses3[u]
                fgclasses = []
                for i in range(len(gclasses)):
                    fgclasses.append(tf.reshape(gclasses[i], [-1]))
                gclasses = tf.concat(fgclasses, axis=0)
                pmask = gclasses > 0
                all_pmask.append(pmask)
            part1=all_pmask[0][0:25600]
            part2_temp=tf.logical_or(all_pmask[0][25600:],all_pmask[1][:],name='or1')
            part2=part2_temp[0:6400]
            part3=tf.logical_or(part2_temp[6400:],all_pmask[2][:],name='or2')
            apmask.append(tf.concat([part1,part2,part3],axis=0))
            apmask.append(tf.concat([part2,part3],axis=0))
            apmask.append(part3)
            for u in range(train_or_eval_test):
                logits=logits3[u]
                localisations=localisations3[u]
                gclasses=gclasses3[u]
                glocalisations=glocalisations3[u]
                gscores=gscores3[u]
                lshape = tfe.get_shape(logits[0], 4)
                num_classes = 2
                batch_size = lshape[0]
                # Flatten out all vectors!
                flogits = []
                fgclasses = []
                fgscores = []
                flocalisations = []
                fglocalisations = []
                for i in range(len(logits)-u):
                    flogits.append(tf.reshape(logits[i+u], [-1, num_classes]))
                    fgclasses.append(tf.reshape(gclasses[i], [-1]))
                    fgscores.append(tf.reshape(gscores[i], [-1]))
                    flocalisations.append(tf.reshape(localisations[i+u], [-1, 4]))
                    fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
                # And concat the crap!
                logits = tf.concat(flogits, axis=0)
                gclasses = tf.concat(fgclasses, axis=0)
                gscores = tf.concat(fgscores, axis=0)
                localisations = tf.concat(flocalisations, axis=0)
                glocalisations = tf.concat(fglocalisations, axis=0)
                dtype = logits.dtype
    
                # Compute positive matching mask...
                pmask = gclasses > 0
                fpmask = tf.cast(pmask, dtype)
                n_positives = tf.reduce_sum(fpmask)
        
                # Hard negative mining...
                #for no_classes, we only care that false positive's label is 0
                #this is why pmask sufice our needs
                no_classes = tf.cast(apmask[u], tf.int32)
                
                nmask = tf.logical_not(apmask[u])
                
                fnmask = tf.cast(nmask, dtype)
            
                # Number of negative entries to select.
                max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
               
                n_neg = tf.cast(negative_ratio * n_positives, tf.int32)
                n_neg = tf.minimum(n_neg, max_neg_entries)
                #avoid n_neg is zero, and cause error when doing top_k later on
                n_neg = tf.maximum(n_neg, 1)
        
                
                extend_weight=1.0
                if u==1:
                    extend_weight=0.5
                elif u==2:
                    extend_weight=0.25
                # Add cross-entropy loss.
                with tf.name_scope('cross_entropy_pos%d' % u):
                    total_cross_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=gclasses)
                    total_cross_pos = tf.reduce_sum(total_cross_pos * fpmask, name="cross_entropy_pos")
                    total_cross_pos=tf.cond(n_positives>0,lambda:tf.div(total_cross_pos,n_positives),lambda:0.)
                    tf.losses.add_loss(total_cross_pos)
        
                with tf.name_scope('cross_entropy_neg%d' % u):
                    total_cross_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=no_classes)
                    val, idxes = tf.nn.top_k(total_cross_neg* fnmask,k=n_neg)   
                    total_cross_neg=tf.reduce_sum(val,name="cross_entropy_neg")                                                   
                    total_cross_neg=tf.cond(n_positives>0,lambda:tf.div(total_cross_neg,n_positives),lambda:0.)
                    tf.losses.add_loss(total_cross_neg)
        
                # Add localization loss: smooth L1, L2, ...
                with tf.name_scope('localization%d' % u):
                    # Weights Tensor: positive mask + random negative.
                    weights = tf.expand_dims(alpha * fpmask, axis=-1)
                    total_loc = custom_layers.abs_smooth_2(localisations - glocalisations)
                    total_loc = tf.reduce_sum(total_loc * weights*extend_weight, name="localization")
                    total_loc=tf.cond(n_positives>0,lambda:tf.div(total_loc,n_positives),lambda:0.)
                    tf.losses.add_loss(total_loc)
            
                total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy%d' % u)
            
            
                # Add to EXTRA LOSSES TF.collection
                tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
                tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
                tf.add_to_collection('EXTRA_LOSSES', total_cross)
                tf.add_to_collection('EXTRA_LOSSES', total_loc)
           
                tf.summary.scalar('postive_num%d' % u, n_positives)
                tf.summary.scalar('negative_num%d' % u, n_neg)
            
            model_loss=tf.get_collection(tf.GraphKeys.LOSSES)
            model_loss = tf.add_n(model_loss)
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses,name='regularization_loss')
            tf.summary.scalar('regularization_loss', regularization_loss)
            total_loss=tf.add(model_loss, regularization_loss)
            return total_loss
   
    
    
    def run(self):
        
        
        return
    
    
g_ssd_model = PyramidBoxModel()

if __name__ == "__main__":   
    obj= PyramidBoxModel()
    obj.run()
