#coding=utf-8
import os
import math
import random
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import scipy.io as sio
from tensorflow.contrib import slim
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from preprocessing import ssd_vgg_preprocessing
from utility import visualization
from nets.ssd import g_ssd_model
import nets.np_methods as np_methods

# TensorFlow session: grow memory when needed.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, data_format, resize=ssd_vgg_preprocessing.Resize.NONE)#WARP_RESIZE
image_4d = tf.expand_dims(image_pre, 0)

# Define the PyramidBox model.
predictions, localisations, _, end_points = g_ssd_model.get_model(image_4d)

# Restore PyramidBox model.
ckpt_filename = tf.train.latest_checkpoint('logs/finetune/')

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)


# Main image processing routine.
def process_image(img, select_threshold=0.05, nms_threshold=0.3):
    # Run PyramidBox network.
    h,w=img.shape[:2]
    if h<w and h<640:
        scale=640./h
        h=640
        w=int(w*scale)
    elif h>=w and w<640:
        scale=640./w
        w=640
        h=int(h*scale)
    if h>5000:
        h=3000
    img_tmp=Image.fromarray(np.uint8(img))
    resized_img=img_tmp.resize((w,h))    
    net_shape=np.array(resized_img).shape[:2]
    rimg, rpredictions, rlocalisations, rbbox_img,e_ps = isess.run([image_4d, predictions, localisations, bbox_img,end_points],feed_dict={img_input: resized_img})
    
    layer_shape=[e_ps['block3'].shape[1:3],e_ps['block4'].shape[1:3],e_ps['block5'].shape[1:3],e_ps['block7'].shape[1:3],e_ps['block8'].shape[1:3],e_ps['block9'].shape[1:3]]
    
    # SSD default anchor boxes.
    ssd_anchors = g_ssd_model.ssd_anchors_all_layers(feat_shapes=layer_shape,img_shape=net_shape)
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations[0], ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=1000)

    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
wider_face_mat = sio.loadmat('eval/eval_tools/ground_truth/wider_face_val.mat')
event_list = wider_face_mat['event_list']
file_list = wider_face_mat['file_list']
save_path = r'eval/pyramidbox_val/'

for ind, event in enumerate(event_list):
    filelist = file_list[ind][0]
    im_dir = event[0][0]
    if not os.path.exists(save_path + im_dir): os.makedirs(save_path + im_dir)

    for num, file in enumerate(filelist):
        im_name = file[0][0]
        impath = 'datasets/widerface/WIDER_val/images/'+'%s/%s.jpg' % (im_dir,im_name)
        if os.path.exists(save_path + im_dir + '/' + im_name + '.txt'):
            continue
        img = Image.open(impath)
        h,w=np.array(img).shape[:2]
        rclasses1, rscores1, rbboxes1=process_image(np.array(img))

        img_flip=img.transpose(Image.FLIP_LEFT_RIGHT) 
        rclasses2, rscores2, rbboxes2=process_image(np.array(img_flip))
        temp=rbboxes2[:,1].copy()
        rbboxes2[:,1]=1.-rbboxes2[:,3]
        rbboxes2[:,3]=1.-temp

        flag1=0
        if h*w*4<2048*2048:
            img_scale1=img.resize((w*2,h*2))
            rclasses3, rscores3, rbboxes3=process_image(np.array(img_scale1))
            index = np.where(np.minimum((rbboxes3[:,2]-rbboxes3[:,0])*h,(rbboxes3[:,3]-rbboxes3[:,1])*w)<=96)[0] # only detect small face
            rclasses3=rclasses3[index]
            rscores3=rscores3[index]
            rbboxes3=rbboxes3[index]
            flag1=1
        flag2=0    
        if h*w/4>960*960:
            img_scale2=img.resize((int(w*0.5),int(h*0.5)))
            rclasses4, rscores4, rbboxes4=process_image(np.array(img_scale2))
            index = np.where(np.minimum((rbboxes4[:,2]-rbboxes4[:,0])*h,(rbboxes4[:,3]-rbboxes4[:,1])*w)>96)[0] # only detect large face
            rclasses4=rclasses4[index]
            rscores4=rscores4[index]
            rbboxes4=rbboxes4[index]
            flag2=1

        if flag1==1 and flag2==1:
            rclasses=np.concatenate((rclasses1,rclasses2,rclasses3,rclasses4))
            rscores=np.concatenate((rscores1,rscores2,rscores3,rscores4))
            rbboxes=np.concatenate((rbboxes1,rbboxes2,rbboxes3,rbboxes4))
        elif flag1==1 and flag2==0:
            rclasses=np.concatenate((rclasses1,rclasses2,rclasses3))
            rscores=np.concatenate((rscores1,rscores2,rscores3))
            rbboxes=np.concatenate((rbboxes1,rbboxes2,rbboxes3))
        elif flag2==1 and flag1==0:
            rclasses=np.concatenate((rclasses1,rclasses2,rclasses4))
            rscores=np.concatenate((rscores1,rscores2,rscores4))
            rbboxes=np.concatenate((rbboxes1,rbboxes2,rbboxes4))
        else:
            rclasses=np.concatenate((rclasses1,rclasses2))
            rscores=np.concatenate((rscores1,rscores2))
            rbboxes=np.concatenate((rbboxes1,rbboxes2))

        rclasses, rscores, rbboxes = np_methods.bboxes_nms_fast(rclasses, rscores, rbboxes, nms_threshold=0.1)
        rbbox_img=[0.,0.,1.,1.]
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        
        f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
        f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir,im_name)))
        f.write('{:d}\n'.format(len(rbboxes)))
        for ii in range(len(rbboxes)):
            y1,x1,y2,x2 = rbboxes[ii]
            y1*=h
            x1*=w
            y2*=h
            x2*=w
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(int(x1),int(y1),int(x2-x1+1),int(y2-y1+1),rscores[ii]))
        f.close()
        print('event:%d num:%d' % (ind + 1, num + 1))


    
        
