import os
import math
import random
import time
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from tensorflow.contrib import slim

# slim = tf.contrib.slim

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
    img_input, None, None, data_format, resize=ssd_vgg_preprocessing.Resize.NONE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.

predictions, localisations, _, end_points = g_ssd_model.get_model(image_4d)

# Restore SSD model.
ckpt_filename = 'model/pyramidbox.ckpt'

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)




# Main image processing routine.
def process_image(img, select_threshold=0.35, nms_threshold=0.1):
    # Run SSD network.
    h,w=img.shape[:2]
    if h<w and h<640:
        scale=640./h
        h=640
        w=int(w*scale)
    elif h>=w and w<640:
        scale=640./w
        w=640
        h=int(h*scale)
    img=Image.fromarray(np.uint8(img))
    resized_img=img.resize((w,h))    
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
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=1200)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = 'demo/images/'
image_names = sorted(os.listdir(path))
for i in range(len(image_names)):
    img = np.array(Image.open(path + image_names[i]))
    rclasses, rscores, rbboxes =  process_image(img)
    visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    plt.show()

