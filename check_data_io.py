#coding=utf-8
import os
import numpy as np
import tensorflow as tf
import cv2
from preparedata import PrepareData
from nets.ssd import g_ssd_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

obj= PrepareData()

image, filename,glabels,gbboxes,gdifficults,gclasses_face, localizations_face, gscores_face,\
gclasses_head, localizations_head, gscores_head,gclasses_body, localizations_body,\
gscores_body=obj.get_wider_face_train_data()

ssd_anchors = g_ssd_model.ssd_anchors_all_layers()

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(5):
        img,picname,label,bbox,gclass,glocal,gscore=sess.run([image, filename,glabels,gbboxes,gclasses_face, localizations_face, gscores_face])
        
        b=np.zeros_like(img[0])
        b[:,:,1]=img[0][:,:,1]
        b[:,:,0]=img[0][:,:,2]
        b[:,:,2]=img[0][:,:,0]
        box=bbox[0]
        hh,ww=b.shape[:2]
        
        gboxes=[]
        for u in range(len(gclass)):
            gbox=glocal[u][0].reshape([-1,4])
            gcls=gclass[u][0].reshape([-1])
            gsc=gscore[u][0].reshape([-1])
            anchor_bboxes=ssd_anchors[u]
            yref, xref, href_src, wref_src = anchor_bboxes
            href=href_src/2.
            wref=wref_src/2.
            xref = np.reshape(xref, [-1])
            yref = np.reshape(yref, [-1])
            cx = gbox[:,  0] * wref * 0.1 + xref
            cy = gbox[:,  1] * href * 0.1 + yref
            w = wref * np.exp(gbox[:,  2] * 0.2)
            h = href * np.exp(gbox[:,  3] * 0.2)
            bboxes = np.zeros_like(gbox)
            bboxes[:,  0] = cy - h / 2.
            bboxes[:,  1] = cx - w / 2.
            bboxes[:,  2] = cy + h / 2.
            bboxes[:,  3] = cx + w / 2.
            
            for f in range(len(gcls)):
                if gcls[f]==1:
                    gboxes.append(gbox[f])
                    cv2.rectangle(b,(int(bboxes[f][1]*640),int(bboxes[f][0]*640)),(int(bboxes[f][3]*640),int(bboxes[f][2]*640)),(0,0,255),3)        
        
        for j in range(len(box)):
            cv2.rectangle(b,(int(box[j][1]*ww),int(box[j][0]*hh)),(int(box[j][3]*ww),int(box[j][2]*hh)),(0,255,0),3)
        cv2.imshow('test',b.astype(np.uint8))
        cv2.waitKey(0)
    
    coord.request_stop()
    coord.join(threads)
