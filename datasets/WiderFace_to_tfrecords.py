
"""Converts Pascal VOC data to TFRecords file format with Example protos.

The raw Pascal VOC data set is expected to reside in JPEG files located in the
directory 'JPEGImages'. Similarly, bounding box annotations are supposed to be
stored in the 'Annotation directory'

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

Each validation TFRecord file contains ~500 records. Each training TFREcord
file contains ~1000 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always'JPEG'


    image/object/bbox/xmin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/xmax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymin: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/ymax: list of float specifying the 0+ human annotated
        bounding boxes
    image/object/bbox/label: list of integer specifying the classification index.
    image/object/bbox/label_text: list of string descriptions.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.
"""
import math
import os
import random

import cv2
import tensorflow as tf

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature


def _process_image(info,dataset_path):
    '''

    :param directory: image directory
    :param name: image name
    :return:
    '''
    #filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    filename = os.path.join(dataset_path, info[0] + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    img = cv2.imread(filename)


    #print(filename)
    #H,W,C
    shape = img.shape
    #print (shape)
    data_example = dict()
    data_example['name'] = filename
    data_example['image'] = image_data
    data_example['shape'] = shape
    #print(info[1])
    # Find annotations.
    bboxes = []
    bbox = dict()


    for i in range(1,len(info)-1):

        info[i] = float(info[i])
        assert  isinstance(info[i],float)
        if i%4 == 1:
           # print(info[i])
            bbox['xmin'] = info[i]
        if i%4 == 2:
            bbox['ymin'] = info[i]
        if i%4 == 3:
            bbox['xmax'] = info[i]
        if i%4 == 0:
            bbox['ymax'] = info[i]
            bboxes.append(bbox)
            bbox = dict()
   # print('filename: %s, bboxes: %s' %(filename,bboxes))
    data_example['bboxes'] = bboxes


    # for bbox in bboxes:
    #     cv2.rectangle(img,
    #                   (int(float(bbox['xmin'])),int(float(bbox['ymin']))),
    #                   (int(float(bbox['xmax'])),int(float(bbox['ymax']))),
    #                   (0,0,255))
    # cv2.imshow('aa', img)






    return data_example


def _convert_to_example(data_example):
    """Build an Example proto for an image example.

    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    shape= []
    for s in data_example['shape']:
        shape.append(s)
    #print(shape)
    image_data = data_example['image']
    filename = data_example['name']
    for bbox in data_example['bboxes']:
        assert len(bbox) == 4
        # pylint: disable=expression-not-assigned
        xmin.append(bbox['xmin']/shape[1])
        ymin.append(bbox['ymin']/shape[0])
        xmax.append(bbox['xmax']/shape[1])
        ymax.append(bbox['ymax']/shape[0])
        # pylint: enable=expression-not-assigned

    #print(xmin)
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/format': bytes_feature(image_format),
            'image/filename': bytes_feature(filename.encode('utf-8')),
            'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(info,dataset_path, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    data_example = \
        _process_image(info,dataset_path)
    example = _convert_to_example(data_example)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)

def _get_dataset_filename(output_dir, name, shard_id, num_shard, records_num):
    output_filename = '%s_%05d-of-%05d-total%05d.tfrecord' % (name, shard_id + 1, num_shard,records_num)
    return os.path.join(output_dir, output_filename)

def run(anno_path,dataset_path, output_dir, name, shuffling=False):


    # Dataset filenames, and shuffling.

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    annotations = get_annotations(anno_path)

    if shuffling:
        random.seed(12345)
        random.shuffle(annotations)

    # Process dataset files.
    num_per_shard = 200
    num_shard = int(math.ceil(len(annotations) / float(num_per_shard)))
    
    for shard_id in range(num_shard):
        start_ndx = shard_id * num_per_shard
        end_ndx = min((shard_id+1) * num_per_shard, len(annotations))
        records_num = end_ndx - start_ndx
        tf_filename = _get_dataset_filename(output_dir, name, shard_id, num_shard, records_num)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            for i in range(start_ndx, end_ndx):
                info = annotations[i].split(' ')
                filename = os.path.join(dataset_path, info[0] + '.jpg')


                _add_to_tfrecord(info, dataset_path,tfrecord_writer)


                
        print('Converting shard %d' % ( shard_id+1))
            #save the file to tfrecords

        # print(filename)
        # img = cv2.imread(filename + '.jpg')
        # cv2.imshow(filename, img)
        # cv2.waitKey(0)
        #




    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the Widerface dataset!')
    
def get_annotations(anno_path):
    anno_file = open(anno_path,'r')
    annotations = anno_file.readlines()
    return annotations




if __name__ == "__main__": 
    
    dataset_dir = '../../DATA/WIDER_train/images/'
    anno_file = 'wider_face_train.txt'
    output_dir = "../../DATA/wider_tfrecord/"
    name='Wider_Face'

    #get_filenames(anno_file,dataset_dir)
    
    run(anno_file,dataset_dir, output_dir, name=name, shuffling=False)
    
    
    
    
    
