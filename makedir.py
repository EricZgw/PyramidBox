import os
if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')
if not os.path.isdir('model'):
    os.mkdir('model')
if not os.path.isdir('tfrecords'):
    os.mkdir('tfrecords')
if not os.path.isdir('datasets/widerface'):
    os.mkdir('datasets/widerface')
