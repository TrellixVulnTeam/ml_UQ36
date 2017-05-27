# Whatever you do, don't uncomment the next line!!!
#import antigravity  # the most important module

import numpy as np
import cv2
import av
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import os, sys
import time

slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/home/share/models/slim')

# Constants
CHECKPOINTS_DIR = '/home/share/checkpoints'
CHECKPOINT_FILE = 'inception_resnet_v2_2016_08_30.ckpt'
SAMPLE_VIDEO = '/home/share/yakinet/data/creech-orchard.mp4'
IMAGE_SIZE = 299
NUM_CLASSES = 1

DEBUG_PYAV_READ = True

# Read in sample video
container = av.open(SAMPLE_VIDEO)
for frame in container.decode(video=0):
    img = frame.to_image()  # PIL/Pillow image
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    if DEBUG_PYAV_READ:
        cv2.imshow('frame', arr)
        cv2.waitKey(100)
    # Infer labels from a full convolution

    # Draw a box with alpha channel as strength
    # Write out video frame