import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append('/home/share/models/slim')

from ..models.datasets import dataset_utils
import tensorflow as tf
url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"