# Tencent is pleased to support the open source community by making PocketFlow available.
#
# Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""celeba dataset."""

import os
import tensorflow as tf

from datasets.abstract_dataset import AbstractDataset
from utils.imagenet_preprocessing import preprocess_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('nb_classes', 19, '# of classes')
tf.app.flags.DEFINE_integer('nb_smpls_train', 75000, '# of samples for training')
tf.app.flags.DEFINE_integer('nb_smpls_val', 5838, '# of samples for validation')
tf.app.flags.DEFINE_integer('nb_smpls_eval', 9162, '# of samples for evaluation')
tf.app.flags.DEFINE_integer('batch_size', 8, 'batch size per GPU for training')
tf.app.flags.DEFINE_integer('batch_size_eval', 8, 'batch size for evaluation')

# celeba specifications
IMAGE_HEI = 512
IMAGE_WID = 512
IMAGE_CHN = 3

def parse_fn(record, is_train):
    
    def _decode_image(content, channels):
        return tf.cond(
            tf.image.is_jpeg(content),
            lambda: tf.image.decode_jpeg(content, channels, dct_method='INTEGER_ACCURATE'),
            lambda: tf.image.decode_png(content, channels)
        )
    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png')
    }
    parsed_features = tf.parse_single_example(record, features)
    image = _decode_image(parsed_features['image/encoded'], channels=3)
    label = _decode_image(parsed_features['image/segmentation/class/encoded'], channels=1)

    image = tf.reshape(image, [IMAGE_HEI, IMAGE_WID, 3])
    image = tf.cast(image, tf.float32) * (2. / 255) - 1
    label = tf.reshape(label, [IMAGE_HEI, IMAGE_WID, 1])
    return image, label

class CelebaDataset(AbstractDataset):
  '''Celeba dataset.'''

  def __init__(self, is_train, data_dir):
    """Constructor function.

    Args:
    * is_train: whether to construct the training subset
    """

    # initialize the base class
    super(CelebaDataset, self).__init__(is_train)

    if not os.path.exists(data_dir):
      raise ValueError('data dir does not exist: ' + data_dir)

    # configure file patterns & function handlers
    if is_train:
      self.file_pattern = os.path.join(data_dir, 'train-*-of-*')
      self.batch_size = FLAGS.batch_size
    else:
      self.file_pattern = os.path.join(data_dir, 'val-*-of-*')
      self.batch_size = FLAGS.batch_size_eval
    self.dataset_fn = tf.data.TFRecordDataset
    self.parse_fn = lambda x: parse_fn(x, is_train=is_train)