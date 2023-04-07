import argparse

parser = argparse.ArgumentParser()

parser.description='''
python trans_tools/trans2quant.py \
--cfg_path = ./cfg/w16_s1.cfg \
--ckpt_path = ./train_record/w16_s1/models/save/model.ckpt-130000 \
--eval_graph_path = ./graph.pb
'''

parser.add_argument("--cfg_path", help="cfg file path", type=str)
parser.add_argument("--ckpt_path", help="ckpt file path", type=str)
parser.add_argument("--eval_graph_path", help="out pb file path", type=str)

args = parser.parse_args()

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

from net.model import HRNet

cfg_path = args.cfg_path
eval_graph_file = args.eval_graph_path

input = tf.placeholder(tf.float32, shape=(1,512,512,3), name = "input")
model = HRNet(cfg_path)
with tf.variable_scope('HRNET'):
    output = model.forward_eval(input)

output = tf.image.resize_images(output, (512,512) )
out = tf.identity(output, name = "output")

g = tf.get_default_graph()
with tf.variable_scope('HRNET'):
    tf.contrib.quantize.create_eval_graph(input_graph = g)
saver = tf.train.Saver(tf.all_variables())
graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
freeze_graph.freeze_graph_with_def_protos(
    graph_def,
    saver.as_saver_def(),
    args.ckpt_path,
    'output',
    restore_op_name = None,
    filename_tensor_name = None,
    output_graph = args.eval_graph_path,
    clear_devices=True,
    initializer_nodes=None
)
