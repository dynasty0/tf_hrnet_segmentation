import argparse

parser = argparse.ArgumentParser()

parser.description='''
python trans_tools/trans2pb.py \
--cfg_path = ./cfg/w16_s1.cfg \
--ckpt_path = ./train_record/w16_s1/models/save/model.ckpt-130000 \
--pb_path = ./graph.pb
'''

parser.add_argument("--cfg_path", help="cfg file path", type=str)
parser.add_argument("--ckpt_path", help="ckpt file path", type=str)
parser.add_argument("--pb_path", help="out pb file path", type=str)

args = parser.parse_args()

import tensorflow as tf
from net.model import HRNet

cfg_path = args.cfg_path
pb_file_path = args.pb_path
input = tf.placeholder(tf.float32, shape=(1,512,512,3), name = 'input')
model = HRNet(cfg_path)
with tf.variable_scope('HRNET'):
    output = model.forward_eval(input)

output = tf.image.resize_images(output, (512,512))
output = tf.math.argmax(output,-1)
out = tf.identity(output, name="output")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, args.ckpt_path)
    frozen_graphdef = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['output'])
    with tf.gfile.FastGFile(pb_file_path, mode = 'wb') as f:
        f.write(frozen_graphdef.SerializeToString())
