import argparse

parser = argparse.ArgumentParser()

parser.description='''
python trans_tools/run_with_pb.py \
--image_path = /data/CelebAMask-HQ/2.jpg \
--pb_path = ./graph.pb \
--seg_path = ./seg.png
'''

parser.add_argument("--image_path", help="input image file path", type=str)
parser.add_argument("--pb_path", help="pb file path", type=str)
parser.add_argument("--seg_path", default = "./seg.png", help="output image path", type=str)

args = parser.parse_args()


import tensorflow as tf
import cv2
import numpy as np

colormap = np.asarray([
    [0,0,0], [0,0,128], [0,128,0], [0,128,128],[128,0,0], 
    [128,0,128], [128,128,0], [128,128,128], [0,0,64], [0,0,192],
    [0,128,64], [0,128,192], [128, 0, 64], [128,64,0], [128,128,64],
    [128,128,192], [0,192,128], [0,64,128],[0,192,0], [0,64,0],[128,0,192]
])


output_graph_path = args.pb_path

img = cv2.imread(args.image_path)
img = cv2.resize(img, (512,512))
img = img[:,:,::-1]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    output_graph_def = tf.GraphDef()
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    input_x = sess.graph.get_tensor_by_name("input:0")

    output = sess.graph.get_tensor_by_name("output:0")

    res = sess.run(output,feed_dict={input_x:[img]})
    cv2.imwrite(args.seg_path, colormap[output[0,:,:,0]])
