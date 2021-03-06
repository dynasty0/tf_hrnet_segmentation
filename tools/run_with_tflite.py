import argparse

parser = argparse.ArgumentParser()

parser.description='''
python trans_tools/run_with_lite.py \
--image_path = /data/CelebAMask-HQ/2.jpg \
--lite_path = ./graph.tflite \
--output_path = ./seg.png
'''

parser.add_argument("--image_path", help="input image file path", type=str)
parser.add_argument("--lite_path", help="tflite file path", type=str)
parser.add_argument("--output_path", default = "./seg.png", help="output image path", type=str)

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

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=args.lite_path)

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
all_details = interpreter.get_tensor_details()

input_shape = input_details[0]['shape']
input_data = cv2.imread(args.image_path)
input_data = input_data[:,:,::-1]
input_data = [input_data]

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
cv2.imwrite(args.output_path, colormap[output_data[0,:,:,0]])
