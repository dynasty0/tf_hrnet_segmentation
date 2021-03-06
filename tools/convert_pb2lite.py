import tensorflow as tf
import sys

pb_path = sys.argv[1]
tflite_path = sys.argv[2]

graph_def_file = pb_path
input_arrays = ["input"]
output_arrays = ["output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open(tflite_path, "wb").write(tflite_model)