# ONNX bertsio berrien adabakia (Patch) onnx_tf-k huts egin ez dezan
import onnx
import onnx.mapping
onnx.mapping = onnx.mapping  # Lotura indartu
import sys
from onnx import helper
sys.modules['onnx.mapping'] = onnx.mapping

# ORAIN BAI, zure jatorrizko inportazioak:
import numpy as np
import tensorflow as tf
from onnx_tf.backend import prepare

print("Adabakia ongi aplikatu da! Bihurketa hasi da...")

# 1. ONNX eredua kargatu eta TensorFlow SavedModel-era esportatu
onnx_model = onnx.load("output/hola_elmo/hola_elmo_simplified.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tf_model_dir")

# 2. Kalibratze datuak prestatu
calib_data = np.load("embeddings_calib.npy")
def representative_data_gen():
    for i in range(len(calib_data)):
        yield [calib_data[i]]

# 3. Kuantizazio osoa (Full INT8) Coral TPUrako
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model_dir")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTIN_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open("coral_model_dir/fully_quantized_integer_quant.tflite", "wb") as f:
    f.write(tflite_model)

print("Bihurketa amaituta! Fitxategia prest dago konpilatzeko.")
