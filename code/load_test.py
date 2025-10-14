# for testing streamlit

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODEL_PATH = "/Users/alfred/Documents/GitHub/DDLS_project/code/models/baseline/model.keras"

print("== Versions ==")
try:
    import tensorflow as tf
    print("tensorflow:", tf.__version__)
except Exception as e:
    print("tensorflow import FAILED:", e)

try:
    import keras
    print("keras:", keras.__version__)
except Exception as e:
    print("keras import FAILED:", e)

print("\n== Try keras.models.load_model ==")
try:
    from keras.models import load_model as kload
    m = kload(MODEL_PATH, compile=False)
    print("KERAS loader OK. input_shape:", getattr(m, "input_shape", None))
except Exception as e:
    print("KERAS loader FAILED:", e)

print("\n== Try tf.keras.models.load_model ==")
try:
    m2 = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("TF.KERAS loader OK. input_shape:", getattr(m2, "input_shape", None))
except Exception as e:
    print("TF.KERAS loader FAILED:", e)
