import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

# ------------------------------------------------------------
# Import TF-compatible decorator
# ------------------------------------------------------------
from keras.saving import register_keras_serializable

# ------------------------------------------------------------
# Custom layer: MobileNetPreprocess
# ------------------------------------------------------------
@register_keras_serializable(package="Custom")
class MobileNetPreprocess(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs  # identity

# ------------------------------------------------------------
# Custom layer: L2Normalize
# ------------------------------------------------------------
@register_keras_serializable(package="Custom")
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

# ------------------------------------------------------------
# Default paths
# ------------------------------------------------------------
DEFAULT_MODEL = "models/classifier.keras"
DEFAULT_CALIB_DIR = "crops/train"
OUT_FLOAT = "model_float32.tflite"
OUT_INT8 = "models/crop.tflite"

def representative_data_gen(calib_dir, img_size=(224, 224), max_samples=200):
    p = Path(calib_dir)
    imgs = []

    for cls in p.iterdir():
        if cls.is_dir():
            imgs.extend(list(cls.glob("*.jpg")) + list(cls.glob("*.png")))

    imgs = imgs[:max_samples]

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, img_size).astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]


def convert():
    print("\n============================")
    print("   TFLITE CONVERSION START  ")
    print("============================\n")

    model_path = DEFAULT_MODEL
    calib_dir = DEFAULT_CALIB_DIR

    if not Path(model_path).exists():
        print("[ERROR] Model not found →", model_path)
        return

    print("[INFO] Loading model:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # FLOAT32
    print("[INFO] Converting FLOAT32...")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = conv.convert()
    with open(OUT_FLOAT, "wb") as f:
        f.write(tflite_model)
    print("[✓] Saved:", OUT_FLOAT)

    # INT8
    if not Path(calib_dir).exists():
        print("[WARNING] No calibration directory → Skipping INT8.")
        return

    print("[INFO] Converting INT8...")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_data_gen(calib_dir)
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8

    tflite_quant = conv.convert()
    with open(OUT_INT8, "wb") as f:
        f.write(tflite_quant)

    print("[✓] Saved:", OUT_INT8)
    print("\n[DONE] Conversion complete!\n")


if __name__ == "__main__":
    convert()
