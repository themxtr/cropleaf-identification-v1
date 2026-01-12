# test.py
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ----------------------------------------------------
# Register custom layers
# ----------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

@tf.keras.utils.register_keras_serializable()
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------
IMG_SIZE = (224, 224)

CLASSES = sorted([p.name for p in Path("crops/train").iterdir() if p.is_dir()])

def load_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    return arr  # return raw image for plotting AND prediction
    


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    MODEL_PATH = "models/classifier.keras"
    TEST_DIR = Path("crops/test")

    if not os.path.exists(MODEL_PATH):
        print("[!] Model not found")
        return

    tf.keras.config.enable_unsafe_deserialization()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "MobileNetPreprocess": MobileNetPreprocess,
            "L2Normalize": L2Normalize
        }
    )

    files = sorted([p for p in TEST_DIR.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    if not files:
        print("[!] No test images found.")
        return

    images = []
    titles = []

    for path in files:
        arr = load_image(path)

        pred = model.predict(np.expand_dims(arr, axis=0), verbose=0)
        idx = int(np.argmax(pred))
        conf = float(pred[0][idx]) * 100

        images.append(arr)
        titles.append(f"{path.name}\n{CLASSES[idx]} ({conf:.1f}%)")

    # ------------- Matplotlib Grid Display -------------
    n = len(images)
    cols = 4
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(16, rows * 4))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
#to run use command python test.py --model models/classifier.keras --test_dir crops/test