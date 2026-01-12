# sanity_check.py
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Adjust these paths
sample_img = None
# try to find an image in your train folder automatically
root = "crops/train"
for cls in os.listdir(root):
    clsdir = os.path.join(root, cls)
    for f in os.listdir(clsdir):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            sample_img = os.path.join(clsdir, f)
            break
    if sample_img:
        break

if not sample_img:
    raise SystemExit("No sample image found in crops/train. Put one and retry.")

print("Using sample:", sample_img)

# load image with same pipeline as dataset.py (0-1 floats)
img = tf.io.read_file(sample_img)
img = tf.image.decode_image(img, channels=3, expand_animations=False)
img = tf.image.convert_image_dtype(img, tf.float32)   # 0-1
img = tf.image.resize(img, (224,224))
img_np = img.numpy()
print("Dataset pipeline -> min/max:", float(img_np.min()), float(img_np.max()), "dtype:", img_np.dtype)

# Test MobileNet preprocess variations
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
p1 = preprocess_input(img_np)            # if you used preprocess_input(inputs)
p2 = preprocess_input(img_np * 255.0)    # correct usage for 0-1 inputs

print("preprocess_input(img)    min/max:", float(p1.min()), float(p1.max()))
print("preprocess_input(img*255) min/max:", float(p2.min()), float(p2.max()))

# quick forward pass through a fresh MobileNetB2 base to see activations
base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')
base.trainable = False

# expand dims
p2_batch = tf.expand_dims(p2, 0)
out = base(p2_batch, training=False)
print("Base output shape (from correct preprocess):", out.shape)
