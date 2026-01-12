#!/usr/bin/env python3
"""
main.py - Unified training, evaluation, prediction pipeline.
AUTO MODE triggers when:
1) python main.py     (no arguments)
2) python run main.py (run file sets AUTO_MODE = True)
"""

from __future__ import annotations
import argparse
import logging
import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, Model
import sys

# AUTO MODE global
AUTO_MODE = False

# ---------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("leaf_classifier")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------
# CUSTOM LAYERS
# ---------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    def get_config(self): return super().get_config()

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

# ---------------------------------------------------------
# DATASET PIPELINE
# ---------------------------------------------------------
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def build_datasets(train_dir, val_dir, batch=8, seed=42):
    train_dir, val_dir = Path(train_dir), Path(val_dir)
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    logger.info("Classes: %s", class_names)

    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch,
        shuffle=True, seed=seed, label_mode="int",
        class_names=class_names, follow_links=True
    )
    val = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=batch,
        shuffle=False, label_mode="int",
        class_names=class_names, follow_links=True
    )

    train = train.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
    val = val.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

    aug = tf.keras.Sequential([
        tf.keras.layers.Resizing(250,250),
        tf.keras.layers.RandomCrop(224,224),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.20),
        tf.keras.layers.RandomContrast(0.20),
    ])

    train = train.map(lambda x, y: (aug(x, training=True), y))
    train = train.cache().prefetch(AUTOTUNE)
    val = val.cache().prefetch(AUTOTUNE)

    return train, val, class_names

# ---------------------------------------------------------
# MODEL CREATION
# ---------------------------------------------------------
def build_embedding_model(input_shape=(224,224,3), embedding_dim=256):
    inp = layers.Input(shape=input_shape)
    x = MobileNetPreprocess()(inp)

    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.35)(x)

    emb = layers.Dense(embedding_dim)(x)
    emb = L2Normalize()(emb)

    return Model(inp, emb)

def build_classifier(emb_model, num_classes):
    inp = emb_model.input
    emb = emb_model.output
    x = layers.Dense(256, activation="relu")(emb)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def compute_class_weights(train_dir):
    train_dir = Path(train_dir)
    counts = [sum(1 for _ in p.rglob("*.*")) for p in sorted(train_dir.iterdir()) if p.is_dir()]
    total = sum(counts)
    return {i: total / (len(counts) * c) for i, c in enumerate(counts)}

def train_model(train_dir, val_dir, models_dir="models"):
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    train_ds, val_ds, class_names = build_datasets(train_dir, val_dir)
    num_classes = len(class_names)
    class_weights = compute_class_weights(train_dir)

    emb = build_embedding_model()
    clf = build_classifier(emb, num_classes)

    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    best_model = models_dir / "best_classifier.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_model), save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    ]

    clf.fit(train_ds, validation_data=val_ds,
            epochs=8, class_weight=class_weights, callbacks=callbacks)

    # Fine tune backbone
    for layer in emb.layers[-40:]:
        layer.trainable = True

    clf.compile(optimizer=tf.keras.optimizers.Adam(5e-6),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    clf.fit(train_ds, validation_data=val_ds,
            epochs=15, class_weight=class_weights, callbacks=callbacks)

    clf.save(str(models_dir / "classifier.keras"), include_optimizer=False)

    emb_dir = models_dir / "embedding_model"
    if emb_dir.exists(): shutil.rmtree(emb_dir)
    emb.save(str(emb_dir))

    logger.info("Training completed.")
    return class_names

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
def load_model_safe(path):
    tf.keras.config.enable_unsafe_deserialization()
    return tf.keras.models.load_model(path, custom_objects={
        "MobileNetPreprocess": MobileNetPreprocess,
        "L2Normalize": L2Normalize
    })

def evaluate(model_path, data_dir):
    classes = sorted([p.name for p in Path(data_dir).iterdir() if p.is_dir()])
    model = load_model_safe(model_path)

    X, y = [], []
    for i, cls in enumerate(classes):
        for f in (Path(data_dir)/cls).glob("*.*"):
            arr = tf.keras.preprocessing.image.load_img(f, target_size=IMG_SIZE)
            arr = tf.keras.preprocessing.image.img_to_array(arr)/255.0
            X.append(arr); y.append(i)

    X, y = np.array(X), np.array(y)
    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=classes))

    cm = confusion_matrix(y, y_pred)
    plt.imshow(cm, cmap="viridis")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
def predict_folder(model_path, test_dir):
    model = load_model_safe(model_path)
    classes = sorted([p.name for p in Path("crops/train").iterdir() if p.is_dir()])

    for f in sorted(Path(test_dir).glob("*.*")):
        arr = tf.keras.preprocessing.image.load_img(f, target_size=IMG_SIZE)
        arr = tf.keras.preprocessing.image.img_to_array(arr)/255.0
        pred = model.predict(np.expand_dims(arr, 0), verbose=0)
        idx = np.argmax(pred)
        conf = pred[0][idx] * 100
        print(f"{f.name} -> {classes[idx]} ({conf:.1f}%)")

# ---------------------------------------------------------
# MAIN + AUTO MODE
# ---------------------------------------------------------
def main():
    global AUTO_MODE

    # 1) AUTO MODE part 1 → triggered when user runs: python main.py
    if len(sys.argv) == 1:
        print("\n[ AUTO MODE ENABLED ] - No arguments provided")
        AUTO_MODE = True

    # 2) AUTO MODE part 2 → if enabled by run file
    if AUTO_MODE:
        print("\n========== AUTO PIPELINE ==========")
        print("Running: TRAIN → EVAL → PREDICT\n")

        train_model("crops/train", "crops/val")
        evaluate("models/best_classifier.keras", "crops/val")
        predict_folder("models/classifier.keras", "crops/test")
        return

    # Otherwise → CLI mode
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train_dir", required=True)
    t.add_argument("--val_dir", required=True)

    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--data_dir", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--test_dir", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train_model(args.train_dir, args.val_dir)
    elif args.cmd == "eval":
        evaluate(args.model, args.data_dir)
    elif args.cmd == "predict":
        predict_folder(args.model, args.test_dir)

if __name__ == "__main__":
    main()
