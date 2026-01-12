# train.py
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import shutil

# ---- Config ----
TRAIN_DIR = "crops/train"
VAL_DIR = "crops/val"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CLS_SAVE = MODELS_DIR / "classifier.keras"
EMB_SAVE_DIR = MODELS_DIR / "embedding_model"

IMG_SIZE = (224, 224)
BATCH = 8

INITIAL_EPOCHS = 8
FINE_TUNE_EPOCHS = 15

LR_HEAD = 1e-4
LR_FINE = 5e-6

EMB_DIM = 256
FINE_TUNE_FROM = -40

# -----------------------------
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

# -----------------------------
# Model builders
# -----------------------------
def build_embedding_model(input_shape=(224,224,3), embedding_dim=256, backbone_trainable=False):
    inp = layers.Input(shape=input_shape)

    x = MobileNetPreprocess()(inp)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        alpha=1.0
    )
    base.trainable = backbone_trainable

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    emb = layers.Dense(embedding_dim)(x)
    emb = L2Normalize()(emb)

    return Model(inp, emb)

def build_classifier_from_embedding(emb_model, num_classes):
    inp = emb_model.input
    emb = emb_model.output

    x = layers.Dense(256, activation="relu")(emb)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out)

# -----------------------------
def compute_class_weights(train_dir):
    class_dirs = sorted([p for p in Path(train_dir).iterdir() if p.is_dir()])
    counts = [sum(1 for _ in p.rglob("*.*")) for p in class_dirs]
    total = sum(counts)
    return {i: total / (len(counts) * c) for i, c in enumerate(counts)}

# -----------------------------
def main():
    tf.keras.utils.set_random_seed(42)

    # IMPORTANT FIX:
    # Automatically handle subfolders inside one class (guava)
    CLASS_NAMES = sorted([p.name for p in Path(TRAIN_DIR).iterdir() if p.is_dir()])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        class_names=CLASS_NAMES,
        label_mode="int",
        follow_links=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        class_names=CLASS_NAMES,
        label_mode="int",
        follow_links=True
    )

    num_classes = len(CLASS_NAMES)
    print("Classes:", CLASS_NAMES)

    # Normalisation
    train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    val_ds = val_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

    # Augmentation
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.06),
        tf.keras.layers.RandomContrast(0.06),
    ])
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    class_weights = compute_class_weights(TRAIN_DIR)

    emb = build_embedding_model(input_shape=IMG_SIZE + (3,), embedding_dim=EMB_DIM)
    clf = build_classifier_from_embedding(emb, num_classes)

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(MODELS_DIR / "best_classifier.keras"),
            save_best_only=True,
            monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=3, factor=0.5),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True)
    ]

    print("\n[i] Stage 1 — Training classifier head")
    clf.fit(train_ds, validation_data=val_ds,
            epochs=INITIAL_EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights)

    print("\n[i] Stage 2 — Fine-tuning MobileNet")
    for layer in emb.layers:
        layer.trainable = False
    for layer in emb.layers[FINE_TUNE_FROM:]:
        layer.trainable = True

    clf.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    clf.fit(train_ds, validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights)

    print("\n[i] Saving classifier...")
    clf.save(str(CLS_SAVE), include_optimizer=False)

    if EMB_SAVE_DIR.exists():
        shutil.rmtree(EMB_SAVE_DIR)

    print("[i] Exporting embedding...")
    emb.export(str(EMB_SAVE_DIR))

if __name__ == "__main__":
    main()
#python 