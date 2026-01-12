# dataset.py
import tensorflow as tf
from pathlib import Path

IMG_SIZE = (224, 224)

def build_datasets(train_dir, val_dir, batch=32, shuffle=True, seed=42):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch,
        label_mode="int",
        shuffle=shuffle,
        seed=seed
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=batch,
        label_mode="int",
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    # Stronger augmentations to fix confusion matrix issues
    data_augment = tf.keras.Sequential([
        tf.keras.layers.Resizing(250, 250),         # Random-resize base
        tf.keras.layers.RandomCrop(224, 224),       # Random crop (VERY powerful)
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12),
        tf.keras.layers.RandomZoom(0.20),
        tf.keras.layers.RandomContrast(0.20),
        tf.keras.layers.RandomBrightness(0.20)
    ], name="data_augmentation")

    def augment(images, labels):
        return data_augment(images, training=True), labels

    # Convert to float and augment
    train_ds = train_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE
    )
    train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
        num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Read class names
    class_names = train_ds.class_names if hasattr(train_ds, "class_names") else None
    if class_names is None:
        class_names = sorted([p.name for p in Path(train_dir).iterdir() if p.is_dir()])

    return train_ds, val_ds, class_names
