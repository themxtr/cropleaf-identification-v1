#test_visualise.py
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

# -------------------------------------------------------
# Register custom layers so model loads safely
# -------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, x):
        return tf.math.l2_normalize(x, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


# -------------------------------------------------------
# Dataset loader
# -------------------------------------------------------

IMG_SIZE = (224, 224)

def load_dataset(path, class_names):
    images = []
    labels = []

    for idx, cls in enumerate(class_names):
        folder = os.path.join(path, cls)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = tf.keras.preprocessing.image.load_img(fpath, target_size=IMG_SIZE)
            arr = tf.keras.preprocessing.image.img_to_array(img)
            arr = arr.astype("float32") / 255.0
            images.append(arr)
            labels.append(idx)

    return np.array(images), np.array(labels)


# -------------------------------------------------------
# Plot Confusion Matrix
# -------------------------------------------------------

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Print numbers on cells
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    # Load class names from folder
    class_names = sorted(os.listdir(args.data))

    print(f"[i] Loading model: {args.model}")

    tf.keras.config.enable_unsafe_deserialization()

    model = tf.keras.models.load_model(
        args.model,
        custom_objects={
            "MobileNetPreprocess": MobileNetPreprocess,
            "L2Normalize": L2Normalize
        }
    )

    print("[i] Loading dataset…")
    X, y_true = load_dataset(args.data, class_names)

    print("[i] Running predictions…")
    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Metrics
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n=== Classification Report ===\n")
    print(report)

    # Plot CM
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()
#run by: python test_visualise.py --model models/classifier.keras --data crops/val