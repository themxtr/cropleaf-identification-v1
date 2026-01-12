# model.py
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SHAPE = (224, 224, 3)


class MobileNetPreprocess(layers.Layer):
    """Serializable preprocessing layer"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x = inputs * 255.0
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    def get_config(self):
        return super().get_config()


def build_embedding_model(input_shape=IMG_SHAPE, embedding_dim=256, backbone_trainable=False):
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Preprocess
    x = MobileNetPreprocess(name="mobilenet_preprocess")(inputs)

    # MobileNetV2 Backbone
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        alpha=1.0
    )
    base.trainable = backbone_trainable

    # Embedding head (IMPROVED)
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dense(512, activation=None, name="proj_dense")(x)
    x = layers.BatchNormalization(momentum=0.85, name="proj_bn")(x)
    x = layers.ReLU(name="proj_relu")(x)
    x = layers.Dropout(0.35, name="proj_drop")(x)

    emb = layers.Dense(embedding_dim, activation=None, name="embedding")(x)

    # Safe normalization (no Lambda)
    emb = tf.nn.l2_normalize(emb, axis=1, name="l2norm")

    model = Model(inputs=inputs, outputs=emb, name="leaf_embedding_model")
    return model


def build_classifier_from_embedding(embedding_model, num_classes):
    inp = embedding_model.input
    emb = embedding_model.output

    # Classifier Head (IMPROVED)
    x = layers.Dense(256, activation="relu", name="clf_dense1")(emb)
    x = layers.BatchNormalization(momentum=0.85, name="clf_bn1")(x)
    x = layers.Dropout(0.35, name="clf_drop1")(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inp, outputs=out, name="leaf_classifier")
    return model
