"""Small Keras 3 models for the worker's TensorFlow preview tests (test fixtures, no real names)."""

from __future__ import annotations

import keras

VARIANTS = {"m900001": {"model": "dropout_mlp"}, "m900002": {"model": "cnn"},
            "m900003": {"model": "function"}, "m900004": {"model": "softmax"},
            "m900005": {"model": "collapsed_head"}}


def build(cfg: dict):
    if cfg.get("model") == "function":  # a raw function over constants: no variables at all
        import tensorflow as tf

        w = tf.constant([[0.5, -1.0, 0.25, 2.0]] * 8)
        return lambda x: tf.nn.relu(tf.matmul(x, w))
    if cfg.get("model") == "collapsed_head":  # a zero-initialized head: the output is constant
        return keras.Sequential([keras.Input((8,)), keras.layers.Dense(16, activation="relu"),
                                 keras.layers.Dense(4, kernel_initializer="zeros")])
    if cfg.get("model") == "softmax":  # every output row sums to one, whatever the input
        return keras.Sequential([keras.Input((8,)), keras.layers.Dense(4),
                                 keras.layers.Softmax()])
    if cfg.get("model") == "cnn":  # image:nhwc input, 1x32x32x3
        inputs = keras.Input((32, 32, 3))
        x = keras.layers.Conv2D(8, 3, padding="same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        return keras.Model(inputs, keras.layers.Dense(5)(x))
    return keras.Sequential([keras.Input((8,)), keras.layers.Dense(16, activation="relu"),
                             keras.layers.Dropout(0.5), keras.layers.Dense(4)])
