import tensorflow as tf
from keras import layers

class GlobalResponseNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer='zeros', trainable=True)

    def call(self, X):
        gx = tf.norm(X, ord=2, axis=(1,2), keepdims=True)
        nx = gx / (tf.reduce_mean(gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (X * nx) + self.beta + X