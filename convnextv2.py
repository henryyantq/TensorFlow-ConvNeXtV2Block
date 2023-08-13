import tensorflow as tf
from keras import layers, activations
from normalization import GlobalResponseNormalization

def ConvNeXtV2Block(input_layer, input_channels=96, hidden_channels=384):
    l = layers.SeparableConv2D(filters=input_channels, kernel_size=7, padding="same")(input_layer)
    l = layers.LayerNormalization()(l)

    l = layers.Conv1D(filters=hidden_channels, kernel_size=1, padding="same")(l)
    l = layers.Activation(activations.gelu)(l)
    l = GlobalResponseNormalization()(l)

    l = layers.Conv2D(filters=input_channels, kernel_size=1, use_bias=False)(l)

    l = layers.Add()([input_layer, l])

    return l
    