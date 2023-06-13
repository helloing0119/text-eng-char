import tensorflow as tf
from tensorflow.keras import layers

class EngCharLayer(tf.keras.Model):
  def __init__(self, filters, strides=1):
    super(EngCharLayer, self).__init__()
    self.strides = strides
    self.filters = filters

    if strides == 2:
      self.pool = layers.MaxPool2D()

    self.max_pool = layers.MaxPool2D((2, 2), input_shape=(128, 128, 1, 1), padding="same")
    self.conv = layers.Conv2D(filters, (3, 3), input_shape=(128, 128, 1, 1), padding="same")


    self.activation = layers.ReLU()
  
  def call(self, x_input):
    x = self.conv(x_input)
    x = self.max_pool(x)
    
    x = self.activation(x)

    return x