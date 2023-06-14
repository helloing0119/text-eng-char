import tensorflow as tf
from tensorflow.keras import layers

class EngCharLayer(tf.keras.Model):
  def __init__(self, filters, size=3):
    super(EngCharLayer, self).__init__()
    self.filters = filters
    self.size = size

    self.max_pool = layers.MaxPool2D((2, 2), padding="same")
    self.conv = layers.Conv2D(filters, (size, size), padding="same")

    self.activation = layers.ReLU()
  
  def call(self, x_input):
    x = self.conv(x_input)
    x = self.activation(x)
    x = self.max_pool(x)

    return x