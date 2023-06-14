import tensorflow as tf
from tensorflow.keras import layers

class EngCharLayer(tf.keras.Model):
  def __init__(self, filters):
    super(EngCharLayer, self).__init__()
    self.filters = filters

    self.max_pool = layers.MaxPool2D((2, 2), padding="same")
    self.conv = layers.Conv2D(filters, (3, 3), padding="same")

    self.activation = layers.ReLU()
  
  def call(self, x_input):
    x = self.conv(x_input)
    x = self.max_pool(x)
    
    x = self.activation(x)

    return x