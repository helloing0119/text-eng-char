import tensorflow as tf

from engCharLayer import EngCharLayer

class EngCharModel(tf.keras.Model):
  def __init__(self):
    super(EngCharModel, self).__init__()
    self.blocks = []
    self.downscales = []

    self.conv = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(128, 128, 1, 1), padding="same")
    self.activation = tf.keras.layers.ReLU()

    self.blocks.append(EngCharLayer(42))
    self.blocks.append(EngCharLayer(48))
    self.blocks.append(EngCharLayer(56))
    self.blocks.append(EngCharLayer(64))

    self.blocks.append(EngCharLayer(72))
    self.blocks.append(EngCharLayer(80))
    self.blocks.append(EngCharLayer(88))
    self.blocks.append(EngCharLayer(96))

    self.blocks.append(EngCharLayer(96))
    self.blocks.append(EngCharLayer(96))
    self.blocks.append(EngCharLayer(96))

    for i in range(10, 6):
      self.downscales.append(tf.keras.layers.Dense(2**i, activation="relu"))

    self.flatten = tf.keras.layers.Flatten()
    self.sigmoid = tf.keras.layers.Dense(64, activation="sigmoid")
    self.softmax = tf.keras.layers.Dense(62, activation="softmax")
    self.result = tf.keras.layers.Reshape((62,))

  def call(self, x):
    x = self.conv(x)
    x = self.activation(x)

    for i in range(0, len(self.blocks)):
      x = self.blocks[i](x)

    x = self.flatten(x)
    for i in range(0, len(self.downscales)):
      x = self.downscales[i](x)
    x = self.sigmoid(x)
    x = self.softmax(x)
    x = self.result(x)    # B, 62, 1

    return x