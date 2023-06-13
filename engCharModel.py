import tensorflow as tf

from engCharLayer import EngCharLayer

class EngCharModel(tf.keras.Model):
  def __init__(self):
    super(EngCharModel, self).__init__()
    self.blocks = []
    self.downscales = []

    self.conv = tf.keras.layers.Conv2D(64, (3, 3), input_shape=(128, 128, 1, 1), padding="same")
    self.activation = tf.keras.layers.ReLU()

    # self.blocks.append(EngCharLayer(24))
    # self.blocks.append(EngCharLayer(28))
    # self.blocks.append(EngCharLayer(32))
    # self.blocks.append(EngCharLayer(36))

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

    self.flatten = tf.keras.layers.Flatten()
    for i in range(10, 7):
      self.downscales.append(tf.keras.layers.Dense(2**i, activation="relu"))
    self.sigmoid = tf.keras.layers.Dense(64, activation="sigmoid")
    self.softmax = tf.keras.layers.Dense(62, activation="softmax")
  
  def call(self, x):
    x = self.conv(x)
    x = self.activation(x)

    for i in range(0, len(self.blocks)):
      x = self.blocks[i](x)

    x = self.flatten(x)
    x = tf.reshape(x, [1,-1])
    for i in range(0, len(self.downscales)):
      x = self.downscales[i](x)
    x = self.sigmoid(x)
    x = self.softmax(x)
    x = tf.reshape(x, [-1,1])

    return x