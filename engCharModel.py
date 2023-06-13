import tensorflow as tf

from engCharLayer import EngCharLayer

class EngCharModel(tf.keras.Model):
  def __init__(self):
    super(EngCharModel, self).__init__()
    self.blocks = []

    self.conv = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding="same")
    self.activation = tf.keras.layers.ReLU()

    self.blocks.append(EngCharLayer(24))
    self.blocks.append(EngCharLayer(28))
    self.blocks.append(EngCharLayer(32, strides=2))
    self.blocks.append(EngCharLayer(36))

    self.blocks.append(EngCharLayer(42))
    self.blocks.append(EngCharLayer(48, strides=2))
    self.blocks.append(EngCharLayer(56))
    self.blocks.append(EngCharLayer(64))

    self.blocks.append(EngCharLayer(72))
    self.blocks.append(EngCharLayer(80))
    self.blocks.append(EngCharLayer(88))
    self.blocks.append(EngCharLayer(96, strides=2))

    self.blocks.append(EngCharLayer(96))
    self.blocks.append(EngCharLayer(96))
    self.blocks.append(EngCharLayer(96))

    self.flatten = tf.keras.layers.Flatten()
    self.relu = tf.keras.layers.Dense(128, activation="relu")
    self.sigmoid = tf.keras.layers.Dense(64, activation="sigmoid")
    self.softmax = tf.keras.layers.Dense(62, activation="softmax")
  
  def call(self, x):
    B, H, W, C = x.shape

    x = self.conv(x)
    x = self.activation(x) # (B, 64, 64, 24)

    for i in range(0, len(self.blocks)):
      x = self.blocks[i](x)

    x = self.flatten(x)
    x = self.relu(x)
    x = self.sigmoid(x)
    x = self.softmax(x)

    return x