import tensorflow as tf

from engCharLayer import EngCharLayer

class EngCharModel(tf.keras.Model):
  def __init__(self):
    super(EngCharModel, self).__init__()
    self.blocks = []
    self.downscales = []

    # enusre input shape == 64, 64, 1
    self.begin_shape = tf.keras.layers.Reshape((64, 64, 1), input_shape=(64, 64, 1))

    # MaxPool & Convolution
    # detect distinctive characteristics first,
    # then detect minor characteristics
    # self.blocks.append(EngCharLayer(24, size=2))
    # self.blocks.append(EngCharLayer(32, size=2))
    # self.blocks.append(EngCharLayer(40, size=2))
    # self.blocks.append(EngCharLayer(48, size=2))
    # self.blocks.append(EngCharLayer(96))
    # self.blocks.append(EngCharLayer(112))
    # self.blocks.append(EngCharLayer(128))
    # self.blocks.append(EngCharLayer(256, size=4))
    # self.blocks.append(EngCharLayer(512, size=5))


    self.blocks.append(EngCharLayer(512, size=5))
    self.blocks.append(EngCharLayer(256))
    self.blocks.append(EngCharLayer(256))


    # Flatten & Downscale to calculate label (classification)
    self.flatten = tf.keras.layers.Flatten()

    # for i in range(6, 3, 1):
    #   self.downscales.append(tf.keras.layers.Dense(256+2**i, activation="relu"))

    self.downscales.append(tf.keras.layers.Dense(512, activation="relu"))
    self.downscales.append(tf.keras.layers.Dense(1024, activation="relu"))
    self.downscales.append(tf.keras.layers.Dense(512, activation="relu"))
    
    self.sigmoid = tf.keras.layers.Dense(256, activation="sigmoid")
    self.softmax = tf.keras.layers.Dense(62, activation="softmax")
    self.result = tf.keras.layers.Reshape((62,))

  def call(self, x):
    x = self.begin_shape(x)
    for i in range(0, len(self.blocks)):
      x = self.blocks[i](x)

    x = self.flatten(x)
    for i in range(0, len(self.downscales)):
      x = self.downscales[i](x)
    # x = tf.keras.layers.Dense(512, activation="relu")(x)
    # x = tf.keras.layers.Dense(1024, activation="relu")(x)
    # x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = self.sigmoid(x)
    x = self.softmax(x)
    x = self.result(x)

    return x