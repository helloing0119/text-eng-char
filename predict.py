import argparse
import tensorflow as tf
import numpy as np

from dataGenerator import DataGenerator
from engCharModel import EngCharModel

def generate_image_tensor(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [64, 64])
    return img / 255.0

def get_label(num):
    if num<10:
        return str(num)
    if num<36:
        return chr(65+num- 10)
    return chr(97 + num - 36)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  '''
  Init configuration
  '''
  # Hyper parmas & customizable arugments
  parser.add_argument('--img-path', default="./test_dataset/Sample050/img050-00009.png", help='img path to predict with model')
  parser.add_argument('--model-path', default="./models", help='model directory')
  parser.add_argument('--hw-accelerator', default="cpu", help='hw_accelerator cpu or gpu' )

  opts = parser.parse_args()

  '''
  Init params with arguments
  '''
  hw_accelerator = opts.hw_accelerator
  img_path = opts.img_path
  model_path = opts.model_path

  '''
  Predict 
  '''
  image_tensor = generate_image_tensor(img_path)
  image_input = tf.data.Dataset.from_tensors(image_tensor).batch(1)
  model = tf.keras.models.load_model(model_path)
  predictions = model.predict(image_input, batch_size=None)
  predicted_label = np.argmax(predictions, axis=1)

  if len(predicted_label) > 0:
    predicted_label = sorted(predicted_label, reverse=True)
    print("predicted label : ", predicted_label[0], "=", get_label(predicted_label[0]))
  else:
    print("NO LABEL")