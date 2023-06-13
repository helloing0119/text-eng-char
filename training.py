import os
import random
import time
import argparse
import tensorflow as tf

from dataGenerator import DataGenerator
from engCharModel import EngCharModel

def train(epochs, train_dataset, optimizer):
  for epoch in range(epochs):
    print("begin learning")
    epoch_time = time.time()
    step = 0
    epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_dataset, epochs=1)
    
    eval_loss, eval_accuracy = model.evaluate(train_dataset, verbose=0)
    
    print('\r', 'Epoch', epoch_count,
          '| Loss:', f"{eval_loss:.4f}", '| Accuracy:', f"{eval_accuracy:.4f}",
          '| Epoch Time:', f"{time.time() - epoch_time:.2f}") 

def fit_param(param, min, max):
  if param > max:
    return max
  if param < min:
    return min
  return param
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  '''
  Init configuration
  '''
  # Hyper parmas & customizable arugments
  parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
  parser.add_argument('--use-random-seed', default=False, help='use random seed for generate dataset')
  parser.add_argument('--seed', type=int, default=777, help='seed for generate dataset, postivie value')
  parser.add_argument('--learning-rate', type=float, default=0.001, help='0.00001 < learning-rate < 0.1')
  parser.add_argument('--save-path', default="./models/result", help='file path to save model')
  parser.add_argument('--hw-accelerator', default="cpu", help='hw_accelerator cpu or gpu' )

  # Customizable arugments, but not recommanded
  parser.add_argument('--csv-path', default="./text_dataset/english.csv", help='file path to label csv')
  parser.add_argument('--img-dir', default="./text_dataset/", help='column name of imgae in csv')
  parser.add_argument('--csv-img-col', default="image", help='column name of imgae in csv')
  parser.add_argument('--csv-label-col', default="label", help='column name of label in csv')
  parser.add_argument('--num-classes', type=int, default=62, help='num of classes for label')

  opts = parser.parse_args()

  '''
  Init params with arguments
  '''
  epochs = opts.epochs
  use_random_seed = opts.use_random_seed
  seed = fit_param(random.randint(1, 100) if use_random_seed else opts.seed, 0, 1000)
  learning_rate = fit_param(opts.learning_rate, 0.0001, 0.1)
  save_path = opts.save_path
  hw_accelerator = opts.hw_accelerator

  csv_path = opts.csv_path
  img_dir = opts.img_dir
  csv_img_col = opts.csv_img_col
  csv_label_col = opts.csv_label_col
  num_classes = opts.num_classes

  gpu_available = tf.test.is_gpu_available()

  if gpu_available or hw_accelerator=="gpu":
      tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
      logical_gpus = tf.config.list_logical_devices('GPU')
  else:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
      tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')

  '''
  Train 
  '''
  generator = DataGenerator(
    csv_path=csv_path, csv_img_col=csv_img_col, csv_label_col=csv_label_col,
    img_dir=img_dir, num_classes=num_classes, seed=seed)
  
  dataset = generator.generate_dataset()
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model = EngCharModel()

  train(epochs, dataset, optimizer)
  model.save(save_path, include_optimizer=False)