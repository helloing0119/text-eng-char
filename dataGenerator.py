import random
import tensorflow as tf
import pandas as pd
import os

class DataGenerator:
    def __init__(self, csv_path, csv_img_col, csv_label_col, img_dir, num_classes, seed=None):
        self.configure(csv_path, csv_img_col, csv_label_col, img_dir, num_classes, seed)
    
    def configure(self, csv_path, csv_img_col, csv_label_col, img_dir, num_classes, seed=None):
        self.csv_path = csv_path
        self.csv_img_col = csv_img_col
        self.csv_label_col = csv_label_col
        self.img_dir = img_dir
        self.num_classes = num_classes

        self.dataset = None
        self.seed = seed
        self.img_paths = []
        self.labels = []

        
    def generate_imgs_and_labels(self, csv_path=None, img_dir=None, csv_img_col=None, csv_label_col=None, save=False):
        _csv_path = self.csv_path if csv_path==None else csv_path
        _img_dir = self.img_dir if img_dir==None else img_dir
        _csv_img_col = self.csv_img_col if csv_img_col==None else csv_img_col
        _csv_label_col = self.csv_label_col if csv_label_col==None else csv_label_col
        dataframe = pd.read_csv(_csv_path)

        img_paths = []
        labels = []
        if save:
            self.img_paths = []
            self.labels = []

        # get all images from csv label data
        for index, row in dataframe.iterrows():
            img_fname = row[_csv_img_col]
            label = row[_csv_label_col]
            img_path = os.path.join(_img_dir, img_fname)
            img_paths.append(img_path)
            labels.append(label)
            if save:
                self.img_paths.append(img_path)
                self.labels.append(label)
        
        return (img_paths, labels)

    def generate_image_tensor(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [128, 128])
        return img
    
    def _getSeed(self, seed=None):
        if seed == None:
            if self.seed:
                return self.seed
            else:
                return random.randint(1, 100)
        else:
            return seed

    def _create_dataset(self, img_paths=None, labels=None, num_classes=None, seed=None):
        _img_paths  = self.img_paths if img_paths==None else img_paths
        _labels  = self.labels if labels==None else labels
        _num_classes  = self.num_classes if num_classes==None else num_classes
        _seed = self._getSeed(seed)
        
        image_tensor = [self.generate_image_tensor(img_path) for img_path in _img_paths]
        label_tensor = tf.one_hot(_labels, _num_classes)
        dataset = tf.data.Dataset.from_tensor_slices((image_tensor, label_tensor))

        return dataset.shuffle(buffer_size=len(_img_paths), seed=_seed)
    
    def generate_dataset(self, labels=None, img_paths=None, seed=None, force_regenerate=False):
        if len(self.img_paths) == 0 or len(self.labels) == 0:
            self.generate_imgs_and_labels(save=True)
        _labels = self.labels if labels == None else labels
        _img_paths = self.img_paths if img_paths == None else img_paths
        _seed = self._getSeed(seed)

        if self.dataset == None or force_regenerate:
            self.dataset = self._create_dataset(_img_paths, _labels, _seed)

        return self.dataset