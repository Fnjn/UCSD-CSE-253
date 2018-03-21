#!/usr/bin/python

### IMPORTS

import os
import numpy as np
import glob
import fnmatch
from random import randint
import PIL
from PIL import Image
import time
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt


import logging
# import colored_traceback
# colored_traceback.add_hook(always=True)
FORMAT = "[%(lineno)4s : %(funcName)-30s ] %(message)s"
#logging.basicConfig(level=logging.INFO, format=FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

dataset_path='dataset_new'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')
#dataset_src_path='fashion_data'
fashion_dataset_path='fashion_data/'
dataset_train_labels_path=os.path.join(dataset_train_path, 'labels.txt')
dataset_test_labels_path=os.path.join(dataset_test_path, 'labels.txt')
dataset_val_labels_path=os.path.join(dataset_val_path, 'labels.txt')

def list_devices():
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()

def show_image(x):
    plt.figure()
    plt.imshow(x) 
    plt.show()

def save_image(x, filename):
    plt.figure()
    plt.imshow(x) 
    plt.savefig(filename)
    
def save_img(I):
    im = Image.fromarray(np.uint8(I))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.div(image, 256.)
    return image, label
    
def prepare_new_datasets(batch_size):
    
    datasets = []
    for i, labels_path, img_path in zip(range(3), [dataset_train_labels_path, 
                                                   dataset_val_labels_path, 
                                                   dataset_test_labels_path],
                                                  [dataset_train_path, dataset_val_path, dataset_test_path]):
        raw_filenames = []
        labels = []
        t = time.time()
        with open(labels_path) as file_label:
#             cnt = 0#dev
            for line in file_label:
                line = line.split()
                raw_filenames.append(img_path + '/' + line[0])
                labels.append([int(x) for x in line[1:]])
#                 cnt += 1 #dev
#                 if cnt >10000:
#                     break
            
            # step 1
            filenames = tf.constant(raw_filenames)
            labels = tf.constant(labels)

            # step 2: create a dataset returning slices of `filenames`
            dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
            
            dataset.shuffle(buffer_size=10000)
            dataset = dataset.map(_parse_function)
            dataset = dataset.batch(batch_size)
            
            dataset = dataset.prefetch(batch_size)
            datasets.append(dataset)
        print('Duration is {}'.format(time.time()-t))
    return datasets
    
def prepare_datasets():
    datasets = []
    for i, labels_path, img_path in zip(range(3), [dataset_train_labels_path, 
                                                   dataset_val_labels_path, 
                                                   dataset_test_labels_path],
                                                  [dataset_train_path, dataset_val_path, dataset_test_path]):
        imgs = []
        labels = []
        t = time.time()
        with open(labels_path) as file_label:
#             cnt = 0#dev
            for line in file_label:
                line = line.split()
                image_file_path = img_path + '/' + line[0]
                img = np.asarray(Image.open(image_file_path))
                label = np.array([int(x) for x in line[1:]])
                imgs.append(img)
                labels.append(label)
#                 cnt += 1 #dev
#                 if cnt >10000:
#                     break
            imgs = np.array(imgs)/255.
            labels = np.array(labels)
            datasets.append(DataSet(imgs, labels))
        print('Duration is {}'.format(time.time()-t))
    return datasets
                            
class DataSet:
    def __init__(self, img, labels):
        """Construct a DataSet.
        """
        assert img.shape[0] == labels.shape[0], (
            'img.shape: %s labels.shape: %s' % (img.shape, labels.shape))
        self._num_examples = img.shape[0]
        self._img = img
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def img(self):
        return self._img

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._img = self.img[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            img_rest_part = self._img[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._img = self.img[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            img_new_part = self._img[start:end]
            labels_new_part = self._labels[start:end]
            return (np.concatenate((img_rest_part, img_new_part), axis=0),
                    np.concatenate((labels_rest_part, labels_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._img[start:end], self._labels[start:end]
            


if __name__ == '__main__':
    
    train_dataset, val_dataset, test_dataset = prepare_datasets()
#     x, y = train_dataset.next_batch(16)
    print(train_dataset.num_examples)
    print(val_dataset.num_examples)
    print(test_dataset.num_examples)