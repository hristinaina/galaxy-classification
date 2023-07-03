import os

import h5py
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split

# change file path to where you have stored your dataset
FILE_PATH = '../../data/Galaxy10_DECals.h5'


def load_data():
    # To get the images and labels from file
    with h5py.File(FILE_PATH, 'r') as F:
        images = np.array(F['images'])
        print("loaded images")
        labels = np.array(F['ans'])
        print("loaded ans")

    # To convert the labels to categorical 10 classes
    labels = keras.utils.to_categorical(labels, 10)

    # To convert to desirable type
    labels = labels.astype(np.float32)
    images = images.astype(np.float32)

    return images, labels


def split_data():
    images, labels = load_data()
    train_idx, test_idx = train_test_split(np.arange(labels.shape[0]), test_size=0.1)
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]


if __name__ == '__main__':
    print('Hello world!')
    split_data()