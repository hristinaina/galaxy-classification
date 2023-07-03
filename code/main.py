import os

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# change file path to where you have stored your dataset
FILE_PATH = '../../data/Galaxy10_DECals.h5'


def load_data():
    # To get the images and labels from file
    print("loading dataset...")
    with h5py.File(FILE_PATH, 'r') as F:
        images = np.array(F['images'])
        print("loaded images")
        labels = np.array(F['ans'])
        print("loaded labels")
    images = images[:9983]  # cutting dataset to 6 from 10 classes
    labels = labels[:9983]  # cutting dataset to 6 from 10 classes

    return images, labels


def split_data():
    images, labels = load_data()

    print("splitting dataset to train and test datasets...")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    print(x_train.shape, x_test.shape)


if __name__ == '__main__':
    split_data()
