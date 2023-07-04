import h5py
import numpy as np
from sklearn.model_selection import train_test_split


"""
    RUN THIS FILE TO SPLIT YOUR DATASET
    Since initial file is too big to be loaded into memory as a whole,
    we will separate it to train and test datasets and save them as separate files.
    This way, we can use ImageDataGenerator class that loads and preprocesses data on-the-fly in batches during training
    instead of loading the entire dataset into the memory
"""

# change file path to where you have stored your dataset
FILE_PATH = '../../data/Galaxy10_DECals.h5'


def load_file():
    # get the images and labels from file
    print("\nLOADING DATASET...")
    with h5py.File(FILE_PATH, 'r') as F:
        images = np.array(F['images'])
        print("loaded images")
        labels = np.array(F['ans'])
        print("loaded labels")

    # shuffle data in dataset (data is initially sorted by labels)
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    return images, labels


def split_data():
    images, labels = load_file()

    print("\nsplitting dataset to train and test datasets...")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    print("\nsplitting train dataset to train and validation datasets...")
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2)

    return x_train, y_train, x_test, y_test, x_validation, y_validation


def save_data(images, labels, file_path):
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('images', data=images)
        hf.create_dataset('ans', data=labels)


def create_datasets():
    x_train, y_train, x_test, y_test, x_validation, y_validation = split_data()

    # Save the training dataset to a separate .h5 file
    save_data(x_train, y_train, '../data/train/train_data.h5')

    # Save the test dataset to a separate .h5 file
    save_data(x_test, y_test, '../data/test/test_data.h5')

    # Save the test dataset to a separate .h5 file
    save_data(x_validation, y_validation, '../data/validation/validation_data.h5')


if __name__ == '__main__':
    create_datasets()

