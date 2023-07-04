import gc

import h5py
import numpy as np
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from sklearn.model_selection import train_test_split

# change file path to where you have stored your dataset
FILE_PATH = '../../data/Galaxy10_DECals.h5'
BATCH_SIZE = 20
EPOCHS = 15


def load_data():
    # To get the images and labels from file
    print("loading dataset...")
    with h5py.File(FILE_PATH, 'r') as F:
        images = np.array(F['images'])
        print("loaded images")
        labels = np.array(F['ans'])
        print("loaded labels")

    return images, labels


def split_data():
    images, labels = load_data()

    print("splitting dataset to train and test datasets...")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    # call garbage collector to free memory
    del images, labels
    gc.collect()

    return x_train, y_train, x_test, y_test


def define_model():
    x_train, y_train, x_test, y_test = split_data()
    model2 = Sequential()

    model2.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(256, 256, 3)))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(Dropout(0.3))

    model2.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model2.add(Dropout(0.3))

    model2.add(Flatten())
    model2.add(Dense(units=120, activation='relu'))
    model2.add(Dense(units=84, activation='relu'))
    model2.add(Dense(units=10, activation='softmax'))

    model_optimizer = Adam(lr=0.0005)

    reduceLR = ReduceLROnPlateau(monitor='accuracy', factor=.001, patience=1, min_delta=0.01, mode="auto")

    model2.compile(optimizer=model_optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    print("training stared...")
    model2.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[reduceLR])

    # print("testing started...")
    # score = model2.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])


if __name__ == '__main__':
    define_model()
