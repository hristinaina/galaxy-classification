import gc
import os

import h5py
import numpy as np
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.regularizers import l2
from keras_preprocessing.image import ImageDataGenerator

BATCH_SIZE = 20
EPOCHS = 15

train_data_dir = '../data/train/train_data.h5'
test_data_dir = '../data/test/test_data.h5'
validation_data_dir = '../data/validation/validation_data.h5'


def load_file(file_path):
    with h5py.File(file_path, 'r') as F:
        print("tuu sam")
        images = np.array(F['images'])
        labels = np.array(F['ans'])

    return images, labels


def get_datasets():
    train_data_generator = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_data_generator = ImageDataGenerator(rescale=1. / 255)
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    # Create the train generator
    images, labels = load_file(train_data_dir)
    train_data_generator.fit(images)
    train_generator = train_data_generator.flow(images, labels, batch_size=BATCH_SIZE)

    # Create the validation generator
    images, labels = load_file(validation_data_dir)
    val_data_generator.fit(images)
    val_generator = train_data_generator.flow(images, labels, batch_size=BATCH_SIZE)

    # Create the test generator
    images, labels = load_file(test_data_dir)
    test_data_generator.fit(images)
    test_generator = test_data_generator.flow(images, labels, batch_size=BATCH_SIZE)

    gc.collect()
    return train_generator, val_generator, test_generator


def define_model():
    train_generator, val_generator, test_generator = get_datasets()
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.002), activation='relu',
                     input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.003), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(units=120, activation='relu', kernel_regularizer=l2(0.003)))
    model.add(Dense(units=84, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dense(units=10, activation='softmax'))

    model_optimizer = Adam(lr=0.0005)

    reduceLR = ReduceLROnPlateau(monitor='accuracy', factor=.001, patience=1, min_delta=0.01, mode="auto")

    model.compile(optimizer=model_optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    print("\nTRAINING STARTED...")
    model.fit(train_generator, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=val_generator, callbacks=[reduceLR])

    gc.collect()
    print("\nTESTING STARTED...")
    test_loss, test_accuracy = model.evaluate(test_generator)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)


if __name__ == '__main__':
    define_model()
