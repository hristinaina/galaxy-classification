import gc

import h5py
import numpy as np
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.regularizers import l2
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# change file path to where you have stored your dataset
FILE_PATH = '../../data/Galaxy10_DECals.h5'
BATCH_SIZE = 25
EPOCHS = 10


def load_data():
    # get the images and labels from file
    print("loading dataset...")
    with h5py.File(FILE_PATH, 'r') as F:
        images = np.array(F['images'])
        print("loaded images")
        labels = np.array(F['ans'])
        print("loaded labels")

    # shuffle data in dataset
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    return images, labels


def split_data():
    images, labels = load_data()

    print("splitting dataset to train and test datasets...")
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

    print("splitting dataset to train and validation datasets...")
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    # call garbage collector to free memory
    del images, labels
    gc.collect()

    return x_train, y_train, x_val, y_val, x_test, y_test


def define_model():
    x_train, y_train, x_val, y_val, x_test, y_test = split_data()

    datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, horizontal_flip=True,
                                 vertical_flip=True)

    train_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
    val_generator = datagen.flow(x_val, y_val, batch_size=BATCH_SIZE)

    model = Sequential()

    model.add(Conv2D(filters=18, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.002), activation='relu',
                     input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.004), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=l2(0.003), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=448, activation='relu', kernel_regularizer=l2(0.003)))
    model.add(Dense(units=224, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dense(units=10, activation='softmax'))

    model_optimizer = Adam(learning_rate=0.0003)
    reduceLR = ReduceLROnPlateau(monitor='accuracy', factor=.1, patience=3, min_delta=0.01, mode="auto")
    model.compile(optimizer=model_optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

    print("\nTRAINING STARTED...")
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[reduceLR])
    gc.collect()

    print("\nTESTING STARTED...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    model.save('../model/model.h5')

    return history


def show():
    # Access the learning rate values and loss values from the callback
    val_loss = history.history['val_loss']

    # Plot the validation loss
    plt.plot(val_loss)
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    history = define_model()
    show()
