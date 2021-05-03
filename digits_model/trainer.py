import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import glob
import os
from cv2 import cv2
import tensorflow as tf
from PIL import Image

# the data, split between train and test sets
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD

def input_data_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return process_input_data(X_train, y_train, X_test, y_test)
    

def input_data_74k_digits():
    train_imgs = []
    train_digits = []
    test_imgs = []
    test_digits = []
    SPLIT_PERCENT = 80 # 80% train, 20% test

    for i in range(10):
        pics_filenames = glob.glob('digits_model/training/char74k_digits/' + str(i) + '/*.png')
        max_train_inputs = int(len(pics_filenames) * (SPLIT_PERCENT / 100))
        counter = 1
        for filename in pics_filenames:
            imgsample = Image.open(filename)
            image = np.array(imgsample)
            image = cv2.bitwise_not(image)
            image = cv2.resize(image, (28, 28))
            if counter < max_train_inputs:
                train_imgs.append(image)
                train_digits.append(i)
            else:
                test_imgs.append(image)
                test_digits.append(i)
            counter += 1  
    
    return process_input_data(np.array(train_imgs), train_digits, np.array(test_imgs), test_digits)


def process_input_data(X_train, y_train, X_test, y_test):
    # Reshape to be samples*pixels*width*height
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

    # One hot Cpde
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # convert from integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # normalize to range [0, 1]
    X_train = (X_train / 255.0)
    X_test = (X_test / 255.0)

    return X_test, y_test, X_train, y_train


def create_datagen(X_train):
    """
    Add some minor randomisation to each image
    """
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    return datagen


def create_model():
    # Building CNN
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def run():
    model = create_model()

    X_train, y_train, X_test, y_test = input_data_mnist()
    datagen = create_datagen(X_train)
    model.fit(datagen.flow(X_train, y_train, batch_size=86), validation_data=(X_test, y_test), epochs=10, batch_size=200)

    X_train, y_train, X_test, y_test = input_data_74k_digits()
    datagen = create_datagen(X_train)
    model.fit(datagen.flow(X_train, y_train, batch_size=86), validation_data=(X_test, y_test), epochs=10, batch_size=200)

    model.save("digits_model")
    print("Saved model")


if __name__ == "__main__":
    run()