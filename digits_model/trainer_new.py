import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.utils.np_utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
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
    return process_input_data(X_test, y_test, X_train, y_train)
    

def input_data_74k_digits():
    train_imgs = []
    train_digits = []
    test_imgs = []
    test_digits = []
    SPLIT_PERCENT = 80 # 80% train, 20% test

    for i in range(10):
        pics_filenames = glob.glob('digits_model/kaggle/char74k_digits/' + str(i) + '/*.png')
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
    
    return process_input_data(np.array(test_imgs), test_digits, np.array(train_imgs), train_digits)
    
def process_input_data(X_test, y_test, X_train, y_train):
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

# -------when you dont want to evaluate the model-------------
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

# -------------------------------------------------------------

# -------evaluate a model using k-fold cross-validation--------
def evaluate_model(X_train, y_Train, n_folds=5):

    accuracy, data = list(), list()

    # prepare 5-cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    for x_train, x_test in kfold.split(X_train):
        # create model
        model = create_model()
        # select rows for train and test
        trainX, trainY, testX, testY = X_train[x_train], y_Train[x_train], X_train[x_test], y_Train[x_test]
        # fit model
        data_fit = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        # stores accuracy
        accuracy.append(acc)
        data.append(data_fit)
    return accuracy, data


# plot diagnostic learning curves
def summarize_diagnostics(data):
    for i in range(len(data)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(data[i].history['loss'], color='red', label='green')
        pyplot.plot(data[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(data[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(data[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(acc):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(acc) * 100, np.std(acc) * 100, len(acc)))

    # box and whisker plots of results
    pyplot.boxplot(acc)
    pyplot.show()

# --------------------------------------------------------------

# This function predicts the images already in the dataset
def test(X_train, model):
    test_images = X_train[1:5]
    test_images = test_images.reshape(test_images.shape[0], 28, 28)

    for i, test_image in enumerate(test_images, start=1):
        org_image = test_image
        test_image = test_image.reshape(1, 28, 28, 1)
        prediction = model.predict_classes(test_image, verbose=0)

        print("Predicted digit: {}".format(prediction[0]))
        plt.subplot(220 + i)
        plt.axis('off')
        plt.title("Predicted digit: {}".format(prediction[0]))
        plt.imshow(org_image, cmap=plt.get_cmap('gray'))

    plt.show()


def run():
    model = create_model()

    X_test, y_test, X_train, y_train = input_data_mnist()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    test(X_train, model)

    X_test, y_test, X_train, y_train = input_data_74k_digits()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    test(X_train, model)

    # Evaluate
    # accuracy, data = evaluate_model(X_train, y_train)
    # summarize_diagnostics(data)
    # summarize_performance(accuracy)
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200)

    # save model and architecture to single file
    model.save("digits_model")
    print("Saved model to disk")


if __name__ == "__main__":
    run()