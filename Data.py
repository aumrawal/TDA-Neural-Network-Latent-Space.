import numpy as np
from numpy import argmax
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Importing tensorflow so we can access anything later
import tensorflow as tf
from tensorflow import keras

# Short aliases
layers = keras.layers

# # keras imports for importing our dataset
# from tensorflow.keras.datasets import mnist
# # keras imports for building our neural network
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.initializers import RandomUniform
# from tensorflow.keras.layers import Dense, Activation
# # keras import for manipulating some of our data
# from tensorflow.keras.utils import to_categorical

# # keras tools for loading an image from disk
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array

# # keras tools for loading an ANN model from disk
# from tensorflow.keras.models import load_model

# Load the Training and Test data
def load_data():
        
    tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.mnist.load_data()# FINISH ME #

    # print("x_train shape:", x_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("x_test shape:", x_test.shape)
    # print("y_test shape:", y_test.shape)

    # ensure x_test is a NumPy array of type float32
    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')
    

    x_train /= 255
    x_test /= 255


    ## building the input 1D vector from the 28x28 pixels
    x_train = tf.reshape(x_train, (60000, 28*28))
    x_test =  tf.reshape(x_test, (10000, 28*28))

    # One-hot encoding of the labels 
    n_classes = 10 # we know there are 10 classes (digits 0 to 9)
    # print("Shape before one-hot encoding: ", y_train.shape)

    y_train_cat = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, n_classes) # FINISH ME #
    # print("Shape after one-hot encoding: ", y_train_cat.shape)
    return (x_train, y_train_cat), (x_test, y_test_cat)

def load_x_test():
    (x_train, y_train_cat), (x_test, y_test_cat) = load_data()

    return x_test

def load_y_test():
    tf.keras.datasets.mnist.load_data(
        path='mnist.npz'
    )
    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.mnist.load_data()# FINISH ME #
    
    return y_test

if __name__ == "__main__":
    # This only runs if you execute: python data.py
    (x_train, y_train_cat), (x_test, y_test_cat) = load_data()
    print("Train data:", x_train.shape, y_train_cat.shape)
    print("Test data:", x_test.shape, y_test_cat.shape)



