import numpy as np
from numpy import argmax
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Importing tensorflow so we can access anything later
import tensorflow as tf
from tensorflow import keras
from tensorflow import python
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

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


(x_train, y_train_cat), (x_test, y_test_cat) = load_data()

def build_model(n_categories=10, n_nodes=256, activation_type='gelu'):
    

    # building a linear stack of layers with the sequential model
    model = Sequential()

    model.add(Dense(n_nodes, activation = activation_type,use_bias=True, kernel_initializer='RandomUniform')) 
    model.add(Dense(n_nodes, activation = activation_type,use_bias=True, kernel_initializer='RandomUniform'))
    model.add(Dense(n_categories,use_bias=True, kernel_initializer='RandomUniform', activation = 'softmax'))
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
    
    
    
    return model

# def train(model, epochs=10, batch_size=600):
#     (x_train, y_train_cat), (x_test, y_test_cat) = load_data()

# batch_size = 600
# epochs = 10

# model = build_model(n_categories=10, n_nodes=256, activation_type='gelu')

# history = model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), verbose=2,
#         batch_size = batch_size, epochs = epochs)

# test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

# print(f"Test accuracy: {test_acc:.4f}")
# print(f"Test loss: {test_loss:.4f}")
    

# 1. Load data and build/train your model
(x_train, y_train), (x_test, y_test) = load_data()
model = build_model()
model.fit(x_train, y_train, epochs=10, batch_size=600)




# 2. Define the Extractor Model
# model.input is the input layer, model.layers[0].output is the 256D space
# activation_model = Model(inputs=model.input, outputs= model.layers[0].output)

# pick the actual inner layer index you need (0 is an example)
activation_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# 3. Extract activations for a small subset (e.g., first 500 images)
# This results in a matrix of shape (500, 256)
subset_x = x_test[:500]
point_cloud = activation_model.predict(subset_x)

print(f"Point cloud shape: {point_cloud.shape}")    

# You might need to install these first: pip install ripser persim
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# 1. Compute persistent homology (up to 1-dimensional holes, H1)
# ripser automatically computes the distance matrix for you!
diagrams = ripser(point_cloud, maxdim=1)['dgms']

# 2. Plot the persistence diagrams
plot_diagrams(diagrams, show=True)