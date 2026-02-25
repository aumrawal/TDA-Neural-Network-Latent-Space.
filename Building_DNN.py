#Building a DNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_model(n_categories=10, n_nodes=256, activation_type='gelu'):
    

    # building a linear stack of layers with the sequential model
    model = Sequential()

    model.add(Dense(n_nodes, activation = activation_type,use_bias=True, kernel_initializer='RandomUniform')) 
    model.add(Dense(n_nodes, activation = activation_type,use_bias=True, kernel_initializer='RandomUniform'))
    model.add(Dense(n_categories,use_bias=True, kernel_initializer='RandomUniform', activation = 'softmax'))
    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='SGD')
    
    
    
    return model

# Now we're going to take a copy of the weights in our first hidden layer of the fit


if __name__ == "__main__":
    # Only runs with python building_dnn.py
    model = build_model()
    model.summary()