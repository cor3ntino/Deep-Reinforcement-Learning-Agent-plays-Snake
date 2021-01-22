import h5py

import keras
from keras.optimizers import Adam
from keras.models import Sequential, save_model, load_model
from keras.layers.core import Dense
from keras.layers import Conv2D, Flatten


class Convolutional_Neural_Network:
    def __init__(self, learning_rate=0.0005):

        self.learning_rate = learning_rate
        
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(4,8,8), data_format='channels_first'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', data_format='channels_first'))
        self.model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', data_format='channels_first'))
        self.model.add(Flatten())
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=3, activation='softmax'))
        self.opt = Adam(self.learning_rate)
        self.model.compile(loss='mse', optimizer=self.opt)
    
        print(self.model.summary())

    def return_value_NN(self, X):
        return(self.model.predict(X, verbose=0))
    
    def save_mod(self, pourcentage):

        save_model(self.model, './saved_models/CNN_' + pourcentage + '.h5')

    def load_mod(self):

        self.model = load_model('./saved_models/CNN_5000_3000_iterations_20_pourcents.h5')
