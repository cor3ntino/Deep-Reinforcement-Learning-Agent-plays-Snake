import h5py

import keras
from keras.optimizers import Adam
from keras.models import Sequential, save_model, load_model
from keras.layers.core import Dense

class Deep_Neural_Network:
    def __init__(self, learning_rate=0.0005):

        self.learning_rate = learning_rate
        
        self.model = Sequential()
        self.model.add(Dense(units=150, activation='relu', input_dim=21))
        self.model.add(Dense(units=150, activation='relu'))
        self.model.add(Dense(units=150, activation='relu'))
        self.model.add(Dense(units=3, activation='softmax'))
        self.opt = Adam(self.learning_rate)
        self.model.compile(loss='mse', optimizer=self.opt)

        print(self.model.summary())

    def return_value_NN(self, X):
        return(self.model.predict(X, verbose=0))
    
    def save_mod(self):

        save_model(self.model, './saved_models/DNN.h5')

    def load_mod(self):

        self.model = load_model('./saved_models/DNN.h5')
