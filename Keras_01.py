
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense

def simple_nn():
    x_train = [[0,0], [0,1], [1,0], [1,1]]
    y_train = [[0],[1],[1],[0]]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model = Sequential()
    model.add(Dense(64, input_dim = 20, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    model.fit(x_train,y_train, nb_epochs = 10, verbose = 2)

simple_nn()


