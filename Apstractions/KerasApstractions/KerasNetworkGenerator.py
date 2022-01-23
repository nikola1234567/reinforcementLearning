import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam


class NeuralNetworkFactory:

    @classmethod
    def default_place_holder_network(cls):
        model = Sequential()
        model.add(Input(shape=(32,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='linear'))
        return model
