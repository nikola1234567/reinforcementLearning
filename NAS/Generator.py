from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from torch import nn


class Generator:
    def __init__(self):
        super().__init__()

    def model(self, state):
        """:param state - object from class State
         :returns sequential model"""

        layers = []
        num_layers = state.num_layers
        num_classes = state.num_classes
        hidden_size = state.hidden_size
        for layer in range(num_layers):
            layers.append(nn.Linear(hidden_size, num_classes))
            layers.append(nn.Softmax())
        layers.append(nn.Dropout(0.25))
        model = nn.Sequential(*layers)
        opt = Adam(lr=0.00001)
        return model.compile(loss='categorical_crossentropy', optimizer=opt)

    def model_keras(self, state, number_of_features, number_of_classes):
        """:param state - object from class State
        :returns sequential keras model"""
        num_layers = state.num_layers
        hidden_size = state.hidden_size
        model = Sequential()
        model.add(Input(shape=(number_of_features, )))
        for layer in range(num_layers-1):
            model.add(Dense(hidden_size*16, activation='relu'))
        model.add(Dense(number_of_classes, activation='softmax'))
        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
