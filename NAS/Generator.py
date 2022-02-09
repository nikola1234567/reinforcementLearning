from keras.models import Sequential
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
            layers.append(nn.Softmax)
        layers.append(nn.Dropout(0.25))
        return nn.Sequential(*layers)

    def model_keras(self, state):
        """:param state - object from class State
        :returns sequential keras model"""
        num_layers = state.num_layers
        num_classes = state.num_classes
        hidden_size = state.hidden_size
        model = Sequential()
        for layer in range(num_layers):
            model.add()

        opt = Adam(lr=self.learning_rate)
        return model.compile(loss='categorical_crossentropy', optimizer=opt)
