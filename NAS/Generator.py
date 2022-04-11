import tensorflow as tf
from keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Generator:
    def __init__(self):
        super().__init__()

    def model_from_state(self, state):
        """:param state - object from class State
        :returns sequential keras model"""
        if state.conv_ize != (0, 0):
            return self.model_conv_from_state(state)

        num_layers = state.num_layers
        hidden_size = state.hidden_size
        model = Sequential()
        model.add(Input(shape=(state.num_features,)))
        for layer in range(num_layers - 1):
            model.add(Dense(hidden_size * 16, activation='relu'))
        if num_layers > 6:
            model.add(Dropout(0.25))
        model.add(Dense(state.num_classes, activation='softmax'))
        opt = Adam(learning_rate=state.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
        return model

    def model_conv_from_state(self, state):
        """:param state - object from class State
        :returns sequential keras model"""

        num_layers = state.num_layers
        conv_2d_param = 32
        model = Sequential()
        model.add(
            Conv2D(conv_2d_param, (3, 3), activation='relu', input_shape=state.input_shape()))
        model.add(MaxPool2D(2, 2))
        for layer in range(num_layers - 1):
            conv_2d_param = conv_2d_param * 2
            model.add(Conv2D(conv_2d_param, (2, 2), activation='relu'))
            model.add(MaxPool2D(2, 2))
        model.add(Flatten())
        model.add(Dense(state.num_classes, activation='softmax'))
        opt = Adam(learning_rate=state.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
        return model


if __name__ == '__main__':
    input_shape = (4, 2, 2, 1)
    X = tf.random.normal(input_shape)
    print(X.shape)