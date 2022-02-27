from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Generator:
    def __init__(self):
        super().__init__()

    def model_from_state(self, state):
        """:param state - object from class State
        :returns sequential keras model"""
        num_layers = state.num_layers
        hidden_size = state.hidden_size
        model = Sequential()
        model.add(Input(shape=(state.num_features,)))
        for layer in range(num_layers - 1):
            model.add(Dense(hidden_size * 16, activation='relu'))
        if num_layers > 6:
            model.add(Dropout(0.25))
        model.add(Dense(state.num_classes, activation='softmax'))
        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
        return model
