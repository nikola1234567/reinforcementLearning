import numpy as np


class State:
    def __init__(self, num_classes, num_features, num_layers, hidden_size, learning_rate):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

    def executable_state(self):
        return np.array([self.num_classes, self.num_features, self.num_layers, self.hidden_size, self.learning_rate])
