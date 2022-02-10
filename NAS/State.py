import numpy as np


class State:
    def __init__(self, num_layers, num_classes, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

    def executable_state(self):
        return np.array([self.num_classes, self.num_layers, self.hidden_size])