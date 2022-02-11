import numpy as np


class Actions:
    def __init__(self, num_layers, hidden_size, learning_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

    def executable_actions(self):
        return np.array([self.num_layers, self.hidden_size, self.learning_rate])
