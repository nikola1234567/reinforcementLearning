

class NASAction:

    def __init__(self, state, network):
        self.state = state
        self.network = network

    def number_of_layers(self):
        return self.state.num_layers

    def hidden_size(self):
        return self.state.hidden_size

    def learning_rate(self):
        return self.state.learning_rate

    def neural_network_model(self):
        return self.network