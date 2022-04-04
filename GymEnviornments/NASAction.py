

class NASAction:

    def __init__(self, state, network, episode_number):
        self.state = state
        self.network = network
        self.episode_number = episode_number

    def number_of_layers(self):
        return self.state.num_layers

    def hidden_size(self):
        return self.state.hidden_size

    def learning_rate(self):
        return self.state.learning_rate

    def neural_network_model(self):
        return self.network

    def episode(self):
        return self.episode_number