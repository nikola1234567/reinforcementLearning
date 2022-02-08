class State:
    def __init__(self, num_layers, num_classes, hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size
