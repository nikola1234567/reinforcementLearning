from torch import nn

class Generator:
    def __init__(self,num_classes, hidden_size,
                 num_layers, learning_rate, activation_function):
        super().__init__()

        layers=[]
        for layer in range(num_layers):
            layers.append(nn.Linear(hidden_size,num_classes))
            layers.append(nn.Softmax)
        layers.append(nn.Dropout(0.25))
        self.model=nn.Sequential(*layers)

    def training(self,data_set):
        self.model.train()


    # def model_keras(self):
    #     model = Sequential()
    #     model.add(Reshape((self.input), input_shape=(self.state_size,)))
    #
    #     model.add(Flatten())
    #     model.add(Dense(64, activation='relu', init='he_uniform'))
    #     model.add(Dense(32, activation='relu', init='he_uniform'))
    #     model.add(Dense(self.action_size, activation='softmax'))
    #     opt = Adam(lr=self.learning_rate)
    #     model.compile(loss='categorical_crossentropy', optimizer=opt)






