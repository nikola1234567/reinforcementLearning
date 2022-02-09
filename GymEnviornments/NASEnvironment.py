import numpy as np
from gym import Env
from Apstractions.KerasApstractions.KerasNetworkMetrics import NeuralNetworkMetrics
from Apstractions.DatasetApstractions.DatasetApstractions import Dataset
from Apstractions.KerasApstractions.KerasNetworkGenerator import NeuralNetworkFactory


class NASEnvironment(Env):

    def __init__(self, dataset_path, target_class_label, delimiter=","):
        self.dataSet = Dataset(dataset_path, target_class_label, delimiter=delimiter)
        self.state = NeuralNetworkFactory.default_place_holder_network()
        self.done_counter = 0

    def step(self, action):
        """
        Takes an action, which is a neural network, and fits and finds accuracy on the previously specified
        dataset.
        :param action: model that will be trained (type - Keras Model)
                       This model needs to be compiled previously.
        :return: Four values:
                    1. state - the model/action which was trained
                    2. reward - accuracy on the specified dataset
                    3. done - if the rewards starts to constantly decrease (currently done after 3 iterations)
                    4. info - empty (for now)
        """
        train_f, train_c, test_f, test_c, train, test = self.dataSet.split_data()
        action.fit(x=train_f, y=train_c, batch_size=10, epochs=30)
        self.done_counter += 1
        predictions = action.predict(x=test_f, batch_size=10, verbose=0)
        rounded_predictions = np.argmax(predictions, axis=-1)
        rounded_classes = np.argmax(test_c, axis=-1)
        reward = NeuralNetworkMetrics.accuracy(rounded_classes, rounded_predictions)
        done = self.done_counter == 2
        state = action
        info = {}
        return state, reward, done, info


    def render(self, mode="human"):
        # TODO: To be decided what to visualise
        # TODO: implement visualisations of diagrams with metrics, model summary etc.
        self.state.summary()

    def reset(self):
        self.state = NeuralNetworkFactory.default_place_holder_network()

