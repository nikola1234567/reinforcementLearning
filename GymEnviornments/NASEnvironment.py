import numpy as np
from gym import Env
from Apstractions.KerasApstractions.KerasNetworkMetrics import NeuralNetworkMetrics
from Apstractions.DatasetApstractions.DatasetApstractions import Dataset
import pandas as pd
from Apstractions.DataPreprocessing.PandasApstractions import DataFrameWorker

REWARD_SERIES_KEY = "rewards during playing"
NUMBER_OF_ACTIONS_EXECUTED_KEY = "taken actions"


class NASEnvironment(Env):

    def __init__(self, dataset_path, delimiter=","):
        self.dataSet = Dataset(dataset_path, delimiter=delimiter)
        self.rewards = []

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
        predictions = action.predict(x=test_f, batch_size=10, verbose=0)
        rounded_predictions = np.argmax(predictions, axis=-1)
        rounded_classes = np.argmax(test_c, axis=1)
        reward = NeuralNetworkMetrics.accuracy(rounded_classes, rounded_predictions)
        self.rewards.append(reward)
        state = action
        done = self.is_done()
        info = {REWARD_SERIES_KEY: self.rewards,
                NUMBER_OF_ACTIONS_EXECUTED_KEY: len(self.rewards)}
        return state, reward, done, info

    def render(self, mode="human"):
        # TODO: To be decided what to visualise
        # TODO: implement visualisations of diagrams with metrics, model summary etc.
        pass

    def reset(self):
        self.rewards = []

    def is_done(self):
        df = pd.DataFrame(self.rewards)
        return DataFrameWorker.decreasing_or_constant(df)