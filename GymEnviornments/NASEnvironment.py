import keras.callbacks
import numpy as np
from gym import Env
from Apstractions.KerasApstractions.KerasNetworkMetrics import NeuralNetworkMetrics
from Apstractions.DatasetApstractions.DatasetApstractions import Dataset, ResultType
import pandas as pd
from Apstractions.DataPreprocessing.PandasApstractions import DataFrameWorker
from TensorBoard.TensorBoardCustomManager import TensorBoardStandardManager
from TensorBoard.utils import get_confusion_matrix

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
        model = action.neural_network_model()
        tensorboard_manager = TensorBoardStandardManager(name=self.dataSet.name())
        train_f, train_c, test_f, test_c, train, test = self.dataSet.split_data()
        model.fit(x=train_f,
                  y=train_c,
                  batch_size=10,
                  epochs=30,
                  callbacks=[tensorboard_manager.callback(iteration=len(self.rewards) + 1,
                                                          episode=action.episode())])
        predictions = model.predict(x=test_f, batch_size=10, verbose=0)
        rounded_predictions = np.argmax(predictions, axis=-1)
        rounded_classes = np.argmax(test_c, axis=1)
        self.log_confusion_matrix(manager=tensorboard_manager,
                                  y=rounded_classes,
                                  y_predictions=rounded_predictions,
                                  episode=action.episode())
        reward = NeuralNetworkMetrics.accuracy(rounded_classes, rounded_predictions)
        self.log_hyper_parameters(manager=tensorboard_manager,
                                  number_of_layers=action.number_of_layers(),
                                  hidden_size=action.hidden_size(),
                                  learning_rate=action.learning_rate(),
                                  accuracy=reward)
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

    def log_confusion_matrix(self, manager, y, y_predictions, episode):
        class_names = self.dataSet.classes(result_type=ResultType.PLAIN)
        matrix = get_confusion_matrix(y_labels=y,
                                      predictions=y_predictions,
                                      class_names=class_names)
        manager.save_confusion_matrix(step=len(self.rewards) + 1,
                                      confusion=matrix,
                                      class_names=class_names,
                                      episode=episode)

    @staticmethod
    def log_hyper_parameters(manager, number_of_layers, hidden_size, learning_rate, accuracy):
        manager.save_hparams(number_of_layers=number_of_layers,
                             hidden_size=hidden_size,
                             learning_rate=learning_rate,
                             accuracy=accuracy)
