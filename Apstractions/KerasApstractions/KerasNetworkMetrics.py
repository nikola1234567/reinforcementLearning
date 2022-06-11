from sklearn.metrics import accuracy_score, mean_squared_error


class NeuralNetworkMetrics:

    @classmethod
    def accuracy(cls, test_targets, predictions):
        return accuracy_score(test_targets, predictions)

    @classmethod
    def loss(cls, test_targets, predictions):
        """
        Loss function using mean squared error
        :param test_targets: array of true values
        :param predictions:  array of predicted values
        :return: mean squared error of loss
        """
        return mean_squared_error(test_targets, predictions)
