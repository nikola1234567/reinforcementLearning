from sklearn.metrics import accuracy_score, f1_score


class NeuralNetworkMetrics:

    @classmethod
    def accuracy(cls, test_targets, predictions):
        return accuracy_score(test_targets, predictions)
