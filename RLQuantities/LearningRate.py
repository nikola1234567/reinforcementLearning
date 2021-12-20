class LearningRate:
    def __init__(self, value):
        self.value = value

    def alpha(self):
        raise NotImplementedError

class StaticLearningRate(LearningRate):
    def __init__(self, value):
        super(StaticLearningRate, self).__init__(value)

    def alpha(self):
        return self.value