class ExplorationStrategy:
    def __init__(self, epsilon_value):
        self.permanent_copy = epsilon_value
        self.epsilon_value = epsilon_value

    def decay(self):
        raise NotImplementedError

    def reset(self):
        self.epsilon_value = self.permanent_copy

    def strategyDescription(self):
        return "N/A"

class EDecayType1(ExplorationStrategy):

    def __init__(self, epsilon_value, min_number_of_episodes, number_of_episodes):
        super(EDecayType1, self).__init__(epsilon_value)
        self.min_number_of_episodes = min_number_of_episodes
        self.number_of_episodes = number_of_episodes
        self.reduction = (self.epsilon_value - min_number_of_episodes) / number_of_episodes

    def decay(self):
        self.epsilon_value -= self.reduction

    def strategyDescription(self):
        return "Decay Type 1"

class EDecayType2(ExplorationStrategy):

    def __init__(self, epsilon_value, min_epsilon, decay_value):
        super(EDecayType2, self).__init__(epsilon_value)
        self.min_epsilon = min_epsilon
        self.decay_value = decay_value

    def decay_factor(self):
        return self.epsilon_value * self.decay_value

    def decay(self):
        self.epsilon_value = max(self.min_epsilon, self.decay_factor())

    def strategyDescription(self):
        return "Decay Type 2"


