from Repository.Repository import Repository
import Repository.RepositoryNames as RepositoryNames
from datetime import datetime

class TaxiV3Repository(Repository):

    REPO_NAME = RepositoryNames.TAXI_V3_REPO_NAME

    def __init__(self):
        super(TaxiV3Repository, self).__init__(self.REPO_NAME)

    def setupRunningParameters(self, learningRate, explorationStrategy, discountFactor):
        self.learningRate = learningRate
        self.explorationStrategy = explorationStrategy
        self.discountFactor = discountFactor

    def fileTitle(self):
        return 'e = {}, a = {}, y = {} [{}]'.format(self.explorationStrategy.epsilon_value,
                                                   self.learningRate,
                                                   self.discountFactor,
                                                    datetime.now())


