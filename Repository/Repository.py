from Repository.csvHandler import CSVHandler
import os
from statistics import mean

kEpisode = "Episode"
kAvgRewards = "Average Rewards/Score"
kExploration = "Epsilon"
kLearningRate = "Alpha (LR)"
kAvgSteps = "Average Steps"
kAvgScore = "Score"

keys = [kEpisode, kExploration, kLearningRate, kAvgScore, kAvgRewards, kAvgSteps]

class Repository:

    def __init__(self, repo_name, savingThreshold):
        self.repo_name = repo_name
        self.savingThreshold = savingThreshold

    @classmethod
    def basePath(cls):
        return "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski Sitemi/repositories"

    def repoName(self):
        return '{} Repositoryy'.format(self.repo_name)

    def repoPath(self):
        return '{}/{}'.format(Repository.basePath(), self.repo_name())

    def loggerPath(self):
        return '{}/{}.csv'.format(self.repoPath(), self.fileTitle())

    def createRepoIfNeeded(self):
        if not os.path.exists(self.repoPath()):
            os.makedirs(self.repoPath())

    def fileTitle(self):
        raise NotImplementedError

    def headerNames(self):
        return keys

    @classmethod
    def propertiesDict(cls, episodeNumber, epsilonValue, alpha,
             avgScore, avgRewards, avgSteps):
        return [
            {
                kEpisode: episodeNumber,
                kExploration: epsilonValue,
                kLearningRate: alpha,
                kAvgScore: avgScore,
                kAvgRewards: avgRewards,
                kAvgSteps: avgSteps
            }
        ]

    def initializeTrainingLogger(self):
        self.createRepoIfNeeded()
        csvHandler = CSVHandler(self.loggerPath())
        csvHandler.saveRow(self.headerNames())

    def save(self, episodeNumber, epsilonValue, alpha, episodeStatistics):
        if episodeNumber % self.savingThreshold != 0:
            return
        avgScore = episodeStatistics[-1][0]
        avgRewards = round(mean(episodeStatistics[-self.repository.savingThreshold:][0]))
        avgSteps = round(mean(episode_statistics[-self.repository.savingThreshold:][1]))
        propertiesDict = Repository.propertiesDict(episodeNumber, epsilonValue, alpha,
                                                   avgScore, avgRewards, avgSteps)
        self.saveProperties(propertiesDict)

    def saveProperties(self, propertiesDict):
        csvHandler = CSVHandler(self.loggerPath())
        csvHandler.saveDictionary(self.headerNames(), propertiesDict)

