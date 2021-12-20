from TaxiV3 import TaxiV3Repository
from RLQuantities import DiscountFactor
from RLScripts import RLQLearningAgent
from GymEnviornments import GymEnviornments

if __name__ == '__main__':

    taxiRepository = TaxiV3Repository()
    lr = StaticLearningRate(0.1)
    e = EDecayType2(0.99, 0.1, 0.9999)
    y = DiscountFactor(0.9)
    agent = RLQLearningAgent(GymEnviornments.TAXI_V3, taxiRepository)
    agent.q_learning(e, lr, y, 5)

