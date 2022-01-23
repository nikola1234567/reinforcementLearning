from gym import Env
from NAS.Generator import Generator
from Apstractions.KerasApstractions.KerasNetworkGenerator import NeuralNetworkFactory


class NASEnviornment(Env):

    def __init__(self, dataset_path):
        # the action generator
        # env.action_space.sample()
        self.action_space = Generator()
        self.datasetPath = dataset_path
        self.state = NeuralNetworkFactory.default_place_holder_network()

    def step(self, action):
        # TODO: should return: state, reward, done, info
        # TODO: when agent is "done"? -> to be decided
        # TODO: what should be the reward, the network accuracy, combination of network metrics?
        pass

    def render(self, mode="human"):
        # TODO: To be decided what to visualise
        # TODO: implement visualisations of diagrams with metrics, model summary etc.
        self.state.summary()

    def reset(self):
        self.state = NeuralNetworkFactory.default_place_holder_network()

if __name__ == '__main__':
    # envInstance = NASEnviornment(pateka)

