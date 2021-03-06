import time

from Apstractions.DatasetApstractions.DatasetApstractions import Dataset, ImageDataSet
from Apstractions.DatasetApstractions.DatasetSamples.DatasetsPaths import *
from GymEnviornments.NASEnvironment import NASEnvironment, NUMBER_OF_ACTIONS_EXECUTED_KEY
from NAS.Actions import Actions
from NAS.Generator import Generator
from NAS.State import State
from RLScripts.RLPolicyAgent import RLPolicyAgent
from GymEnviornments.NASAction import NASAction
from TensorBoard.TensorBoardCustomManager import TensorBoardCustomManager


EPISODE_ITERATIONS = "episode_iteration"


def get_class_attributes(class_object):
    """"help function for attributes present in a class
    :returns  attributes of an object from a class"""
    return list(class_object.__dict__.keys())


def from_state_to_action(state):
    """help function for creating an action space from state
    :returns actions object"""
    return Actions(num_layers=state.num_layers,
                   hidden_size=state.hidden_size,
                   learning_rate=state.learning_rate)


class Controller:
    def __init__(self, dataset_path, dataset_delimiter=",", dataset_image=False):
        super().__init__()
        self.dataset_path = dataset_path
        if dataset_image:
            self.dataSet = ImageDataSet(self.dataset_path, delimiter=dataset_delimiter)
        else:
            self.dataSet = Dataset(self.dataset_path, delimiter=dataset_delimiter)
        self.initial_state = State(self.dataSet.number_of_classes(),
                                   self.dataSet.number_of_features(), 1, 1, 0.0001,
                                   self.dataSet.complex_type_features())
        self.current_state = self.initial_state
        self.actions = from_state_to_action(self.initial_state)
        self.action_space = len(get_class_attributes(self.actions))
        self.generator = Generator()
        self.nas_environment = NASEnvironment(self.dataSet)
        self.policy = RLPolicyAgent(len(get_class_attributes(self.actions)), self.action_space)
        self.num_episodes = 2
        self.action_decoding_dict = self.create_action_dict()
        self.tensor_board_manager = TensorBoardCustomManager(name='ReinforceScalars')

    def create_action_dict(self):
        """:returns dict {action number:attribute name}"""
        dictionary = dict()
        attributes = get_class_attributes(self.actions)

        if len(attributes) != self.action_space:
            raise RuntimeError("Action space and actions attributes not compatible")

        for index in range(self.action_space):
            dictionary[index] = attributes[index]
        return dictionary

    def controller_reset(self):
        """resets the environment and the initial state"""
        self.nas_environment.reset()
        self.current_state = self.initial_state
        self.actions = from_state_to_action(self.current_state)

    def get_model(self):
        """:returns a neural network generated by the generator with values specified in the class"""
        return self.generator.model_from_state(self.current_state)

    def implement_action(self, action):
        """"sets the new current state """
        attribute = self.action_decoding_dict.get(action)
        previous_value = self.current_state.__getattribute__(attribute)
        if attribute == "learning_rate":
            self.current_state.__setattr__(attribute, previous_value + 0.0001)
        else:
            self.current_state.__setattr__(attribute, previous_value + 1)
        self.actions = from_state_to_action(self.current_state)

    def run_episode(self, episode_number):
        """"runs one episode until the done flag from the environment is true
         -gets the next action and probability from the policy
         -implements the action in the current state
         -generator creates a model from the current state parameters
         -environment trains the model with current state and returns reward and done flag
         -policy updates """
        done = False
        info = {}
        while not done:
            action, prob = self.policy.act(self.actions.executable_actions())
            self.implement_action(action)
            action_model = self.get_model()
            nas_action = NASAction(state=self.current_state,
                                   network=action_model,
                                   episode_number=episode_number)
            state, reward, done, info = self.nas_environment.step(nas_action)
            self.policy.memorize(self.actions.executable_actions(), action, prob, reward)
        self.tensor_board_manager.save(scalar_name=EPISODE_ITERATIONS,
                                       scalar=info[NUMBER_OF_ACTIONS_EXECUTED_KEY],
                                       step=int(time.time()))

    def controller_preform(self):
        """preforms number of episodes and returns the best state
        :returns state"""
        for episode in range(self.num_episodes):
            self.run_episode(episode_number=episode)
            self.controller_reset()
            self.policy.train()

        self.policy.memorize_network(self.dataset_path)
        return self.current_state


if __name__ == '__main__':
    controller = Controller(dataset_path=FER_2013_PATH, dataset_image=True)
    model = controller.controller_preform()
    print(model)
