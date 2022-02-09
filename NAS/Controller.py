
from Apstractions.DatasetApstractions.DatasetApstractions import Dataset
from GymEnviornments.NASEnvironment import NASEnvironment
from NAS.Generator import Generator
from NAS.State import State
from RLScripts.RLPolicyAgent import RLPolicyAgent


def get_class_attributes(state):
    """"help function for attributes present in a class
    :returns  attributes of an object from State class"""
    attr_list = State.__dir__(state)
    attributes = [attribute for attribute in attr_list if
                  not (attribute.startswith('__') and attribute.endswith('__'))]
    return attributes


class Controller:
    def __init__(self, dataset_path, target_class_label):
        super().__init__()
        # TODO:set target class label
        self.dataset_path = dataset_path
        self.dataSet = Dataset(self.dataset_path, target_class_label)
        self.initial_state = State(self.dataSet.number_of_classes(), 1, 1)
        self.current_state = self.initial_state
        self.action_space = len(get_class_attributes(self.current_state))
        self.generator = Generator()
        self.nas_environment = NASEnvironment(dataset_path, target_class_label)
        self.policy = RLPolicyAgent(self.action_space, self.action_space) # pagja
        self.done = False
        self.num_episodes = 15
        self.action_decoding_dict = self.create_action_dict()

    def create_action_dict(self):
        """:returns dict {action number:attribute name}"""
        dictionary = dict()
        attributes = get_class_attributes(self.current_state)

        if len(attributes) != self.action_space:
            raise RuntimeError("Action space and state attributes not compatible")

        for index in range(self.action_space):
            dictionary[index] = attributes[index]
        return dictionary

    def controller_reset(self):
        """resets the environment and the initial state"""
        self.nas_environment = NASEnvironment.reset(self.nas_environment)
        self.current_state = self.initial_state

    def get_model(self):
        """:returns a neural network generated by the generator with values specified in the class"""
        return self.generator.model(self.current_state)

    def get_reward(self):
        """ gives a model to the environment and returns its reward and done flag
        :returns reward,done"""
        return self.nas_environment.step(self.current_state)

    def implement_action(self, action):
        """"sets the new current state """
        attribute = self.action_decoding_dict.get(action)
        previous_value = self.current_state.__getattribute__(attribute)
        self.current_state.__setattr__(attribute, previous_value + 1)

    def run_episode(self):
        """"runs one episode until the done flag from the environment is true
         -gets the next action and probability from the policy
         -implements the action in the current state
         -generator creates a model from the current state parameters
         -environment trains the model with current state and returns reward and done flag
         -policy updates """
        while not self.done:
            action, prob = self.policy.act(self.current_state)
            self.implement_action(action)
            model = self.get_model()
            reward, done = self.nas_environment.step(model)
            self.policy.memorize(self.current_state, action, prob, reward)
            self.done = done

    def controller_preform(self):
        """preforms number of episodes and returns the best state
        :returns state"""
        for episode in range(self.num_episodes):
            self.run_episode()
            self.controller_reset()
            self.policy.train()

        self.run_episode()
        self.policy.train()
        return self.current_state


if __name__ == '__main__':
    datasetPath = "C:/Users/DELL/Desktop/documents/nikola-NEW/Inteligentni Informaciski " \
                  "Sitemi/datasets/car.csv "
    controller = Controller(datasetPath, "acceptability")
    controller.controller_preform()
