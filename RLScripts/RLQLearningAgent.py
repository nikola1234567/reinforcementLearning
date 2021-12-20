import gym
import numpy as np
from statistics import mean

class RLQLearningAgent:
    def __init__(self,enviornment_name, repository):
        self.enviornment = gym.make(enviornment_name)
        self.repository = repository

    def get_random_action(self, env):
        """
        Returns a random action for the specific environment.
        :param env: OpenAI Gym environment
        :return: random action
        """
        return env.action_space.sample()

    def get_best_action(self, q_table, state):
        """
        Returns the best action for the current state given the q table.
        :param q_table: q table
        :param state: current state
        :return: best action
        """
        return np.argmax(q_table[state])

    def get_action(seelf, env, q_table, state, epsilon):
        """
        Returns the best action following epsilon greedy policy for the current state given the q table.
        :param env: OpenAI Gym environment
        :param q_table: q table
        :param state: current state
        :param epsilon: exploration rate
        :return:
        """

        num_actions = env.action_space.n
        probability = np.random.random() + epsilon / num_actions
        if probability < epsilon:
            return get_random_action(env)
        else:
            return get_best_action(q_table, state)

    def random_q_table(self, min_val, max_val, size):
        """
        Returns randomly initialized n-dimensional q table.
        :param min_val: lower bound of values
        :param max_val: upper bound of values
        :param size: size of the q table
        :return: n-dimensional q table
        """
        return np.random.uniform(low=min_val, high=max_val, size=size)

    def calculate_new_q_value(self, q_table, old_state, new_state, action, reward, lr=0.1, discount_factor=0.99):
        """
        Calculates new q value for the current state given the new state, action and reward.
        :param q_table: n-dimensional q table
        :param old_state: old (current) state
        :param new_state: new (next) state
        :param action: action to be taken at state old_state
        :param reward: reward received for performing action
        :param lr: learning rate
        :param discount_factor: discount factor
        :return: new q value for old_state and action
        """
        max_future_q = np.max(q_table[new_state])
        if isinstance(old_state, tuple):
            current_q = q_table[old_state + (action,)]
        else:
            current_q = q_table[old_state, action]

        return (1 - lr) * current_q + lr * (reward + discount_factor * max_future_q)

    def run_q_learning(self, q_table, explorationStrategy, learningRate, discountFactor,
                       number_of_episodes):
        self.repository.initializeTrainingLogger()
        episode_statistics = []
        for episode in range(number_of_episodes):
            current_state = self.enviornment.reset()
            self.enviornment.render()
            done = False
            episode_rewards = []
            # explorationStrategy.reset() # dali treba da se resetira od epizoda vo epizoda?
            while done == False:
                action = self.get_action(env, Q_table, current_state, explorationStrategy.epsilon_value)
                state_new, reward, done, info = env.step(action)
                self.enviornment.render()
                if done:
                    q_table[current_state, action] = reward
                else:
                    q_table[current_state, action] = self.calculate_new_q_value(q_table, current_state,
                                                                           state_new, action, reward,
                                                                           learningRate.alpha(),
                                                                           discountFactor.gama())
                episode_rewards.append(reward)
                explorationStrategy.decay() # dali treba vo sekoja epizoda vo sekoja iteracija da se namaluva ?
                # if epsilon_value > min_number_of_episodes: # ???????
                #     epsilon_value -= reduction
                # printanje vo repo izvestaj za iteracija vo epizoda ?

            # printanje vo repo izvestaj average rewards do nekoja epizoda itn. -> vidi lab 6
            episode_statistics.append((sum(episode_rewards), episode_rewards.count()))
            self.repository.save(episode, explorationStrategy.epsilon_value,
                                 learningRate.alpha(), episode_statistics)

        env.close()
        return average_rewards

    def q_learning(self, explorationStrategy, learningRate, discountFactor,
                       number_of_episodes):
        number_of_states = self.enviornment.observation_space.n
        number_of_actions = self.enviornment.action_space.n
        table_size = (number_of_states, number_of_actions)
        table = self.random_q_table(0, 0, table_size)
        self.run_q_learning(table, explorationStrategy, learningRate, discountFactor, number_of_episodes)