from datetime import datetime

import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from Apstractions.FileApstractions.FileWorker import FileWorker
from Apstractions.KerasApstractions.KerasLogger import KerasLogger, PolicyWeightsNotFound
from configurations import POLICY_LOGS_DIR, POLICY_EPOCH_TRACKER


def get_log_name():
    current_datetime = datetime.now()
    cd_string = current_datetime.strftime("%d_%m_%Y_%H_%M")
    return 'policy_{}'.format(cd_string)


def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


class RLPolicyAgent:
    def __init__(self, state_size, action_size):
        KerasLogger.create_policy_dir_if_needed()
        FileWorker.create_if_not_exist(POLICY_LOGS_DIR)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self.load_model()
        self.model.summary()
        self.epoch_id = self.get_epoch()
        self.tensorBoardCallback = TensorBoard(log_dir='{}/{}'.format(POLICY_LOGS_DIR, get_log_name()),
                                               histogram_freq=True,
                                               write_graph=True,
                                               write_grads=True)
        self.tensorBoardCallback.set_model(self.model)

    def load_model(self):
        try:
            return KerasLogger.load_latest_policy()
        except PolicyWeightsNotFound:
            return self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', 'mse'])
        return model

    def memorize(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        # TODO: if the prob is 0
        state = state.reshape([1, state.shape[0]])
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        train_logs = self.model.train_on_batch(X, Y)
        self.tensorBoardCallback.on_epoch_end(epoch=self.epoch_id,
                                              logs=named_logs(model=self.model,
                                                              logs=train_logs))
        self.set_epoch()
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def memorize_network(self, dataset_path):
        KerasLogger.save_network(self.model, dataset_path)

    @staticmethod
    def get_epoch():
        return int(FileWorker.read_first_character(POLICY_EPOCH_TRACKER))

    def set_epoch(self):
        self.epoch_id += 1
        FileWorker.force_save(POLICY_EPOCH_TRACKER, self.epoch_id)

    @staticmethod
    def clean_logged_policy():
        KerasLogger.clean_policy_log_history()


if __name__ == '__main__':
    RLPolicyAgent.clean_logged_policy()
