import gym
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReinforceModel(nn.Module):

    def __init__(self, num_action, num_input):
        super(ReinforceModel, self).__init__()
        self.num_action = num_action
        self.num_input = num_input

        self.layer1 = nn.Linear(num_input, 64)
        self.layer2 = nn.Linear(64, num_action)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        x = F.relu(self.layer1(x))
        actions = F.softmax(self.layer2(x))
        action = self.get_action(actions)
        log_prob_action = torch.log(actions.squeeze(0))[action]
        return action, log_prob_action

    def get_action(self, a):
        return np.random.choice(ACTION_SPACE, p=a.squeeze(0).detach().cpu().numpy())