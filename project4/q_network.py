import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        # --------- YOUR CODE HERE --------------
        self.fc1 = nn.Linear(env.observation_space.shape[0], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)

        # ---------------------------------------

    def forward(self, x, device):
        # --------- YOUR CODE HERE --------------
        # print(x.shape)
        x = torch.FloatTensor(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output
        # ---------------------------------------

    def select_discrete_action(self, obs, device):
        # Put the observation through the network to estimate q values for all possible discrete actions
        # print(obs.reshape((1,) + obs.shape).shape)
        est_q_vals = self.forward(obs.reshape((1,) + obs.shape), device)
        # Choose the discrete action with the highest estimated q value
        # print("est",est_q_vals.shape)
        discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
        # print(discrete_action)
        return discrete_action

    def action_discrete_to_continuous(self, discrete_action):
        # --------- YOUR CODE HERE --------------0.1
        # candidate = {0: [0.4, -0.2], 1: [-0.3, -0.1], 2: [-0.1, 0], 3: [0.1, 0.2], 4: [0.2, 0.1], 5: [0.3, -0.4],
        #              6: [0.2, -0.2], 7: [-0.1, 0.3], 8: [-0.2, 0.1], 9: [0.1, 0.1], 10: [0.5, -0.3], 11: [0.3, 0.3],
        #              12: [-0.2, -0.2], 13: [0.4, 0.5], 14: [-0.4, -0.1], 15: [-0.2, -0.4]}

        # candidate = {0: [0.3, 0.6], 1: [-0.3, -0.15], 2: [-0.15, 0], 3: [0.1, 0.25], 4: [0.2, 0.75], 5: [0.3, -0.4],
        #              6: [0.2, -0.25], 7: [-0.1, 0.3], 8: [-0.2, 0.1], 9: [0.6, 0], 10: [0.5, -0.3], 11: [0.3, 0.3],
        #              12: [-0.75, -0.25], 13: [0.8, 0.5], 14: [-0.4, -0.6], 15: [-0.2, -0.4]}
        # candidate = {0: [0.3, 0.6], 1: [0.3, 0.3], 2: [0.6, 0.6], 3: [0., 0.3], 4: [0.6, 0.3], 5: [-0.3, 0],
        #              6: [-0.6, 0], 7: [0.3, 0], 8: [0.6, 0.1], 9: [-0.3, 0.6], 10: [-0.3, -0.3], 11: [-0.3, -0.6],
        #              12: [-0.6, -0.3], 13: [-0.6, 0.3], 14: [-0.4, -0.6], 15: [-0.2, -0.4]}
        # 14/15
        candidate = {0: [0.4, -0.2], 1: [-0.3, -0.15], 2: [-0.15, 0], 3: [0.1, 0.25], 4: [0.2, 1.2], 5: [0.3, -0.4],
                     6: [0.2, -0.25], 7: [-0.1, 0.3], 8: [-0.2, 0.1], 9: [0.15, 0.9], 10: [0.5, -0.3], 11: [0.3, 0.3],
                     12: [-0.75, -0.25], 13: [0.8, 0.5], 14: [-0.4, -0.6], 15: [-0.2, -0.4]}

        action = candidate[discrete_action]

        return np.array(action).reshape(2, -1)

        # ---------------------------------------
