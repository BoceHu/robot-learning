from base import RobotPolicy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import dataset

torch.manual_seed(6616)
np.random.seed(6616)


class position(dataset.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        return X, Y

    def __len__(self):
        return self.X.shape[0]


class POSBCRobot(RobotPolicy):
    """ Implement solution for Part1 below """

    def train(self, data):
        for key, val in data.items():
            print(key, val.shape)
        print("Using dummy solution for POSBCRobot")

        X = data['obs']
        X = torch.from_numpy(X).float()
        Y = data['actions']
        Y = Y.reshape(-1, 1)
        Y = torch.from_numpy(Y).long()
        self.DNN = nn.Sequential(
            nn.Linear(X.shape[-1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        optimizer = Adam(self.DNN.parameters(), lr=0.003)
        batch_size = 128
        epochs = 260
        train_set = position(X, Y)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        loss_f = nn.CrossEntropyLoss()
        batch_num = 0
        for epoch in range(epochs):
            train_loss = 0.
            self.DNN.train()
            for batch_index, (train_data, train_label) in enumerate(train_loader):
                optimizer.zero_grad()
                train_label_predict = self.DNN(train_data)
                train_label = train_label.reshape(-1)
                loss = loss_f(train_label_predict, train_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_num = batch_index

            print(f"epoch: {epoch + 1}/{epochs}, loss value: ", train_loss / (batch_num + 1))

    def get_action(self, obs):
        obs_t = torch.tensor(obs).float()
        self.DNN.eval()
        action = self.DNN(obs_t)
        _, predicted = torch.max(action, 0)

        return predicted
