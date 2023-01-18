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


class simple_cnn(nn.Module):
    def __init__(self):
        super(simple_cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        X = self.conv1(x)
        X = F.relu(X)
        X = self.pool(X)
        X = X.view(-1, 16 * 16 * 16)
        X = F.relu(self.fc1(X))
        output = self.fc2(X)

        return output


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


class RGBBCRobot1(RobotPolicy):
    """ Implement solution for Part2 below """

    def train(self, data):
        for key, val in data.items():
            print(key, val.shape)
        print("Using dummy solution for RGBBCRobot1")

        X = data['obs']
        X = X.transpose((0, 3, 1, 2))
        X = torch.from_numpy(X).float()
        Y = data['actions']
        Y = Y.reshape(-1, 1)
        Y = torch.from_numpy(Y).long()
        self.CNN = simple_cnn()
        optimizer = Adam(self.CNN.parameters(), lr=0.003)
        batch_size = 32
        epochs = 60
        train_set = position(X, Y)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        loss_f = nn.CrossEntropyLoss()
        batch_num = 0
        for epoch in range(epochs):
            train_loss = 0
            self.CNN.train()
            for batch_index, (train_data, train_label) in enumerate(train_loader):
                optimizer.zero_grad()
                train_label_predict = self.CNN(train_data)
                train_label = train_label.reshape(-1)
                loss = loss_f(train_label_predict, train_label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_num = batch_index
            print(f"epoch: {epoch + 1}/{epochs}, loss value: ", train_loss / (batch_num + 1))

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = obs.transpose((0, 3, 1, 2))
        obs_t = torch.tensor(obs).float()
        self.CNN.eval()
        action = self.CNN(obs_t)
        action = torch.squeeze(action)
        _, predicted = torch.max(action, 0)

        return predicted
