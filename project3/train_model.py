import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time

import models
from models import *

np.set_printoptions(suppress=True)


class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    # ---
    # Your code goes here
    pass
    # ---


def train(model, train_loader, epoch, total_num):
    model.train()
    lr = 0.008
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True, min_lr=1e-5)

    loss_sum = 0
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        X_train = batch_data[0]
        Y_labels = batch_data[1]
        Y_train = model(X_train)
        loss = criterion(Y_labels, Y_train)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    scheduler.step(loss_sum)
    print(f"epoch:{epoch} loss:{loss_sum / total_num}")


def test(model, test_loader, total_num):
    model.eval()
    criterion = nn.MSELoss()

    loss_sum = 0
    for i, batch_data in enumerate(test_loader):
        X_test = batch_data[0]
        Y_labels = batch_data[1]
        Y_test = model(X_test)
        loss = criterion(Y_labels, Y_test)

        loss_sum += loss.item()

    test_loss = loss_sum / total_num
    print(f"--------TEST LOSS--------:{test_loss}")

    return test_loss


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    args = get_args()
    num_links = args.num_links
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Loading Data")
    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1500)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1500)
    total_train = train_set.__len__()
    total_test = test_set.__len__()
    loss_sum = float('inf')
    time_step = 0.01

    model = models.build_model(num_links, time_step)

    print("Start Train")
    epochs = 1500
    best = None
    for epoch in range(epochs):
        train(model, train_loader, epoch, total_train)
        current_loss = test(model, test_loader, total_test)

        if loss_sum > current_loss:
            loss_sum = current_loss
            best = epoch
            model_folder_name = f'epoch_{epoch:04d}_loss_{loss_sum:.8f}'
            if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
                os.makedirs(os.path.join(args.save_dir, model_folder_name))
            torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
            print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')

        print(f"----------Best Epoch:{best} Current Epoch:{epoch}---------------")


if __name__ == '__main__':
    main()
