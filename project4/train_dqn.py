import argparse
import numpy as np
import time
import random
import os
import signal
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.optim as optim
import time

from replay_buffer import ReplayBuffer
from q_network import QNetwork
from arm_env import ArmEnv


# ---------- Utils for setting time constraints -----#
class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# ---------- End of timing utils -----------#


class TrainDQN:

    @staticmethod
    def add_arguments(parser):
        # Common arguments
        parser.add_argument('--learning_rate', type=float, default=7e-4,
                            help='the learning rate of the optimizer')
        # LEAVE AS DEFAULT THE SEED YOU WANT TO BE GRADED WITH
        parser.add_argument('--seed', type=int, default=1,
                            help='seed of the experiment')
        parser.add_argument('--save_dir', type=str, default='models',
                            help="the root folder for saving the checkpoints")
        parser.add_argument('--gui', action='store_true', default=False,
                            help="whether to turn on GUI or not")
        # 7 minutes by default
        parser.add_argument('--time_limit', type=int, default=7 * 60,
                            help='time limits for running the training in seconds')

    def __init__(self, env, device, args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        self.env = env
        self.env.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        self.device = device
        self.q_network = QNetwork(env).to(self.device)
        self.q_network_T = QNetwork(env).to(self.device)
        self.epsilon = 0.2
        self.capacity = 10000
        self.r_buffer = ReplayBuffer(buffer_limit=self.capacity)
        self.batch_size = 50
        self.gamma = 0.9
        # self.gamma = 0.85
        self.criterion = nn.MSELoss()
        self.lr = 0.001
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        print(self.device.__repr__())
        print(self.q_network)

    def save_model(self, episode_num, episode_reward, args):
        model_folder_name = f'episode_{episode_num:06d}_reward_{round(episode_reward):03d}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(self.q_network.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')

    def train_style(self, obs):
        self.epsilon *= 0.8
        if random.random() < self.epsilon:
            index = np.random.randint(0, 16)
        else:
            index = self.q_network.select_discrete_action(obs, self.device)

        # action = self.q_network.action_discrete_to_continuous(index)

        return index

    def train_model(self):
        self.optimizer.zero_grad()
        s_lst, a_lst, r_lst, s_prime_lst, _ = self.r_buffer.sample(self.batch_size)
        s_lst = torch.from_numpy(s_lst).float()
        a_lst = torch.from_numpy(a_lst).long()
        r_lst = torch.from_numpy(r_lst).float()
        s_prime_lst = torch.from_numpy(s_prime_lst).float()
        # done_mask_lst = torch.FloatTensor(done_mask_lst)
        # print("s_list", s_lst.shape)
        # print(s_lst.reshape((1,) + s_lst.shape).shape)
        q = self.q_network.forward(s_lst, self.device).gather(1, a_lst.view(-1, 1)).squeeze()
        # print("qshape", q)
        # print("rshape", r_lst.shape)
        # print("alst",a_lst)

        # discrete_action = torch.argmax(q, dim=1).tolist()
        # print("dis", len(discrete_action))
        q_t = self.q_network_T.forward(s_prime_lst, self.device)
        q_t_max = torch.max(q_t.detach(), dim=1)[0]
        # print("qtshape", q_t_max)
        q_tar = r_lst + self.gamma * q_t_max
        loss = self.criterion(q, q_tar)

        loss.backward()
        for p in self.q_network.parameters():
            p.grad.data.clamp_(-5, 5)
        self.optimizer.step()
        return loss.item()

    def train(self, args):
        # self.q_network_T.load_state_dict(self.q_network.state_dict())
        # --------- YOUR CODE HERE --------------

        # for episode in range(795):
        for episode in range(800):
            episode_reward = 0
            s = self.env.reset()
            # print(s)
            # print(s.shape)
            # counter = 0
            # train_loss = 0
            # total_loss = 0
            while True:
                # print(s)
                index = self.train_style(s)
                action = self.q_network.action_discrete_to_continuous(index)
                # print("action:",action)
                s_prime, r, done, info = self.env.step(action)
                if done:
                    break

                # s = s.reshape((-1,1))
                # s_prime = s_prime.reshape((-1,1))
                self.r_buffer.put((s, index, r, s_prime, done))

                episode_reward += r

                if len(self.r_buffer.buffer) >= self.capacity:
                    # counter += 1
                    # print("putin", len(self.r_buffer.buffer))
                    loss = self.train_model()
                    # train_loss += loss

                s = s_prime
            # if counter:
            #     total_loss = train_loss / counter

            if episode % 25 == 0:
                self.q_network_T.load_state_dict(self.q_network.state_dict())
            # self.save_model(episode, episode_reward, args)

            print("Episode:", episode, "  reward:", episode_reward)

        pass

        # ---------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed: args.seed = int(time.time())

    env = ArmEnv(args)
    device = torch.device('cpu')
    # declare Q function network and set seed
    tdqn = TrainDQN(env, device, args)
    # run training under time limit
    try:
        with time_limit(args.time_limit):
            tdqn.train(args)
    except TimeoutException as e:
        print("You ran out of time and your training is stopped!")
