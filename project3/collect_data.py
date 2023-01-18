import math
import time

import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os

# np.set_printoptions(suppress=False)
np.set_printoptions(threshold=np.inf)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    # X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    # Y = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))
    # ---
    print(arm_teacher.dynamics.get_state_dim())
    print(arm_teacher.dynamics.get_action_dim())
    print(math.pi)

    total_num = 50000
    num_links = dynamics_teacher.num_links
    X = []
    Y = np.zeros((num_links * 2, total_num))

    angle = np.pi * 2 * np.random.rand(num_links, total_num) - np.pi * 3 / 2
    ang_v = np.array([np.random.uniform(-1, 1) for num in range(num_links) for _ in range(total_num)]).reshape(
        num_links, -1)
    actions = 6 * (1 - 2 * np.random.rand(num_links, total_num))
    X.append(angle)
    X.append(ang_v)
    X.append(actions)
    X = np.concatenate(X)

    for i in range(total_num):
        arm_teacher.set_state(X[:num_links * 2, i].reshape(-1, 1))
        arm_teacher.set_action(X[2 * num_links:, i].reshape(-1, 1))

        arm_teacher.advance()
        new_state = arm_teacher.get_state()
        for j in range(2 * num_links):
            Y[j][i] = new_state[j]
    #     print(new_state.shape)
    #
    # print(ang_v.shape)
    # print(angle.shape)
    # print(actions.shape)
    # print(X.shape)

    print('X shape:', X.shape, 'Y shape:', Y.shape)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
