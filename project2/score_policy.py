from simple_maze import SimpleMaze
import random
from data_utils import load_data
import torch
import numpy as np
import argparse
import time


def test_model(policy, n_test, obs_type, maps, gui=False):
    success = 0
    max_dists = np.zeros(n_test)
    min_dists = np.zeros(n_test)
    env = SimpleMaze(maps=maps, gui=gui, obs_type=obs_type)
    for i in range(n_test):
        obs = env.reset()
        goal_pos = env.map.get_goal_spawn_pos()
        agent_pos = env.map.get_agent_spawn_pos()
        max_dists[i] = np.linalg.norm(goal_pos - agent_pos)
        done = False
        agent_poses = []
        for step in range(50):
            act = policy.get_action(obs)
            obs, _, done, info = env.step(act)
            agent_poses.append(info['agent'])
            if done:
                success += 1
                break
        dist = np.min(np.linalg.norm(np.asarray(agent_poses)[:, :2] - goal_pos, axis=-1))
        if done:
            dist = 0

        min_dists[i] = dist

    success_rate = success / n_test

    # print('success_rate', success_rate)

    score = np.mean((max_dists - min_dists) / max_dists)
    return success_rate, score


def score_pos_bc(gui=False, model=None):
    data = load_data('./data/map1.pkl')
    data.pop('rgb')
    data.pop('agent')
    data['obs'] = data.pop('poses')

    from solutions.pos_bc_robot import POSBCRobot
    policy = POSBCRobot()
    if model is not None:
        policy.network.load_state_dict(torch.load(model))
    else:
        policy.train(data)
    _, score = test_model(policy, n_test=100, obs_type="poses", maps=[0], gui=gui)
    return score


def score_rgb_bc1(gui=False, model=None):
    data = load_data('./data/map1.pkl')
    data.pop('poses')
    data.pop('agent')
    data['obs'] = data.pop('rgb')

    from solutions.rgb_bc_robot1 import RGBBCRobot1
    policy = RGBBCRobot1()

    if model is not None:
        policy.network.load_state_dict(torch.load(model))
    else:
        policy.train(data)
    _, score = test_model(policy, n_test=100, obs_type="rgb", maps=[0], gui=gui)

    return score


def score_rgb_bc2(gui=False, model=None):
    data = load_data('./data/all_maps.pkl')
    data['obs'] = data.pop('rgb')
    from solutions.rgb_bc_robot2 import RGBBCRobot2
    policy = RGBBCRobot2()
    if model is not None:
        policy.network.load_state_dict(torch.load(model))
    else:
        policy.train(data)
    _, score = test_model(policy, n_test=100, obs_type="rgb", maps=None, gui=gui)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--part1_model_path', type=str)
    parser.add_argument('--part2_model_path', type=str)
    parser.add_argument('--part3_model_path', type=str)
    args = parser.parse_args()
    part1_bound = 0.99
    part2_bound = 0.95
    part3_bound = 0.95
    score_pos = score_pos_bc(gui=args.gui, model=args.part1_model_path)
    score_rgb1 = score_rgb_bc1(gui=args.gui, model=args.part2_model_path)
    score_rgb2 = score_rgb_bc2(gui=args.gui, model=args.part3_model_path)
    print('\n\n\n--------SCORES--------')
    print('BC with positions:', score_pos)
    print('BC with rgb images:', score_rgb1)
    print('BC with multiple maps:', score_rgb2)
    print('----------------------')
    print('\n\n\n--------Grades--------')
    grade1 = score_pos / part1_bound * 4 if score_pos < part1_bound else 4
    grade2 = score_rgb1 / part2_bound * 5 if score_rgb1 < part2_bound else 5
    grade3 = score_rgb2 / part3_bound * 6 if score_rgb2 < part3_bound else 6
    total_grade = grade1 + grade2 + grade3
    print(f'Grade for part 1: {score_pos:.2f} / {part1_bound:.2f} * 4 = {grade1:.2f}')
    print(f'Grade for part 2: {score_rgb1:.2f} / {part2_bound:.2f} * 5 = {grade2:.2f}')
    print(f'Grade for part 3: {score_rgb2:.2f} / {part3_bound:.2f} * 6 = {grade3:.2f}')
    print(f'Total grade: {total_grade:.2f} / 15.00')
    print('----------------------')
