import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from arm_gui import Renderer
from arm_dynamics import ArmDynamics
from robot import Robot
import time
import argparse


class ArmEnv(gym.Env):

    # ---------- IMPLEMENT YOUR ENVIRONMENT HERE ---------------------
    @staticmethod
    def cartesian_goal(radius, angle):
        return radius * np.array([np.cos(angle), np.sin(angle)]).reshape(-1, 1)

    @staticmethod
    def random_goal():
        radius_max = 2.0
        radius_min = 1.5
        angle_max = 0.5
        angle_min = -0.5
        radius = (radius_max - radius_min) * np.random.random_sample() + radius_min
        angle = (angle_max - angle_min) * np.random.random_sample() + angle_min
        angle -= np.pi / 2
        return ArmEnv.cartesian_goal(radius, angle)

    def __init__(self, arm):
        self.arm = arm  # DO NOT modify
        self.goal = None  # Used for computing observation
        self.np_random = np.random  # Use this for random numbers, as it will be seeded appropriately
        self.observation_space = None  # You will need to set this appropriately
        self.action_space = None  # You will need to set this appropriately
        # Fill in the rest of this function as needed
        self.arm.reset()
        obs_shape = list(self.arm.state.shape)

        obs_shape[0] = obs_shape[0] + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,))
        self.action_space = spaces.Box(-1.2, 1.0, shape=(2,))
        self.num_steps = 0

    # We will be calling this function to set the goal for your arm during testing.
    def set_goal(self, goal):
        self.goal = goal
        self.arm.goal = goal

    # For repeatable stochasticity
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Fill in any additional functions you might need
    def step(self, action):
        self.num_steps += 1

        # decoded_action = self._decode_action(action)
        self.arm.set_action(action)

        for _ in range(1):
            self.arm.advance()

        new_state = self.arm.get_state()

        # compute reward
        pos_ee = self.arm.dynamics.compute_fk(new_state)
        dist = np.linalg.norm(pos_ee - self.goal)
        vel_ee = np.linalg.norm(self.arm.dynamics.compute_vel_ee(new_state))
        reward = -dist ** 2

        done = False
        if self.num_steps >= 200:
            done = True
        info = dict(pos_ee=pos_ee, vel_ee=vel_ee, success=True)
        observation = np.append(new_state, self.goal).squeeze()
        # print(observation.shape)
        return observation, reward, done, info
        pass

    def reset(self, goal=None):
        self.arm.reset()
        if goal is None:
            self.goal = ArmEnv.random_goal()
        else:
            self.goal = goal

        self.arm.goal = self.goal
        self.num_steps = 0

        return np.append(self.arm.get_state(), self.goal).squeeze()

    def render(self, mode="human"):
        pass
