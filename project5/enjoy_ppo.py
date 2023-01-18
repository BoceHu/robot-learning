from ast import arg
from cmath import inf
import os
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from arm_env import ArmEnv
from geometry import polar2cartesian
from robot import Robot
from arm_dynamics import ArmDynamics
from arm_gui import Renderer
import time

# test_utils should be obfuscated
from test_utils import test_policy, score_policy


def random_goal():
    radius_max = 2.0
    radius_min = 1.5
    angle_max = 0.5
    angle_min = -0.5
    radius = (radius_max - radius_min) * np.random.random_sample() + radius_min
    angle = (angle_max - angle_min) * np.random.random_sample() + angle_min
    angle -= np.pi/2
    goal = polar2cartesian(radius, angle)
    return goal

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="model")
    parser.add_argument('--gui', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    parser.add_argument('--random_actions', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    parser.add_argument('--random_goals', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    # arm
    parser.add_argument('--num_links', type=int, default=2)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--dt', type=float, default=0.01)

    args = parser.parse_args()
    return args

def make_arm(args):
    arm = Robot(
        ArmDynamics(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.dt

        )
    )
    arm.reset()
    return arm

def test_random_goal(policy, env, args, renderer=None):
    env.reset()
    env.set_goal( random_goal() )
    env.arm.reset() # force arm to be in vertical configuration
    min_dist = 100000
    obs, rewards, done, info = env.step(env.action_space.sample() * 0)
    while True:
        action, _states = policy.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if renderer is not None:
            renderer.plot([(env.arm, "tab:blue")])
        if done:
            break
    
def random_goals(args, policy, env, renderer):
    while (True):
        test_random_goal(policy, env, args, renderer)

def random_actions(args, env, renderer):
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if args.gui:
            renderer.plot([(env.arm, "tab:blue")])
        if done:
            env.reset()
                
def main(args):
    set_random_seed(seed=100)

    # Create arm robot
    arm = make_arm(args)

    # Create environment
    env = ArmEnv(arm)
    env.seed(100)

    renderer = None
    if args.gui:
        renderer = Renderer()
        time.sleep(1)

    if (not args.random_actions):
        # Load and test policy
        policy = PPO.load(args.model_path)
        if (not args.random_goals):
            score_policy(args, policy, env, renderer)
        else:
            random_goals(args, policy, env, renderer)
    else:
        random_actions(args, env, renderer)

if __name__ == "__main__":

    main(get_args())
