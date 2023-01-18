import argparse
import os
import time
from stable_baselines3.common.utils import set_random_seed
from vec_env_utils import make_vec_env
from robot import Robot
from arm_dynamics import ArmDynamics
from stable_baselines3 import PPO


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--nenv', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--num_links', type=int, default=2)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--dt', type=float, default=0.01)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr) if args.suffix is None \
        else os.path.join(args.save_dir, args.timestr + '_' + args.suffix)

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

def train(args):

    set_random_seed(args.seed)

    # create arm
    arm = make_arm(args)

    # create parallel envs
    vec_env = make_vec_env(arm=arm, nenv=args.nenv, seed=args.seed)

    # ------ IMPLEMENT YOUR TRAINING CODE HERE ------------
    model = PPO("MlpPolicy",vec_env,verbose=1,batch_size=64)
    model.learn(total_timesteps=2500000)
    model.save("final2")

    # Don't forget to save your model

    raise NotImplementedError

if __name__ == "__main__":

    train(get_args())
