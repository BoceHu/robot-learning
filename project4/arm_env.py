import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from arm_gui import ArmGUI, Renderer
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import time

class ArmEnv:

    @staticmethod
    def cartesian_goal(radius,angle):
        return radius * np.array([np.cos(angle), np.sin(angle)]).reshape(-1,1)

    @staticmethod
    def random_goal():
        radius_max = 2.0
        radius_min = 1.5
        angle_max = 0.5
        angle_min = -0.5
        radius = (radius_max - radius_min) * np.random.random_sample() + radius_min
        angle = (angle_max - angle_min) * np.random.random_sample() + angle_min
        angle -= np.pi/2
        return ArmEnv.cartesian_goal(radius, angle)

    def __init__(self, args):

        num_links=2
        link_mass=0.1
        link_length=1
        friction=0.1
        self.timestep=0.01

        self.num_links = num_links
        self.arm = Robot(
            ArmDynamicsTeacher(
                num_links=num_links,
                link_mass=link_mass,
                link_length=link_length,
                joint_viscous_friction=friction,
                dt=self.timestep
            )
        )
        self.arm.reset()

        obs_shape = list(self.arm.state.shape)
        #account for goal in obs space
        obs_shape[0] = obs_shape[0] + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs_shape)
        
        self.num_steps = 0
        self.gui = args.gui
        if self.gui:
            self.renderer = Renderer()
            time.sleep(1)            
                        
    def step(self, action):
        self.num_steps += 1

        #decoded_action = self._decode_action(action)
        self.arm.set_action(action)

        for _ in range(1):
            self.arm.advance()

        if self.gui:
            self.renderer.plot([(self.arm, "tab:blue")])

        new_state = self.arm.get_state()

        # compute reward
        pos_ee = self.arm.dynamics.compute_fk(new_state)
        dist = np.linalg.norm(pos_ee - self.goal)
        vel_ee = np.linalg.norm(self.arm.dynamics.compute_vel_ee(new_state))
        reward = -dist**2

        done = False
        if self.num_steps >= 200:
            done = True
        info = dict(pos_ee=pos_ee, vel_ee=vel_ee, success=True)
        observation = np.append(new_state,self.goal).squeeze()
        return observation, reward, done, info
    
    def reset(self, goal=None):
        self.arm.reset()
        if goal is None:
            self.goal = ArmEnv.random_goal()
        else:
            self.goal = goal
        self.arm.goal = self.goal
        self.num_steps = 0
        return np.append(self.arm.get_state(),self.goal).squeeze()

    def render(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
