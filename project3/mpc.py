import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch

np.set_printoptions(suppress=True)


class MPC:

    def __init__(self, ):
        self.control_horizon = 10
        self.delta_u = 0.27
        self.repeat_n = 14
        # Define other parameters here

    def add_delta_u(self, action, i):
        u1 = action.copy()
        u1[i][0] += self.delta_u
        u2 = action.copy()
        u2[i][0] -= self.delta_u
        return u1, u2

    def compute_loss(self, dynamics, state, goal, static=False):
        if dynamics.num_links!=3:
            alpha = 1.2
            beta = 2.25
        else:
            alpha = 1.5
            beta = 3.3

        end_effector_pose = dynamics.compute_fk(state)
        cost_1 = math.sqrt(
            math.pow(goal[0, 0] - end_effector_pose[0, 0], 2) + math.pow(goal[1, 0] - end_effector_pose[1, 0], 2))

        if static:
            cost_2 = 0
        else:
            cost_2 = dynamics.compute_energy(state)[-1]

        return alpha * cost_1 + beta * cost_2

    def compute_action(self, dynamics, state, goal, action):
        # print(state)
        action_current = np.array(action)

        lowest_cost = self.compute_loss(dynamics, state, goal, static=True)

        total = 0
        while True:
            total += 1
            current_cost = np.zeros((dynamics.num_links * 2))
            for i in range(dynamics.num_links):
                u1, u2 = self.add_delta_u(action_current, i)
                new_state_1 = new_state_2 = state

                for _ in range(self.repeat_n):
                    new_state_1 = dynamics.advance(new_state_1, u1)
                    new_state_2 = dynamics.advance(new_state_2, u2)
                current_cost_1 = self.compute_loss(dynamics, new_state_1, goal)
                current_cost_2 = self.compute_loss(dynamics, new_state_2, goal)
                current_cost[i] = current_cost_1
                current_cost[i + dynamics.num_links] = current_cost_2

            if min(current_cost) >= lowest_cost:
                break
            else:
                lowest_cost = min(current_cost)
            index = current_cost.argmin()
            if index < dynamics.num_links:
                action_current[index][0] += self.delta_u
            else:
                action_current[index - dynamics.num_links][0] -= self.delta_u

        # print(action_current)

        return_action = action_current
        return return_action


def main(args):
    # Arm
    arm = Robot(
        ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
    )

    # Dynamics model used for control
    if args.model_path is not None:
        dynamics = ArmDynamicsStudent(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        dynamics.init_model(args.model_path, args.num_links, device=torch.device("cpu"))
    else:
        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )

    # Controller
    controller = MPC()

    # Control loop
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))
    goal[0, 0] = args.xgoal
    goal[1, 0] = args.ygoal
    arm.goal = goal

    if args.gui:
        renderer = Renderer()
        time.sleep(0.25)

    dt = args.time_step
    k = 0
    while True:
        t = time.time()
        arm.advance()
        if args.gui:
            renderer.plot([(arm, "tab:blue")])
        k += 1
        time.sleep(max(0, dt - (time.time() - t)))
        if k == controller.control_horizon:
            state = arm.get_state()
            action = controller.compute_action(dynamics, state, goal, action)
            print(action)
            arm.set_action(action)
            k = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_links", type=int, default=2)
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--gui", action="store_const", const=True, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--xgoal", type=float, default=-1.4)
    parser.add_argument("--ygoal", type=float, default=-1.4)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
