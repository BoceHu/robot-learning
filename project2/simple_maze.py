
"""

SimpleMaze is square map of unit size consisting of obstacles.
The agent must navigate to the goal position.

The maze uses a co-ordinate sytem where x spans from 0 to 1 but 
y spans from -0.5, 0.5

"""

import time
import gym
import gym.spaces as spaces
import numpy as np
import pybullet as p
from copy import deepcopy

class Block:
    """ A simple rectangular block """
    def __init__(self, size, pos, rgba):
        self.size = size
        self.pos = pos
        self.rgba = rgba

    def check_collision(self, size, pos):
        """ check collision of given size """
        if pos[0] > self.pos[0] + self.size[0] + size[0]:  
            return False
        if pos[0] < self.pos[0] - self.size[0] - size[0]:
            return False
        if pos[1] > self.pos[1] + self.size[1] + size[1]:  
            return False
        if pos[1] < self.pos[1] - self.size[1] - size[1]:
            return False
        return True

RED     = [1., 0., 0., 1.]
BLACK   = [0., 0., 0., 1.]
GRAY1   = [.9, .9, .9, 1.]

WALLS = [
    Block(size=np.asarray([.01, 0.5]), pos=np.asarray([-.5, 0.]), rgba=BLACK),
    Block(size=np.asarray([.5, .01]), pos=np.asarray([.0, .5]), rgba=BLACK),
    Block(size=np.asarray([.5, .01]), pos=np.asarray([0., -.5]), rgba=BLACK),
    Block(size=np.asarray([.01, .5]), pos=np.asarray([.5, 0.]), rgba=BLACK),
]

class Map:

    def __init__(self):
        self._obstacles = []
        self._goal_spawn_pos = None
        self._agent_spawn_pos = None

    def add_obstacle(self, obstacle):
        self._obstacles.append(obstacle)

    def obstacles(self):
        return self._obstacles

    def reset_agent_and_goal(self):
        while True:
            self._goal_spawn_pos = self.np_random.uniform(low=-0.45, high=0.45, size=(2,))
            self._agent_spawn_pos = self.np_random.uniform(low=-0.45, high=0.45, size=(2,))
            if np.linalg.norm(self._goal_spawn_pos-self._agent_spawn_pos) < 0.1:
                continue
            collision = False
            for obstacle in self.obstacles():
                if obstacle.check_collision(size=[0.05, 0.05], pos=self._agent_spawn_pos):
                    collision = True
                if obstacle.check_collision(size=[0.05, 0.05], pos=self._goal_spawn_pos):
                    collision = True
            if not collision:
                return

    def get_goal_spawn_pos(self):
        if self._goal_spawn_pos is not None:
            return self._goal_spawn_pos
        else:
            raise ValueError('goal spwan pos not set')

    def get_agent_spawn_pos(self):
        if self._agent_spawn_pos is not None:
            return self._agent_spawn_pos
        else:
            raise ValueError('goal spwan pos not set')

    def set_random(self, random):
        """ set the RNG to be used for randomly choosing spawn positions """
        assert isinstance(random, np.random.RandomState)
        self.np_random = random



OBSTACLES = {
    'sq1': Block(size=np.asarray([.1, .1]), pos=np.asarray([-0.2, -0.1]), rgba=BLACK),
    'sq2': Block(size=np.asarray([.1, .1]), pos=np.asarray([0.2, 0.1]), rgba=BLACK),
    'hbar1': Block(size=np.asarray([0.35, 0.05]), pos=np.asarray([-.15, -0.15]), rgba=BLACK),
    'hbar2': Block(size=np.asarray([0.35, 0.05]), pos=np.asarray([0.15, 0.15]), rgba=BLACK),
    'vbar1': Block(size=np.asarray([0.05, 0.35]), pos=np.asarray([-0.15, -0.15]), rgba=BLACK),
    'vbar2': Block(size=np.asarray([0.05, 0.35]), pos=np.asarray([0.15, 0.15]), rgba=BLACK),
}


MAP1 = Map()
MAP1.add_obstacle(OBSTACLES['sq1'])
MAP1.add_obstacle(OBSTACLES['sq2'])

MAP2 = Map()
MAP2.add_obstacle(OBSTACLES['hbar1'])
MAP2.add_obstacle(OBSTACLES['hbar2'])

MAP3 = Map()
MAP3.add_obstacle(OBSTACLES['vbar1'])
MAP3.add_obstacle(OBSTACLES['vbar2'])

MAPS = [MAP1, MAP2, MAP3]

class SimpleMaze(gym.Env):
    
    def __init__(self, obs_type, maps=None, gui=False):
        super().__init__()
        
        if maps is None:
            self.maps = MAPS
        else:
            assert isinstance(maps, list)
            self.maps = [MAPS[m] for m in maps] 

        self.obs_type = obs_type # choose between image or position as observation
        self.gui = gui
        self.step_vel = 0.1
        self.time_step = 0.1
        self.imgH = 64
        self.imgW = 64
        if self.obs_type == 'rgb':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.imgH, self.imgW, 3), dtype=np.float32)
        elif self.obs_type == 'poses':
            self.observation_space = spaces.Box(low=-0.5, high=0.5, shape=(4, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
    
        self.seed(0)    # default seed = 0, use set_seed() to change
        if self.gui:
            self._physics_client = p.connect(p.GUI)  # graphical version
        else:
            self._physics_client = p.connect(p.DIRECT)  # non-graphical version

        self.reset()

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def reset(self):
        p.resetSimulation() # removes everything
        
        for wall in WALLS:
            self._load_body(wall, collision=True)
        self._reset_map()
        agent_spawn_pos = self.map.get_agent_spawn_pos() 
        self._load_agent(pos=agent_spawn_pos)
        for obstacle in self.map.obstacles():
            self._load_body(obstacle, collision=True)
        self._goal_pos = self.map.get_goal_spawn_pos()
        self._load_goal(pos=self._goal_pos)
        
        p.setTimeStep(self.time_step)
        self._setup_top_view()

        obs = self._get_obs()

        if self.obs_type == "rgb":
            obs = obs['rgb'].copy()
        elif self.obs_type == "poses":
            obs = np.hstack([obs['agent'], obs['goal']])

        return obs

    def _reset_map(self):
        """ select a map randomly and also select the goal and agent pos """
        self.map = deepcopy(self.np_random.choice(self.maps, 1, replace=False))[0] # deepcopy to support parallel env
        self.map.set_random(self.np_random)
        self.map.reset_agent_and_goal()

    def _load_agent(self, pos):
        self._agent_body_id = p.loadMJCF('mjcf/point_mass.xml')[0] # load the agent first 
        p.resetBasePositionAndOrientation(self._agent_body_id, [pos[0], pos[1], 0.01], [0.0, 0.0, 0.0, 1.0])

    def _load_goal(self, pos):
        goal = Block(size=np.asarray([0.05, 0.05]), pos=pos, rgba=RED)
        self._load_body(body=goal, collision=False)

    def _load_body(self, body, collision):

        pos = np.asarray([body.pos[0], body.pos[1], 0.025])
        size = [body.size[0], body.size[1], 0.025]

        if collision:
            collision_shape_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=size)
        else:
            collision_shape_id = -1

        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=size,
            rgbaColor=body.rgba)

        # create body in bullet and fix it
        body_id = p.createMultiBody(
            basePosition=[0., 0., 0.,],
            linkMasses=[1],
            linkCollisionShapeIndices=[collision_shape_id],
            linkVisualShapeIndices=[visual_shape_id],
            linkPositions=[pos],
            linkParentIndices=[0],
            linkInertialFramePositions=[pos],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkOrientations=[p.getQuaternionFromEuler([0, 0, 0])],
            linkJointTypes=[p.JOINT_FIXED],
            linkJointAxis=[[0, 0, 0]])

    def _setup_top_view(self):
        self.viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[0.0, 0, 1.3], 
            cameraTargetPosition=[0.0, 0, 0], 
            cameraUpVector=[0, 1, 0])

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0, 
            aspect=1.0, 
            nearVal=0.1, 
            farVal=3.1)

    def step(self, action):
        self._step(action)
        obs = self._get_obs()
        done = False
        dist = np.linalg.norm(obs['goal']-obs['agent'])
        if dist < 0.1:
            done = True

        # info contains everything (even observations)
        info = {
            'rgb': obs['rgb'].copy(),
            'goal': obs['goal'].copy(),
            'agent': obs['agent'].copy(),
            'success': done
        }

        # distance as reward 
        reward = -dist

        if self.obs_type == "rgb":
            obs = obs['rgb'].copy()
        elif self.obs_type == "poses":
            obs = np.hstack([obs['agent'], obs['goal']])

        if self.gui:
            time.sleep(0.05)
        
        return obs, reward, done, info

    def _step(self, action):
        # action    command     on maze
        # 0         Forward     y++
        # 1         Left        x--
        # 2         Right       x++
        # 3         Back        y--

        x_vel = 0
        y_vel = 0
        if action == 0:
            y_vel = self.step_vel
        elif action == 1:
            x_vel = -self.step_vel
        elif action == 2:
            x_vel = self.step_vel
        elif action == 3:
            y_vel = -self.step_vel

        p.setJointMotorControl2(self._agent_body_id, 0, p.VELOCITY_CONTROL, targetVelocity=x_vel)
        p.setJointMotorControl2(self._agent_body_id, 1, p.VELOCITY_CONTROL, targetVelocity=y_vel)
        for _ in range(10):
            p.stepSimulation()

    def _get_obs(self):
        # collect all obs
        obs = {
            'rgb': self._get_top_view(),
            'goal': self._get_goal_pos(),
            'agent': self._get_agent_pos()
            }
        return obs

    def _get_top_view(self):
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.imgW,
            height=self.imgH, 
            viewMatrix=self.viewMatrix,
            projectionMatrix=self.projectionMatrix)

        return rgbImg[..., :3].copy()/255.0

    def _get_agent_pos(self):
        return np.asarray(p.getLinkState(self._agent_body_id, 1)[4][:2])

    def _get_goal_pos(self):
        return self._goal_pos

    def close(self):
        p.disconnect()


    

    
