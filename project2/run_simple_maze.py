from simple_maze import SimpleMaze
import numpy as np
import time


def main():

	env = SimpleMaze(obs_type="poses", gui=True)
	
	while 1:
		for _ in range(20):
			obs, _, _, _ = env.step(np.random.randint(4))
		print('obs shape:', obs.shape)
		time.sleep(1)
		env.reset()
	input()

if __name__ == '__main__':

	main()