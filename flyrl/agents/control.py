import flyrl
import gym
import numpy as np
import msvcrt

SENSIVITY = 5

env = gym.make('DogfightRascal-debug')

obs = env.reset()

for i in range(1000):
    aileron_cmd = 0.0
    elevator_cmd = 0.0
    rudder_cmd = 0.0
    if msvcrt.kbhit() == True:
        k = msvcrt.getch()
        c = k.decode()
        if(c == 'd'):
            if(aileron_cmd + SENSIVITY * 0.1 < 1.0):
                aileron_cmd += SENSIVITY * 0.1
        elif(c == 'a'):
            if(aileron_cmd - SENSIVITY * 0.1 > -1.0):
                aileron_cmd -= SENSIVITY * 0.1
        elif(c == 's'):
            if(elevator_cmd + SENSIVITY * 0.1 < 1.0):
                elevator_cmd += SENSIVITY * 0.1
        elif(c == 'w'):
            if(elevator_cmd - SENSIVITY * 0.1 > -1.0):
                elevator_cmd -= SENSIVITY * 0.1

    obs,reward,trunc,tr,info = env.step(np.array([aileron_cmd,elevator_cmd,rudder_cmd]))
    env.render()
    if trunc or tr:
        break
