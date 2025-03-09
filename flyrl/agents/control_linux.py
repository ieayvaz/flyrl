import flyrl
import gymnasium as gym
import numpy as np
import sys
import tty
import termios
import select

SENSIVITY = 5

env = gym.make('DogfightRascal')
obs = env.reset()

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

for i in range(1000):
    aileron_cmd = 0.0
    elevator_cmd = 0.0
    rudder_cmd = 0.0
    
    if select.select([sys.stdin], [], [], 0)[0]:
        c = get_key()
        if c == 'd':
            if aileron_cmd + SENSIVITY * 0.1 < 1.0:
                aileron_cmd += SENSIVITY * 0.1
        elif c == 'a':
            if aileron_cmd - SENSIVITY * 0.1 > -1.0:
                aileron_cmd -= SENSIVITY * 0.1
        elif c == 's':
            if elevator_cmd + SENSIVITY * 0.1 < 1.0:
                elevator_cmd += SENSIVITY * 0.1
        elif c == 'w':
            if elevator_cmd - SENSIVITY * 0.1 > -1.0:
                elevator_cmd -= SENSIVITY * 0.1

    obs, reward, trunc,tr, info = env.step(np.array([aileron_cmd, elevator_cmd]))
    env.render()
    if trunc or tr:
        break