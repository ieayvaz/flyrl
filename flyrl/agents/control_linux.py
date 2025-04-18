import gymnasium as gym
import numpy as np
import flyrl
import termios
import tty
import sys
import select

def get_key_press():
    """Non-blocking keyboard input detection for Linux"""
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        return sys.stdin.read(1)
    return None

# Save terminal settings
old_settings = termios.tcgetattr(sys.stdin)
try:
    # Configure terminal for single character input without echo
    tty.setcbreak(sys.stdin.fileno())
    
    SENSITIVITY = 5
    env = gym.make('DogfightRascal')
    obs = env.reset()
    
    aileron_cmd = 0.0
    elevator_cmd = 0.0
    com = 1

    for i in range(1000):
        # Check for key press
        key = get_key_press()
        com = 1
        
        if key is not None:
            if key == 'd':
                com = 0
            elif key == 'a':
                com = 2
            elif key == 's':
                if elevator_cmd + SENSITIVITY * 0.1 < 1.0:
                    elevator_cmd += SENSITIVITY * 0.1
            elif key == 'w':
                if elevator_cmd - SENSITIVITY * 0.1 > -1.0:
                    elevator_cmd -= SENSITIVITY * 0.1
            elif key == 'q':  # Added a key to quit the simulation
                break
                
        obs, reward, trunc, tr, info = env.step(com)
        env.render()
        
        if trunc or tr:
            break
            
finally:
    # Restore terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)