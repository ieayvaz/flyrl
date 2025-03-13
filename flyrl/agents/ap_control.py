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
    
    target_roll = 0.0
    target_pitch = 2.0
    
    for i in range(100000):
        # Check for key press
        key = get_key_press()
        
        if key is not None:
            if key == 'd':
                if target_roll + SENSITIVITY * 4 <= 60:
                    target_roll += SENSITIVITY * 4
            elif key == 'a':
                if target_roll - SENSITIVITY * 4 >= -60:
                    target_roll -= SENSITIVITY * 4
            elif key == 's':
                if target_pitch + SENSITIVITY * 0.5 <= 10:
                    target_pitch += SENSITIVITY * 0.5
            elif key == 'w':
                if target_pitch - SENSITIVITY * 0.5 >= -10:
                    target_pitch -= SENSITIVITY * 0.5
            elif key == 'q':  # Added a key to quit the simulation
                break
                
        # Take a step in the environment
        print(f"target roll: {target_roll} , target_pitch : {target_pitch}")
        obs, reward, trunc, tr, info = env.step(np.array([target_roll, target_pitch]))
        env.render()
        
        if trunc or tr:
            break
            
finally:
    # Restore terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)