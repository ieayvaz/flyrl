import gymnasium as gym
import numpy as np
import flyrl

env_id = "MultiDogfightRascal-debug"
env = gym.make(env_id)

obs = env.reset()[0]
for _ in range(100000):
    action = np.array([0,700,0.8])
    obs, reward, done,_, info = env.step(action)
    env.render()
    if done:
        print("Resetting")
        obs = env.reset()

# Close environment
env.close()