import gymnasium as gym
import flyrl
import numpy as np

env_id = "DogfightAP2P-debug"
env = gym.make(env_id)

obs = env.reset()[0]
for _ in range(1000):
    action = np.array([0,0.5,0])
    obs, reward, done,_, info = env.step(action)
    env.render()
    if done:
        print("Resetting")
        obs = env.reset()

# Close environment
env.close()