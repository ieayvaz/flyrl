import gymnasium as gym
import flyrl
from stable_baselines3 import PPO
import sys

env_id = "DogfightAP-debug"
env = gym.make(env_id)

model = PPO.load(sys.argv[1])

# Test the trained agent
obs = env.reset()[0]
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done,_, info = env.step(action)
    env.render()
    if done:
        print("Resetting.")
        obs = env.reset()[0]

# Close environment
env.close()