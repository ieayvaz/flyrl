import gymnasium as gym
import flyrl
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import sys

env_id = "DogfightRascal-debug"
env = gym.make(env_id)

model = RecurrentPPO.load(sys.argv[1])

# Test the trained agent
obs = env.reset()[0]
for _ in range(3600):
    action,_ = model.predict(obs)
    #action = env.action_space.sample()
    print(f"Action: {action}")
    obs, reward, done,_, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()[0]

# Close environment
env.close()