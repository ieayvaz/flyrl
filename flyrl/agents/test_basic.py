import gymnasium as gym
import flyrl
from stable_baselines3 import PPO
import sys

env_id = "MultiDogfightRascal-debug"
env = gym.make(env_id)

model = PPO.load(sys.argv[1])

# Test the trained agent
total_rew = 0
total_eps = 0
obs = env.reset()[0]
for _ in range(3600):
    action,_ = model.predict(obs, deterministic=True)
    #action = env.action_space.sample()
    obs, reward, done,_, info = env.step(action)
    total_rew += reward
    env.render()
    if done:
        print(f"Total Reward: {total_rew}")
        obs = env.reset()[0]

# Close environment
env.close()