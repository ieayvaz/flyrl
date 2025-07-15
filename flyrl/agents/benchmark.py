import gymnasium as gym
import numpy as np
import flyrl
from stable_baselines3 import PPO
import sys


env_id = "MultiDogfightRascal"
env = gym.make(env_id)

obs = env.reset()[0]
total_rew = 0
total_eps = 0
win_count = 0
eps_rew = 0
#model = PPO.load(sys.argv[1])
for _ in range(10000):
    #action,_ = model.predict(obs)
    action = np.array(0)
    obs, reward, done,_, info = env.step(action)
    total_rew += reward
    eps_rew += reward
    if done:
        if eps_rew > 0:
            win_count += 1
        total_eps += 1
        eps_rew = 0
        obs = env.reset()[0]

print(f"Win rate: {win_count*100/total_eps}")
# Close environment
env.close()