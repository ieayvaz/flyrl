import gymnasium as gym
import flyrl
from stable_baselines3 import PPO
import sys

env_id = "DogfightAP2P-debug"
env = gym.make(env_id)

model = PPO.load(sys.argv[1])

# Test the trained agent
obs = env.reset()[0]
for _ in range(100000):
    action, _states = model.predict(obs, deterministic=True)
    print(f"Action: {action}, State: {obs}")
    obs, reward, done,_, info = env.step(action)
    env.render()
    if info["success"] == True:
        print("PLAYER LOCKED SUCCESFULLY")
    if done:
        obs = env.reset()[0]

# Close environment
env.close()