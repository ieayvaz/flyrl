import gymnasium as gym
import flyrl

env_id = "DogfightRascal-debug"
env = gym.make(env_id)

obs = env.reset()[0]
for _ in range(1000):
    obs, reward, done,_, info = env.step(env.action_space.sample())
    env.render()
    if done:
        print("Resetting")
        obs = env.reset()

# Close environment
env.close()