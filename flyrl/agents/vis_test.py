from flyrl.visualizer_dogfight import DogfightVisualizer
import gymnasium as gym
import flyrl

env_id = "DogfightRascal-debug"
env = gym.make(env_id)
obs = env.reset()[0]
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done,_, info = env.step(action)
    env.render()
    if done:
        print("Resetting")
        obs = env.reset()

# Close environment
env.close()