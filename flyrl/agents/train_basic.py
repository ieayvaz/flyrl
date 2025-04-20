import flyrl
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
    "env_name": "DogfightRascal",
    "learning_rate": 3e-6,
}
run = wandb.init(
    project="dogfight-task-v1",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

# Create environment
def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

# Instantiate the PPO agent
model = PPO(
    config["policy_type"],
    env,  # or vec_env if using vectorized envs
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    learning_rate=config["learning_rate"],
)

# Train the agent
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        verbose=2,
    ),
)

run.finish()

model.save(f"models/{run.name}/finish")