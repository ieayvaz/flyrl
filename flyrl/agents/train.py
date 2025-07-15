from git import Object
import flyrl
import gymnasium as gym
import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# --- Training Script ---

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
    "env_name": "DogfightRascal",
    "learning_rate": 1.9010245319870364e-05,
    "eval_freq": 2048,     # How often to run evaluation (in training steps)
    "n_eval_episodes": 8,  # Number of episodes for each evaluation run
}

run = wandb.init(
    project="dogfight-task-v1",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# Create training environment function
def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)  # Wrap with Monitor to log episode stats
    return env

# Create training VecEnv
train_env = DummyVecEnv([make_env])

# Create SEPARATE evaluation VecEnv
eval_env = DummyVecEnv([make_env])

# Instantiate the PPO agent
model = PPO(
    config["policy_type"],
    train_env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    learning_rate=config["learning_rate"],
    batch_size=32,
    gamma=0.995,
    gae_lambda=0.98,
    ent_coef=0.04744427686266667,
    vf_coef=0.9725056264596474,
    clip_range=0.1,
    n_steps=2**9
)

# --- Set up Callbacks ---

# 1. WandbCallback for general logging, model saving, gradients
wandb_callback = WandbCallback(
    gradient_save_freq=10000,
    model_save_path=f"models/{run.id}",
    model_save_freq=config["eval_freq"],
    log="all",
    verbose=2,
)

# 2. EvalCallback for deterministic evaluation and logging
eval_callback_sb3 = EvalCallback(
    eval_env,
    best_model_save_path=f"models/{run.id}/best_model",
    log_path=f"models/{run.id}/eval_logs",
    eval_freq=config["eval_freq"],
    n_eval_episodes=config["n_eval_episodes"],
    deterministic=True,  # Ensure deterministic policy evaluation (no exploration)
    render=False,
    verbose=1,
    callback_after_eval=None  # Optional: Add custom logic after eval if needed
)

# --- Train the agent with multiple callbacks ---
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[wandb_callback, eval_callback_sb3],  # Include both callbacks
    progress_bar=True  # Optional: Add a progress bar for training
)

run.finish()

# Save final model manually
model.save(f"models/{run.name}/final_model")
print(f"Final model saved to models/{run.name}/final_model")

# Close environments
train_env.close()
eval_env.close()