import datetime
import flyrl
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpLstmPolicy",  # Changed to LSTM policy
    "total_timesteps": 1000000,
    "env_name": "DogfightRascal",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [256, 256, 128],  # Network architecture for LSTM
        "lstm_hidden_size": 256,      # LSTM hidden state size
        "n_lstm_layers": 2,           # Number of LSTM layers
        "shared_lstm": False,         # Whether to share LSTM between actor and critic
        "enable_critic_lstm": True,   # Enable LSTM for critic network
    }
}

run = wandb.init(
    project="dogfight-task-v1-lstm",  # Updated project name
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
    id="12-07-2025-15-lstm"  # Updated run ID
)

def make_env():
    env = gym.make(config["env_name"])
    env = Monitor(env)
    return env

# Use multiple environments for better sampling
n_envs = 4
env = DummyVecEnv([make_env for _ in range(n_envs)])  # Fixed syntax error

# Evaluation environment
eval_env = DummyVecEnv([make_env])

# Instantiate the RecurrentPPO agent with LSTM
model = RecurrentPPO(
    config["policy_type"],
    env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    learning_rate=config["learning_rate"],
    n_steps=config["n_steps"],
    batch_size=config["batch_size"],
    n_epochs=config["n_epochs"],
    gamma=config["gamma"],
    gae_lambda=config["gae_lambda"],
    clip_range=config["clip_range"],
    ent_coef=config["ent_coef"],
    vf_coef=config["vf_coef"],
    max_grad_norm=config["max_grad_norm"],
    policy_kwargs=config["policy_kwargs"]
)

# Setup callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/best_{run.id}",
    log_path=f"./logs/eval_{run.id}",
    eval_freq=10000,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=f"./models/checkpoints_{run.id}",
    name_prefix="checkpoint"
)

# Train with callbacks
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[eval_callback, checkpoint_callback],
)

run.finish()