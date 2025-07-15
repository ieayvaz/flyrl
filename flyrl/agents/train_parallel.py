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
import multiprocessing as mp
import os

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5000000,  # Increased for complex task
    "env_name": "MultiDogfightRascal",
    "learning_rate": 3e-4,
    "n_steps": 2048,  # Increased for better sample efficiency
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,  # Encourage exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [256, 256, 128],  # Larger network for complex task
    }
}

# Create unique run ID for logging
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Starting training run: {run_id}")

continue_training = False
model_path = "./models/checkpoints_13-07-2025-Multi-2/checkpoint_400000_steps.zip"

def make_env():
    """Factory function to create environment instances"""
    def _init():
        env = gym.make(config["env_name"])
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    # Set multiprocessing start method (important for parallel environments)
    mp.set_start_method('spawn', force=True)
    
    # Use multiple environments for better sampling
    n_envs = min(mp.cpu_count(), 8)  # Use available CPUs, max 8 for memory efficiency
    print(f"Using {n_envs} parallel environments")
    
    # Create parallel training environment using SubprocVecEnv for true parallelization
    env = SubprocVecEnv([make_env() for _ in range(n_envs)], start_method="spawn")
    
    # Evaluation environment (single environment is sufficient for evaluation)
    eval_env = DummyVecEnv([make_env()])
    
    # Instantiate the PPO agent
    if continue_training and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=env, tensorboard_log=f"runs/{run_id}")
        # Optional: override any hyperparameters if needed
        model.learning_rate = config["learning_rate"]
        model.n_steps = config["n_steps"]
        model.batch_size = config["batch_size"]
        model.n_epochs = config["n_epochs"]
        model.gamma = config["gamma"]
        model.gae_lambda = config["gae_lambda"]
        model.ent_coef = config["ent_coef"]
        model.vf_coef = config["vf_coef"]
        model.max_grad_norm = config["max_grad_norm"]
    else:
        print("Starting from scratch...")
        model = PPO(
            config["policy_type"],
            env,
            verbose=1,
            tensorboard_log=f"runs/{run_id}",
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=config["policy_kwargs"]
        )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{run_id}",
        log_path=f"./logs/eval_{run_id}",
        eval_freq=4000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/checkpoints_{run_id}",
        name_prefix="checkpoint"
    )
    
    try:
        # Train with callbacks
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[eval_callback, checkpoint_callback],
        )
        
        # Final evaluation
        print("Training completed. Running final evaluation...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Save final model
        model.save(f"./models/final_model_{run_id}")
        print(f"Final model saved to ./models/final_model_{run_id}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save(f"./models/interrupted_model_{run_id}")
        print(f"Model saved to ./models/interrupted_model_{run_id}")
    
    finally:
        # Clean up
        env.close()
        eval_env.close()
        print("Training session completed and resources cleaned up.")