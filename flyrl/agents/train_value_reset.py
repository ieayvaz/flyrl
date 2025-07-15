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

# Configuration for sparse reward fine-tuning
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,  # Reduced for fine-tuning
    "env_name": "MultiDogfightRascal",
    "learning_rate": 1e-5,  # Much lower for fine-tuning
    "n_steps": 4096,  # Increased for sparse rewards
    "batch_size": 128,  # Larger batches for stability
    "n_epochs": 4,  # Fewer epochs to prevent overfitting
    "gamma": 0.995,  # Higher gamma for sparse rewards
    "gae_lambda": 0.95,
    "ent_coef": 0.001,  # Lower entropy for fine-tuning
    "vf_coef": 0.5,
    "max_grad_norm": 0.1,  # Lower gradient clipping
    "clip_range": 0.1,  # Smaller clip range for fine-tuning
    "policy_kwargs": {
        "net_arch": [256, 256, 128],
    }
}

# Create unique run ID for logging
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_sparse"
print(f"Starting sparse reward fine-tuning run: {run_id}")

continue_training = True
model_path = "./models/checkpoints_13-07-2025-Multi-2/checkpoint_400000_steps.zip"

def make_env():
    """Factory function to create environment instances"""
    def _init():
        env = gym.make(config["env_name"])
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Use fewer environments for fine-tuning (more stable)
    n_envs = min(mp.cpu_count(), 8)
    print(f"Using {n_envs} parallel environments")
    
    # Create environments
    env = SubprocVecEnv([make_env() for _ in range(n_envs)], start_method="spawn")
    eval_env = DummyVecEnv([make_env()])
    
    # Load and modify the model properly
    if continue_training and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        
        # Load the model first
        model = PPO.load(model_path, env=env, tensorboard_log=f"runs/{run_id}")
        
        # Create a new model with fine-tuning hyperparameters
        # This preserves the policy network but allows hyperparameter changes
        model_fine = PPO(
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
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=config["policy_kwargs"]
        )
        
        # Copy the policy parameters
        model_fine.policy.load_state_dict(model.policy.state_dict())
        
        # Reset the value function for new reward scale
        # This is crucial for sparse rewards!
        print("Resetting value function for sparse rewards...")
        for param in model_fine.policy.value_net.parameters():
            if param.dim() > 1:
                param.data.normal_(0, 0.01)
            else:
                param.data.fill_(0)
        
        model = model_fine
        
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
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            vf_coef=config["vf_coef"],
            max_grad_norm=config["max_grad_norm"],
            policy_kwargs=config["policy_kwargs"]
        )
    
    # Setup callbacks with different frequencies for fine-tuning
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/best_{run_id}",
        log_path=f"./logs/eval_{run_id}",
        eval_freq=8000,  # Less frequent evaluation
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # Less frequent checkpoints
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