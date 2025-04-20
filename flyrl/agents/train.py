from git import Object
import flyrl
import gymnasium as gym
import numpy as np
import math # For math.degrees
from stable_baselines3 import PPO
# from sb3_contrib import RecurrentPPO # Keep if you use it
# from stable_baselines3 import SAC # Keep if you use it
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv # Import VecEnv for type hint
from stable_baselines3.common.callbacks import BaseCallback # Import BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

# --- Custom Callback for Dogfight Metrics ---
class DogfightMetricsCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    It runs periodic evaluations on a separate environment and logs
    mean Angle-Off-Boresight (AOB) and Aspect Angle (AA).

    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should use stochastic or deterministic actions.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, eval_env: VecEnv,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 deterministic: bool = True,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        # Ensure eval_env is different from training_env
        # Note: This check might fail if make_env() always returns the exact same obj
        # It's mainly a conceptual check - they should be separate instances.
        # assert eval_env is not self.training_env, "Eval env should be different instance"


    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        # Check if it's time to evaluate
        # self.num_timesteps is the total steps taken across all environments
        # self.n_calls is the number of times _on_step has been called
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"Running evaluation on step {self.num_timesteps}...")
            self._run_evaluation()
        return True # Return True to continue training

    def _run_evaluation(self):
        """
        Runs evaluation episodes and logs mean AOB and AA.
        """
        all_aobs_rad = []
        all_aas_rad = []
        total_eval_steps = 0

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            terminated = False
            # Loop until episode finishes (terminated or truncated)
            while not (terminated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rews, terminateds, infos = self.eval_env.step(action) # Use VecEnv step

                # Handle VecEnv output (assuming DummyVecEnv, results are in lists of size 1)
                terminated = terminateds[0]
                info = infos[0]

                total_eval_steps += 1

                # --- Access Metrics from the *underlying* environment ---
                try:
                    # For DummyVecEnv -> Monitor -> DogfightTask:
                    # self.eval_env.envs[0] gets the Monitor wrapper
                    # .env gets the original DogfightTask instance
                    # If you have other wrappers, adjust the chain accordingly
                    monitor_env = self.eval_env.envs[0]  # Get Monitor wrapper
                    order_enforcing_env = monitor_env.env  # Unwrap Monitor → OrderEnforcing
                    passive_checker_env = order_enforcing_env.env  # Unwrap OrderEnforcing → PassiveEnvChecker
                    underlying_env = passive_checker_env.env  # Unwrap PassiveEnvChecker → Your actual environment
                    cache = underlying_env.task._kinematics_cache

                    # Use .get() with a default value in case cache isn't ready or key missing
                    aob_rad = cache.get('AOB', math.pi) # Default to max angle (180 deg)
                    aa_rad = cache.get('AA', math.pi)  # Default to max angle (180 deg)

                    all_aobs_rad.append(aob_rad)
                    all_aas_rad.append(aa_rad)

                except AttributeError as e:
                    # Handle cases where the structure is different or cache not ready
                    if total_eval_steps <= self.eval_env.num_envs: # Print only once per eval run start
                         print(f"Warning: Could not access _kinematics_cache in DogfightMetricsCallback. Error: {e}")
                    # Append default/neutral values if access fails to avoid crashing
                    all_aobs_rad.append(math.pi)
                    all_aas_rad.append(math.pi)
                except Exception as e: # Catch other potential errors
                     if total_eval_steps <= self.eval_env.num_envs:
                         print(f"Warning: Error accessing metrics in DogfightMetricsCallback. Error: {e}")
                     all_aobs_rad.append(math.pi)
                     all_aas_rad.append(math.pi)


        # --- Log aggregate metrics ---
        if total_eval_steps > 0:
            mean_aob_rad = np.mean(all_aobs_rad)
            mean_aa_rad = np.mean(all_aas_rad)
            mean_aob_deg = math.degrees(mean_aob_rad)
            mean_aa_deg = math.degrees(mean_aa_rad)

            if self.verbose > 0:
                print(f"Evaluation complete ({self.n_eval_episodes} episodes, {total_eval_steps} steps):")
                print(f"  Mean AOB: {mean_aob_deg:.2f} deg")
                print(f"  Mean AA: {mean_aa_deg:.2f} deg")

            # Log metrics using the logger provided by BaseCallback.
            # This logger integrates with TensorBoard and WandB (if WandbCallback is used).
            self.logger.record('eval/mean_AOB_deg', mean_aob_deg)
            self.logger.record('eval/mean_AA_deg', mean_aa_deg)

            # You could add other metrics here, e.g., % time below threshold
            aob_threshold_deg = 10.0
            aa_threshold_deg = 30.0
            time_aob_lt_thresh = np.mean(np.array(all_aobs_rad) < math.radians(aob_threshold_deg)) * 100
            time_aa_lt_thresh = np.mean(np.array(all_aas_rad) < math.radians(aa_threshold_deg)) * 100
            self.logger.record(f'eval/pct_time_AOB_lt_{aob_threshold_deg}', time_aob_lt_thresh)
            self.logger.record(f'eval/pct_time_AA_lt_{aa_threshold_deg}', time_aa_lt_thresh)

        else:
             if self.verbose > 0:
                print("Warning: Evaluation finished with zero total steps. Check environment termination logic.")

        # It's important to dump the logs, especially if using TensorBoard directly
        # WandbCallback might handle this automatically via sync_tensorboard
        self.logger.dump(step=self.num_timesteps)


# --- Training Script ---

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1000000,
    "env_name": "DogfightRascal", # Use the registered Env ID for your dogfight task
    "learning_rate": 3e-5, # Adjusted LR slightly higher, 3e-6 is very low for PPO
    "eval_freq": 5000,     # How often to run evaluation (in training steps)
    "n_eval_episodes": 5   # Number of episodes for each evaluation run
}

run = wandb.init(
    project="dogfight-task-v1",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,       # auto-upload the videos of agents playing the game (if eval env uses VideoRecorder)
    save_code=True,
)

# Create training environment function
def make_env():
    env = gym.make(config["env_name"])
    # Monitor should wrap the base env to record episode stats (reward, length, time)
    env = Monitor(env)
    return env

# Create training VecEnv
# Consider using SubprocVecEnv for potential speedup if JSBSim allows parallel instances
# from stable_baselines3.common.vec_env import SubprocVecEnv
# train_env = SubprocVecEnv([make_env for _ in range(num_cpu)]) # Example for parallel
train_env = DummyVecEnv([make_env])


# Create SEPARATE evaluation VecEnv
# Use the same make_env function but ensure it's a distinct instance
eval_env = DummyVecEnv([make_env])

# Instantiate the PPO agent
model = PPO(
    config["policy_type"],
    train_env, # Use the training environment here
    verbose=1,
    tensorboard_log=f"runs/{run.id}", # SB3 logs to TensorBoard, WandbCallback syncs it
    learning_rate=config["learning_rate"],
    # Add other hyperparameters like n_steps, batch_size, gamma, gae_lambda etc.
    # n_steps=2048,
    # batch_size=64,
    # gamma=0.99,
    # gae_lambda=0.95,
)

# --- Set up Callbacks ---
# 1. WandbCallback for general logging, model saving, gradients
wandb_callback = WandbCallback(
        gradient_save_freq=10000, # Log gradients less frequently
        model_save_path=f"models/{run.id}", # Save model checkpoints within wandb run
        model_save_freq=config["eval_freq"], # Save model whenever evaluation runs
        log="all", # Log histograms, gradients, etc.
        verbose=2,
    )

# 2. Our custom callback for logging AOB/AA during evaluation
# It uses the eval_env created earlier
metrics_callback = DogfightMetricsCallback(
    eval_env=eval_env,
    eval_freq=config["eval_freq"],
    n_eval_episodes=config["n_eval_episodes"],
    verbose=1
)

# Optional: Use EvalCallback for saving best model based on reward
# from stable_baselines3.common.callbacks import EvalCallback
# eval_callback_sb3 = EvalCallback(eval_env, best_model_save_path=f'models/{run.id}/best_model',
#                              log_path=f'models/{run.id}/eval_logs', eval_freq=config["eval_freq"],
#                              n_eval_episodes=config["n_eval_episodes"], deterministic=True, render=False)


# --- Train the agent with multiple callbacks ---
model.learn(
    total_timesteps=config["total_timesteps"],
    # Pass a list of callbacks to be executed in order
    callback=[wandb_callback, metrics_callback], # Add eval_callback_sb3 here if using
)

run.finish()

# Save final model manually as well (optional, wandb_callback saves periodically)
model.save(f"models/{run.name}/final_model")
print(f"Final model saved to models/{run.name}/final_model")

# Close environments
train_env.close()
eval_env.close()