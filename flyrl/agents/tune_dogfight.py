from typing import Any, Dict
import gymnasium as gym
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
import torch
import flyrl
import torch.nn as nn
import time
import datetime
import warnings
import pandas as pd # Import pandas

# --- Configuration Constants ---
N_TRIALS = 50
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 4
N_TIMESTEPS = int(6e4) # Reduced for faster testing, increase as needed
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 5

ENV_ID = "DogfightRascal"

STUDY_NAME = "ppo-dogfight-tuning-excel" # Updated study name
EXCEL_FILENAME = f"{STUDY_NAME}_results.xlsx" # Define output Excel filename

# --- Default Hyperparameters ---
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}

# --- Helper Function to Create Environment ---
def make_env():
    env = gym.make(ENV_ID)
    env = Monitor(env)
    return env

# --- Hyperparameter Sampler Function ---
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 8, 12)
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    vf_coef = trial.suggest_float("vf_coef", 0.2, 1.0)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])

    batch_size_options = [32, 64, 128, 256]
    valid_batch_sizes = [bs for bs in batch_size_options if n_steps % bs == 0]
    if not valid_batch_sizes:
        if n_steps >= 64: valid_batch_sizes = [64]
        else: valid_batch_sizes = [32]
        if n_steps % valid_batch_sizes[0] != 0:
             print(f"\nWarning: Forcing batch_size={n_steps} as no standard divisor found for n_steps={n_steps}")
             valid_batch_sizes = [n_steps]

    batch_size = trial.suggest_categorical("batch_size", valid_batch_sizes)

    return {
        "learning_rate": learning_rate, "n_steps": n_steps, "gamma": gamma,
        "gae_lambda": gae_lambda, "ent_coef": ent_coef, "vf_coef": vf_coef,
        "clip_range": clip_range, "batch_size": batch_size,
    }

# --- Custom Callback for Optuna Pruning & Logging ---
class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating, reporting trial to Optuna, and manual logging."""
    def __init__(
        self, eval_env: gym.Env, trial: optuna.Trial, n_eval_episodes: int = 5,
        eval_freq: int = 10000, deterministic: bool = True, verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq,
            deterministic=deterministic, verbose=verbose, best_model_save_path=None, log_path=None,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
        self._eval_start_time = None

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self._eval_start_time is None: self._eval_start_time = time.time()
            eval_start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{eval_start_str}] Trial {self.trial.number}: Starting Eval #{self.eval_idx + 1}/{N_EVALUATIONS} at step {self.num_timesteps}", flush=True)
            continue_training = super()._on_step()
            eval_duration = time.time() - self._eval_start_time if self._eval_start_time else 0
            self._eval_start_time = time.time()
            mean_reward = self.last_mean_reward
            self.eval_idx += 1
            print(f"    Trial {self.trial.number} Eval {self.eval_idx}/{N_EVALUATIONS} Complete. Mean Reward: {mean_reward:.3f} (Duration: {eval_duration:.2f}s)", flush=True)
            self.trial.report(mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                print(f"    Trial {self.trial.number} pruned by Optuna based on intermediate results.", flush=True)
                return False
        return True

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """Objective function to be maximized by Optuna."""
    global global_start_time
    current_trial_num = trial.number + 1
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{current_time_str}] === Starting Trial {current_trial_num} / {N_TRIALS} ===", flush=True)

    if current_trial_num > N_STARTUP_TRIALS:
        completed_trials = [t for t in trial.study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL) and t.number < trial.number]
        if completed_trials:
            durations = [(t.datetime_complete - t.datetime_start).total_seconds() for t in completed_trials if t.datetime_complete is not None and t.datetime_start is not None]
            if durations:
                avg_duration = np.mean(durations)
                remaining_trials = N_TRIALS - (current_trial_num -1)
                est_remaining_seconds = avg_duration * remaining_trials
                est_finish_time = datetime.datetime.now() + datetime.timedelta(seconds=est_remaining_seconds)
                print(f"  Est. Avg Trial Duration: {avg_duration:.2f}s. Est. Remaining Time: {datetime.timedelta(seconds=int(est_remaining_seconds))} (until ~{est_finish_time.strftime('%H:%M:%S')})", flush=True)

    trial_start_time = time.time()
    kwargs = DEFAULT_HYPERPARAMS.copy()
    sampled_params = sample_ppo_params(trial)
    kwargs.update(sampled_params)
    print(f"  Params: {sampled_params}", flush=True)

    env = None
    eval_env = None
    model = None
    eval_callback = None
    final_mean_reward = -float('inf')
    nan_or_inf_encountered = False
    try:
        env = make_env()
        model = PPO(env=env, tensorboard_log=None, verbose=0, **kwargs)
        eval_env = make_env()
        eval_callback = TrialEvalCallback(
            eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
        )
        print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] Trial {current_trial_num}: Starting training ({N_TIMESTEPS} steps)...", flush=True)
        model.learn(N_TIMESTEPS, callback=eval_callback)
        if not eval_callback.is_pruned:
            final_mean_reward = eval_callback.last_mean_reward
            print(f"  [{datetime.datetime.now().strftime('%H:%M:%S')}] Trial {current_trial_num}: Finished training. Final Mean Reward: {final_mean_reward:.3f}", flush=True)

    except (AssertionError, ValueError, FloatingPointError) as e:
        print(f"  Trial {current_trial_num} failed validation/runtime: {e}", flush=True)
        nan_or_inf_encountered = True
    except Exception as e:
         print(f"  Trial {current_trial_num} failed with unexpected error: {e}", flush=True)
         nan_or_inf_encountered = True
    finally:
        if model is not None and hasattr(model, 'env') and model.env is not None:
             try: model.env.close()
             except Exception: pass
        if eval_env is not None:
             try: eval_env.close()
             except Exception: pass
        trial_duration = time.time() - trial_start_time
        end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "PRUNED" if eval_callback and eval_callback.is_pruned else ("FAILED" if nan_or_inf_encountered else "COMPLETE")
        print(f"[{end_time_str}] === Finished Trial {current_trial_num} / {N_TRIALS}. Status: {status}. Duration: {trial_duration:.2f}s ===", flush=True)

    if nan_or_inf_encountered: return float("nan")
    if eval_callback and eval_callback.is_pruned: raise optuna.exceptions.TrialPruned()
    return final_mean_reward

# --- Main Execution Block ---
if __name__ == "__main__":
    global global_start_time
    global_start_time = time.time()

    print("="*70, flush=True)
    print(f"Starting Optuna study: {STUDY_NAME}", flush=True)
    print(f"  - Environment: {ENV_ID}", flush=True)
    print(f"  - Max Trials: {N_TRIALS}", flush=True)
    print(f"  - Training Steps per Trial: {N_TIMESTEPS}", flush=True)
    print(f"  - Pruning Evaluations per Trial: {N_EVALUATIONS}", flush=True)
    print(f"  - Evaluation Episodes per Eval: {N_EVAL_EPISODES}", flush=True)
    print(f"  - Output File: {EXCEL_FILENAME}", flush=True) # Indicate Excel output
    print("="*70, flush=True)

    # torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=42)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 2)

    # --- Create Study (In-Memory Storage) ---
    study = optuna.create_study(
        study_name=STUDY_NAME,
        # storage=STORAGE_URL, # REMOVED: Use in-memory storage
        # load_if_exists=True, # REMOVED: Cannot resume from memory
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
    )

    # --- Run Optimization ---
    start_opt_time = time.time()
    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=None,
            n_jobs=1,
            show_progress_bar=False # Explicitly disable tqdm bar
            )
    except KeyboardInterrupt:
        print("\nOptimization stopped manually.", flush=True)
    finally:
        elapsed_time = time.time() - start_opt_time
        print(f"\nTotal optimization loop time: {datetime.timedelta(seconds=int(elapsed_time))}", flush=True)


    # --- Process and Save Results to Excel ---
    print("\n" + "="*70, flush=True)
    print("Optimization Finished!", flush=True)
    print(f"Study Name: {study.study_name}", flush=True)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    fail_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"  Pruned trials: {len(pruned_trials)}", flush=True)
    print(f"  Completed trials: {len(complete_trials)}", flush=True)
    print(f"  Failed trials: {len(fail_trials)}", flush=True)


    if study.trials: # Check if any trials ran
        # Calculate average time
        all_durations = [(t.datetime_complete - t.datetime_start).total_seconds() for t in study.trials if t.datetime_complete is not None and t.datetime_start is not None]
        if all_durations:
             avg_time_per_trial = np.mean(all_durations)
             print(f"Average time per trial (incl. pruned/failed): {avg_time_per_trial:.2f} seconds", flush=True)
        else:
             print("Could not calculate average trial time.", flush=True)

        # --- Get best trial info ---
        print("\nBest trial:", flush=True)
        try:
            best_trial = study.best_trial
            print(f"  Number: {best_trial.number}", flush=True)
            print(f"  Value (Mean Reward): {best_trial.value:.4f}", flush=True)
            print("  Params: ", flush=True)
            for key, value in best_trial.params.items():
                print(f"    {key}: {value}", flush=True)
        except ValueError:
            print("  No best trial found (perhaps all failed or were pruned).", flush=True)

        # --- Create DataFrame and Save to Excel ---
        try:
            print(f"\nSaving results for all trials to {EXCEL_FILENAME}...", flush=True)
            df = study.trials_dataframe()
            # Make columns more readable if needed (e.g., split params)
            # df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
            df.to_excel(EXCEL_FILENAME, index=False, engine='openpyxl') # Specify engine
            print(f"Successfully saved results to {EXCEL_FILENAME}", flush=True)
        except ImportError:
            print("Error: Could not save to Excel. Make sure 'pandas' and 'openpyxl' are installed (`pip install pandas openpyxl`).", flush=True)
        except Exception as e:
            print(f"Error saving results to Excel: {e}", flush=True)

    else:
        print("\nNo trials were run or completed.", flush=True)

    print("="*70, flush=True)
    print("\nUse the 'Best trial' parameters printed above for final training.", flush=True)