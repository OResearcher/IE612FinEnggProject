import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import time
import os

# Import the custom environment
from gbwm_env import GBWMEnv

# --- Configuration ---
# Environment Parameters
NUM_GOALS = 4
TIME_HORIZON = 16
INITIAL_WEALTH_FACTOR = 12.0
INITIAL_WEALTH_EXPONENT = 0.85
W_MAX = 1_000_000.0
MAX_STEPS = TIME_HORIZON + 5  # Safety parameter to prevent infinite running

# PPO Hyperparameters (Adjusted for better stability)
N_TRAJ = 50_000
LEARNING_RATE = 0.0003  # Reduced from 0.01 for better stability
CLIP_RANGE = 0.2  # More standard value
N_NEUR = 64

# SB3 Specific Hyperparameters
N_ENVS = 8
N_STEPS = 2048  # Steps per environment per update
BATCH_SIZE = 256  # Larger mini-batch size
N_EPOCHS = 10
GAMMA = 1.0
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
VF_COEF = 0.5

# Calculate total timesteps needed
TOTAL_TIMESTEPS = N_TRAJ * (TIME_HORIZON + 1)  # 50,000 * 17 = 850,000

# --- File Paths ---
LOG_DIR = "./ppo_gbwm_logs/"
MODEL_SAVE_PATH = "./ppo_gbwm_model"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(f"{LOG_DIR}/checkpoints", exist_ok=True)

# --- Custom Callback for Progress Tracking ---
class ProgressTrackerCallback(BaseCallback):
    """Custom callback for tracking episodes and timesteps"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.last_log_time = time.time()
        
    def _on_step(self):
        # Count completed episodes
        for done in self.locals.get("dones", []):
            if done:
                self.episode_count += 1
                
        # Log progress every 60 seconds
        current_time = time.time()
        if current_time - self.last_log_time > 60:
            progress_pct = (self.num_timesteps / TOTAL_TIMESTEPS) * 100
            print(f"Progress: {progress_pct:.1f}% | Episodes: {self.episode_count} | Steps: {self.num_timesteps}/{TOTAL_TIMESTEPS}")
            self.last_log_time = current_time
            
        return True

# --- Environment Setup ---
def make_env():
    def _init():
        env = GBWMEnv(
            num_goals=NUM_GOALS,
            time_horizon=TIME_HORIZON,
            initial_wealth_factor=INITIAL_WEALTH_FACTOR,
            initial_wealth_exponent=INITIAL_WEALTH_EXPONENT,
            w_max=W_MAX,
            max_steps=MAX_STEPS  # Add safety parameter to prevent infinite running
        )
        return env
    return _init

if __name__ == "__main__":
    try:
        print("Setting up vectorized environment...")
        
        # Create and monitor vectorized environment
        env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
        env = VecMonitor(env, f"{LOG_DIR}/monitor")
        
        print(f"Using {N_ENVS} parallel environments.")

        # Create callbacks for checkpointing and progress tracking
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,  # Save model every 50k steps
            save_path=f"{LOG_DIR}/checkpoints",
            name_prefix="ppo_gbwm_model"
        )
        
        progress_callback = ProgressTrackerCallback(verbose=1)
        
        # Define network architecture with tanh activation for stability
        policy_kwargs = dict(
            net_arch=dict(pi=[N_NEUR, N_NEUR], vf=[N_NEUR, N_NEUR]),
            activation_fn=torch.nn.Tanh
        )
        
        print("Defining PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=LOG_DIR,
            device="auto"
        )
        
        print("Model defined:")
        print(model.policy)
        
        print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
        start_time = time.time()
        
        # Try to run with tensorboard, catch error if not installed
        try:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=[checkpoint_callback, progress_callback],
                log_interval=10,
                tb_log_name="PPO_GBWM_Run"
            )
        except Exception as e:
            if "tensorboard is not installed" in str(e):
                print("\nTensorBoard error: Please install tensorboard with 'pip install tensorboard'")
                print("Continuing training without TensorBoard logging...\n")
                
                # Try again without tensorboard
                model = PPO(
                    "MlpPolicy", env, learning_rate=LEARNING_RATE, n_steps=N_STEPS,
                    batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, gamma=GAMMA,
                    gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE,
                    ent_coef=ENT_COEF, vf_coef=VF_COEF,
                    policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=None
                )
                
                model.learn(
                    total_timesteps=TOTAL_TIMESTEPS,
                    callback=[checkpoint_callback, progress_callback],
                    log_interval=10
                )
            else:
                raise e
                
        end_time = time.time()
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
        
        print(f"Saving trained model to {MODEL_SAVE_PATH}.zip")
        model.save(MODEL_SAVE_PATH)
        print("Model saved.")
        
    except Exception as e:
        print(f"Error during training: {e}")
    
    finally:
        # Clean up
        try:
            env.close()
        except:
            pass
        print("Environment closed.")
