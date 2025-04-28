import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os
import warnings

# Import the custom environment
from gbwm_env import GBWMEnv

# --- Configuration ---
# Environment Parameters (MUST match the environment used for training)
NUM_GOALS = 4
TIME_HORIZON = 16
INITIAL_WEALTH_FACTOR = 12.0
INITIAL_WEALTH_EXPONENT = 0.85
W_MAX = 1_000_000.0
MAX_STEPS = TIME_HORIZON + 5  # Same safety parameter as in training

# Evaluation Parameters
N_EVAL_EPISODES = 1000  # Number of episodes to run for evaluation
DETERMINISTIC_POLICY = True  # Use deterministic actions from the policy
EVALUATION_SEED = 42  # Seed for reproducibility
EPISODE_TIMEOUT = 100  # Maximum steps per episode (safety)

# --- File Paths ---
MODEL_LOAD_PATH = "./ppo_gbwm_model.zip"  # Path to the saved model
MONITOR_DIR = "./evaluation_logs/"
os.makedirs(MONITOR_DIR, exist_ok=True)

if __name__ == "__main__":
    try:
        # --- Create and prepare environment ---
        print("Creating evaluation environment...")
        env = GBWMEnv(
            num_goals=NUM_GOALS,
            time_horizon=TIME_HORIZON,
            initial_wealth_factor=INITIAL_WEALTH_FACTOR,
            initial_wealth_exponent=INITIAL_WEALTH_EXPONENT,
            w_max=W_MAX,
            max_steps=MAX_STEPS  # Add safety parameter
        )
        
        # Wrap environment with Monitor
        env = Monitor(env, MONITOR_DIR)
        
        # --- Load Model ---
        if not os.path.exists(MODEL_LOAD_PATH):
            print(f"Error: Model file not found at {MODEL_LOAD_PATH}")
            print("Please train the model first using train_ppo.py")
            exit()
        
        print(f"Loading trained model from {MODEL_LOAD_PATH}...")
        model = PPO.load(MODEL_LOAD_PATH, env=env)
        print("Model loaded.")
        
        # --- Option 1: Using SB3's evaluate_policy function ---
        print("\nRunning official SB3 evaluation...")
        try:
            mean_reward, std_reward = evaluate_policy(
                model, 
                env, 
                n_eval_episodes=min(100, N_EVAL_EPISODES),  # Use smaller sample for quick test
                deterministic=DETERMINISTIC_POLICY
            )
            print(f"Official evaluation results: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            print(f"Official evaluation failed: {e}")
            print("Continuing with custom evaluation...")
            
        # --- Option 2: Custom Evaluation Loop ---
        print(f"\nStarting custom evaluation for {N_EVAL_EPISODES} episodes...")
        
        all_episode_rewards = []
        all_final_wealths = []
        all_episode_lengths = []
        start_time = time.time()
        
        # Set seed for reproducibility
        env.reset(seed=EVALUATION_SEED)
        
        for episode in range(N_EVAL_EPISODES):
            # Reset environment at the start of each episode
            obs, info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            step_count = 0
            
            # Run a single episode
            while not (terminated or truncated):
                # Safety timeout
                if step_count >= EPISODE_TIMEOUT:
                    print(f"Warning: Episode {episode+1} timed out after {step_count} steps!")
                    truncated = True
                    break
                
                # Get action from the loaded policy
                action, _states = model.predict(obs, deterministic=DETERMINISTIC_POLICY)
                
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Accumulate reward and increment step counter
                episode_reward += reward
                step_count += 1
            
            # Store results for this episode
            all_episode_rewards.append(episode_reward)
            all_final_wealths.append(info['current_wealth'])
            all_episode_lengths.append(step_count)
            
            # Print progress more frequently
            if (episode + 1) % max(1, N_EVAL_EPISODES // 20) == 0:
                completed_pct = 100 * (episode + 1) / N_EVAL_EPISODES
                print(f" Completed: {completed_pct:.1f}% ({episode + 1}/{N_EVAL_EPISODES} episodes)")
        
        end_time = time.time()
        print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")
        
        # --- Calculate and Print Results ---
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        median_reward = np.median(all_episode_rewards)
        
        mean_final_wealth = np.mean(all_final_wealths)
        std_final_wealth = np.std(all_final_wealths)
        median_final_wealth = np.median(all_final_wealths)
        
        mean_episode_length = np.mean(all_episode_lengths)
        
        print("\n--- Evaluation Results ---")
        print(f"Number of episodes: {N_EVAL_EPISODES}")
        print(f"Policy type: {'Deterministic' if DETERMINISTIC_POLICY else 'Stochastic'}")
        print(f"Average episode length: {mean_episode_length:.2f} steps")
        
        print(f"\nAccumulated Utility (Reward):")
        print(f" Mean: {mean_reward:.2f}")
        print(f" Median: {median_reward:.2f}")
        print(f" Std Dev: {std_reward:.2f}")
        
        print(f"\nFinal Wealth:")
        print(f" Mean: {mean_final_wealth:.2f}")
        print(f" Median: {median_final_wealth:.2f}")
        print(f" Std Dev: {std_final_wealth:.2f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
    
    finally:
        # Clean up
        try:
            env.close()
            print("\nEnvironment closed.")
        except:
            pass


# import gymnasium as gym
# import numpy as np
# from stable_baselines3 import PPO
# import time
# import os

# # Import the custom environment
# from gbwm_env import GBWMEnv # Assuming gbwm_env.py is in the same directory

# # --- Configuration ---
# # Environment Parameters (MUST match the environment used for training)
# NUM_GOALS = 4
# TIME_HORIZON = 16
# INITIAL_WEALTH_FACTOR = 12.0
# INITIAL_WEALTH_EXPONENT = 0.85
# W_MAX = 1_000_000.0

# # Evaluation Parameters
# N_EVAL_EPISODES = 1000 # Number of episodes to run for evaluation (paper uses 100k)
# # Use a smaller number for quicker testing, increase for more robust results.
# DETERMINISTIC_POLICY = True # Use deterministic actions from the policy

# # --- File Paths ---
# MODEL_LOAD_PATH = "./ppo_gbwm_model.zip" # Path to the saved model

# # --- Environment Setup ---
# # Create a single instance of the environment for evaluation
# env = GBWMEnv(num_goals=NUM_GOALS,
#               time_horizon=TIME_HORIZON,
#               initial_wealth_factor=INITIAL_WEALTH_FACTOR,
#               initial_wealth_exponent=INITIAL_WEALTH_EXPONENT,
#               w_max=W_MAX)

# if __name__ == "__main__":
#     # --- Load Model ---
#     if not os.path.exists(MODEL_LOAD_PATH):
#         print(f"Error: Model file not found at {MODEL_LOAD_PATH}")
#         print("Please train the model first using train_ppo.py")
#         exit()

#     print(f"Loading trained model from {MODEL_LOAD_PATH}...")
#     # No need to pass env to load if parameters are saved with the model (SB3 default)
#     # However, good practice to ensure consistency if custom envs are involved.
#     # We pass custom_objects if the environment itself isn't automatically known/registered
#     # For simple class imports like this, it often works without, but being explicit is safer.
#     # model = PPO.load(MODEL_LOAD_PATH, env=env) # Option 1: Pass env instance
#     model = PPO.load(MODEL_LOAD_PATH, custom_objects={'action_space': env.action_space, 'observation_space': env.observation_space}) # Option 2: Pass spaces if needed

#     print("Model loaded.")

#     # --- Evaluation Loop ---
#     print(f"Starting evaluation for {N_EVAL_EPISODES} episodes...")
#     all_episode_rewards = []
#     all_final_wealths = []
#     start_time = time.time()

#     for episode in range(N_EVAL_EPISODES):
#         obs, info = env.reset()
#         terminated = False
#         truncated = False
#         episode_reward = 0.0

#         while not terminated and not truncated:
#             # Get action from the loaded policy
#             action, _states = model.predict(obs, deterministic=DETERMINISTIC_POLICY)

#             # Step the environment
#             obs, reward, terminated, truncated, info = env.step(action)

#             # Accumulate reward
#             episode_reward += reward

#         # Store results for this episode
#         all_episode_rewards.append(episode_reward)
#         all_final_wealths.append(info['current_wealth'])

#         if (episode + 1) % (N_EVAL_EPISODES // 10) == 0: # Print progress
#              print(f"  Completed episode {episode + 1}/{N_EVAL_EPISODES}")

#     end_time = time.time()
#     print(f"Evaluation finished in {end_time - start_time:.2f} seconds.")

#     # --- Calculate and Print Results ---
#     mean_reward = np.mean(all_episode_rewards)
#     std_reward = np.std(all_episode_rewards)
#     median_reward = np.median(all_episode_rewards)

#     mean_final_wealth = np.mean(all_final_wealths)
#     std_final_wealth = np.std(all_final_wealths)
#     median_final_wealth = np.median(all_final_wealths)

#     print("\n--- Evaluation Results ---")
#     print(f"Number of episodes: {N_EVAL_EPISODES}")
#     print(f"Policy type: {'Deterministic' if DETERMINISTIC_POLICY else 'Stochastic'}")
#     print(f"\nAccumulated Utility (Reward):")
#     print(f"  Mean:   {mean_reward:.2f}")
#     print(f"  Median: {median_reward:.2f}")
#     print(f"  Std Dev:{std_reward:.2f}")
#     print(f"\nFinal Wealth:")
#     print(f"  Mean:   {mean_final_wealth:.2f}")
#     print(f"  Median: {median_final_wealth:.2f}")
#     print(f"  Std Dev:{std_final_wealth:.2f}")

#     # Note: The paper calculates "RL Efficiency" by comparing this mean_reward
#     # to the optimal reward obtained via Dynamic Programming (DP).
#     # Implementing the DP solver is complex and not included here.
#     # This script provides the RL agent's average performance.
#     print("\nNote: To calculate 'RL Efficiency' as in the paper,")
#     print("compare the Mean Accumulated Utility with the result from a DP solver.")

#     # --- Clean up ---
#     env.close()
#     print("\nEnvironment closed.")

