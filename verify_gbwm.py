# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import os
# import seaborn as sns
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import load_results

# from gbwm_env import GBWMEnv

# def create_verification_plots(
#     model_path="./ppo_gbwm_model.zip",
#     log_dir="./ppo_gbwm_logs/",
#     output_dir="./verification_plots",
#     n_evaluation_episodes=50
# ):
#     """
#     Create fundamental verification plots to demonstrate model training and performance.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load environment and model
#     env = GBWMEnv()  # Default parameters will be used
#     num_goals = env.num_goals
#     time_horizon = env.time_horizon
    
#     model = PPO.load(
#         model_path,
#         env=env,
#         custom_objects={
#             'action_space': env.action_space,
#             'observation_space': env.observation_space
#         }
#     )
    
#     # PLOT 1: TRAINING TIMESTEPS VERIFICATION
#     try:
#         # Set style for better looking plots
#         sns.set_style("whitegrid")
        
#         training_data = load_results(log_dir)
#         timesteps = np.array(np.cumsum(training_data['l']), dtype=float)
#         rewards = np.array(training_data['r'], dtype=float)
#         total_timesteps = timesteps[-1] if len(timesteps) > 0 else 0
#         target_timesteps = 50000 * time_horizon  # From paper
        
#         plt.figure(figsize=(12, 7))
        
#         # Plot individual episode rewards with low opacity
#         plt.scatter(timesteps, rewards, s=10, alpha=0.15, color='#ff7f0e', label='Individual Episodes')
        
#         # Plot multiple smoothing levels for better trend visibility
#         window_sizes = [10, 50, 100]
#         colors = ['#1f77b4', '#2ca02c', '#d62728']
#         labels = ['10-Episode Average', '50-Episode Average', '100-Episode Average']
        
#         for i, window in enumerate(window_sizes):
#             if len(rewards) > window:
#                 smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
#                 plt.plot(timesteps, smoothed, linewidth=1.5+i*0.5, color=colors[i], label=labels[i])
        
#         # Add target timesteps line
#         plt.axvline(x=target_timesteps, color='red', linestyle='--', 
#                    label=f'Target Timesteps ({target_timesteps:,})')
        
#         # Calculate completion percentage
#         completion_pct = min(100, (total_timesteps / target_timesteps) * 100)
#         completion_status = "✓ COMPLETED" if total_timesteps >= target_timesteps else "IN PROGRESS"
        
#         # Add training status box
#         plt.annotate(
#             f"Training Status: {completion_status}\n"
#             f"Total Steps: {total_timesteps:,} ({completion_pct:.1f}% of target)\n"
#             f"Total Episodes: {len(rewards):,}",
#             xy=(0.02, 0.95), xycoords='axes fraction',
#             fontsize=12, fontweight='bold',
#             bbox=dict(boxstyle="round,pad=0.5", fc="#e8f4f8", ec="gray", alpha=0.9)
#         )
        
#         plt.title('GBWM Training Verification', fontsize=16)
#         plt.xlabel('Timesteps', fontsize=14)
#         plt.ylabel('Episode Reward', fontsize=14)
#         plt.legend(loc='lower right')
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/training_verification.png", dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"Created training verification plot showing {total_timesteps:,}/{target_timesteps:,} timesteps")
#     except Exception as e:
#         print(f"Could not create training verification plot: {e}")
    
#     # PLOT 2: GOAL ACHIEVEMENT DEMONSTRATION
#     # Compare trained policy vs random policy for clear performance visualization
#     policies = {
#         'Trained Model': lambda obs: model.predict(obs, deterministic=True)[0],
#         'Random Policy': lambda obs: env.action_space.sample()
#     }
    
#     results = {}
    
#     # Run evaluations for each policy
#     for policy_name, action_fn in policies.items():
#         goal_counts = {t: 0 for t in env.goal_times}
#         total_rewards = []
#         final_wealths = []
        
#         for _ in range(n_evaluation_episodes):
#             obs, _ = env.reset()
#             done = False
#             truncated = False
#             episode_reward = 0
            
#             while not (done or truncated):
#                 action = action_fn(obs)
#                 obs, reward, done, truncated, info = env.step(action)
#                 episode_reward += reward
                
#                 # Track goal achievements
#                 if reward > 0:
#                     goal_time = info['current_time'] - 1  # Adjust for time increment in step()
#                     if goal_time in goal_counts:
#                         goal_counts[goal_time] += 1
            
#             total_rewards.append(episode_reward)
#             final_wealths.append(info['current_wealth'])
        
#         # Calculate statistics
#         results[policy_name] = {
#             'avg_reward': np.mean(total_rewards),
#             'std_reward': np.std(total_rewards),
#             'avg_wealth': np.mean(final_wealths),
#             'goal_achievement': goal_counts,
#             'success_rate': sum(1 for r in total_rewards if r > 0) / n_evaluation_episodes * 100
#         }
    
#     # Create goal achievement comparison plot
#     plt.figure(figsize=(12, 8))
    
#     # Setup for side-by-side bars
#     goal_times = sorted(env.goal_times)
#     x = np.arange(len(goal_times))
#     width = 0.35
    
#     # Calculate achievement rates
#     trained_rates = [results['Trained Model']['goal_achievement'][t] / n_evaluation_episodes * 100 
#                     for t in goal_times]
#     random_rates = [results['Random Policy']['goal_achievement'][t] / n_evaluation_episodes * 100 
#                    for t in goal_times]
    
#     # Create main bar chart
#     ax = plt.subplot(2, 1, 1)
#     trained_bars = ax.bar(x - width/2, trained_rates, width, label='Trained Model', 
#                           color='green', alpha=0.7)
#     random_bars = ax.bar(x + width/2, random_rates, width, label='Random Policy', 
#                          color='gray', alpha=0.7)
    
#     # Add percentage labels on bars
#     for bar, rate in zip(trained_bars, trained_rates):
#         if rate > 5:  # Only label if bar is big enough to see
#             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
#                    f"{rate:.0f}%", ha='center', fontsize=9)
    
#     ax.set_title('Goal Achievement by Time Period', fontsize=16)
#     ax.set_xlabel('Goal Time Period', fontsize=12)
#     ax.set_ylabel('Achievement Rate (%)', fontsize=12)
#     ax.set_xticks(x)
#     ax.set_xticklabels(goal_times)
#     ax.set_ylim(0, 100)
#     ax.grid(axis='y', alpha=0.3)
#     ax.legend(loc='upper right')
    
#     # Add summary metrics comparison
#     ax2 = plt.subplot(2, 1, 2)
    
#     # Compare key metrics
#     metrics = ['Average Reward', 'Goals Achieved', 'Final Wealth']
    
#     # Calculate values for comparison
#     trained_values = [
#         results['Trained Model']['avg_reward'],
#         sum(results['Trained Model']['goal_achievement'].values()) / n_evaluation_episodes,
#         results['Trained Model']['avg_wealth'] / env.initial_wealth  # Normalize to initial wealth
#     ]
    
#     random_values = [
#         results['Random Policy']['avg_reward'],
#         sum(results['Random Policy']['goal_achievement'].values()) / n_evaluation_episodes,
#         results['Random Policy']['avg_wealth'] / env.initial_wealth  # Normalize to initial wealth
#     ]
    
#     # Calculate improvement percentages
#     improvements = []
#     for t, r in zip(trained_values, random_values):
#         imp = ((t - r) / max(0.01, abs(r))) * 100 if r != 0 else 100
#         improvements.append(imp)
    
#     # Create bar chart
#     x = np.arange(len(metrics))
#     trained_bars = ax2.bar(x - width/2, trained_values, width, label='Trained Model', 
#                           color='green', alpha=0.7)
#     random_bars = ax2.bar(x + width/2, random_values, width, label='Random Policy', 
#                          color='gray', alpha=0.7)
    
#     # Add improvement annotations
#     for i, (t_bar, imp) in enumerate(zip(trained_bars, improvements)):
#         if metrics[i] == 'Final Wealth':
#             # For wealth show as multiple of initial wealth
#             ax2.text(t_bar.get_x() + t_bar.get_width()/2, t_bar.get_height() + 0.05, 
#                     f"{trained_values[i]:.1f}x", ha='center')
#         else:
#             ax2.text(t_bar.get_x() + t_bar.get_width()/2, t_bar.get_height() + 0.05, 
#                     f"{trained_values[i]:.1f}", ha='center')
        
#         # Add improvement arrow
#         if imp > 0:
#             ax2.annotate(
#                 f"+{imp:.0f}%", 
#                 xy=(i, max(trained_values[i], random_values[i]) + 0.2),
#                 ha='center', color='red', fontweight='bold', fontsize=10
#             )
    
#     ax2.set_title('Performance Metrics Comparison', fontsize=16)
#     ax2.set_ylabel('Value', fontsize=12)
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(metrics, fontsize=12)
#     ax2.grid(axis='y', alpha=0.3)
    
#     # Add model verification summary
#     avg_goals_trained = sum(results['Trained Model']['goal_achievement'].values()) / n_evaluation_episodes
#     avg_goals_random = sum(results['Random Policy']['goal_achievement'].values()) / n_evaluation_episodes
#     goal_improvement = ((avg_goals_trained - avg_goals_random) / max(0.01, avg_goals_random)) * 100
    
#     plt.figtext(
#         0.5, 0.01,
#         f"Model Verification Summary: Trained on {total_timesteps:,} timesteps with {num_goals} goals\n"
#         f"Average goals achieved: {avg_goals_trained:.2f}/{num_goals} ({goal_improvement:.0f}% better than random)",
#         fontsize=12, ha='center', bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray")
#     )
    
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.12)  # Make room for summary text
#     plt.savefig(f"{output_dir}/goal_achievement_verification.png", dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Created goal achievement verification showing {avg_goals_trained:.2f}/{num_goals} goals achieved")
#     return True

# if __name__ == "__main__":
#     create_verification_plots(
#         model_path="./ppo_gbwm_model.zip",
#         log_dir="./ppo_gbwm_logs/",
#         output_dir="./verification_plots"
#     )

#to create env parameters table#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import load_results

# Import the custom environment
from gbwm_env import GBWMEnv

def verify_model_replication(
    model_path="./ppo_gbwm_model.zip",
    log_dir="./ppo_gbwm_logs/",
    num_goals=16,
    time_horizon=16,
    initial_wealth_factor=12.0,
    initial_wealth_exponent=0.85,
    n_eval_episodes=100,
    output_dir="./verification_plots",
):
    """
    Generate plots to verify model replication against the paper specifications.
    
    Args:
        model_path: Path to the trained model
        log_dir: Directory containing training logs
        num_goals: Number of goals in the environment
        time_horizon: Time horizon for planning
        n_eval_episodes: Number of episodes for evaluation
        output_dir: Directory to save verification plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the trained model and set up environment
    print(f"Loading model from {model_path}...")
    env = GBWMEnv(
        num_goals=num_goals,
        time_horizon=time_horizon,
        initial_wealth_factor=initial_wealth_factor,
        initial_wealth_exponent=initial_wealth_exponent,
    )
    
    model = PPO.load(
        model_path,
        env=env,
        custom_objects={
            'action_space': env.action_space,
            'observation_space': env.observation_space
        }
    )
    
    # 2. VERIFY TRAINING STEPS
    # Extract training data from Monitor files
    print("Analyzing training progress...")
    try:
        training_data = load_results(log_dir)
        
        # Calculate total timesteps
        total_timesteps = training_data['l'].sum()
        episode_count = len(training_data)
        
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(training_data['l']), training_data['r'], label='Episode Reward')
        plt.plot(np.cumsum(training_data['l']), 
                 pd.Series(training_data['r']).rolling(10).mean(), 
                 label='Moving Average (10 episodes)')
        
        # Highlight the target timesteps from the paper
        target_timesteps = 50000 * time_horizon  # N_TRAJ * TIME_HORIZON
        if total_timesteps >= target_timesteps:
            plt.axvline(x=target_timesteps, color='r', linestyle='--', 
                       label=f'Target Timesteps ({target_timesteps})')
        
        
        print(f"Trained for {total_timesteps:,} steps across {episode_count} episodes")
        print(f"Paper target: {target_timesteps:,} steps")
    except Exception as e:
        print(f"Could not load training data from {log_dir}: {e}")
        print("Skipping training progress plot.")
    
    # 3. VERIFY ENVIRONMENT PARAMETERS
    print("\nVerifying environment parameters...")
    
    # Extract environment parameters
    env_params = {
        "Number of Goals": env.num_goals,
        "Time Horizon": env.time_horizon,
        "Initial Wealth Factor": env.initial_wealth_factor,
        "Initial Wealth Exponent": env.initial_wealth_exponent,
        "Initial Wealth": env.initial_wealth,
        "Number of Portfolios": env.n_portfolios,
        "PPO Learning Rate": 0.01,  # From train_ppo.py
        "PPO Clip Range": 0.5,      # From train_ppo.py
        "Neural Network Size": "64x64", # From train_ppo.py
        "Batch Size": 4800,          # From train_ppo.py
    }
    
    # Paper parameters for comparison
    paper_params = {
        "Number of Goals": "1, 2, 4, 8, or 16",
        "Time Horizon": "16",
        "Initial Wealth Factor": "12.0",
        "Initial Wealth Exponent": "0.85",
        "Initial Wealth": f"{12.0 * (num_goals ** 0.85):.2f}",
        "Number of Portfolios": "15",
        "PPO Learning Rate": "0.01 (η)",
        "PPO Clip Range": "0.50 (ε)",
        "Neural Network Size": "64x64 (N_neur)",
        "Batch Size": "4800 (M)",
    }
    
    # Format as a table for plotting
    param_df = pd.DataFrame({
        'Parameter': list(env_params.keys()),
        'Implemented': [str(val) for val in env_params.values()],
        'Paper': list(paper_params.values())
    })
    
    # Create a figure for parameters
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    tbl = plt.table(
        cellText=param_df.values,
        colLabels=param_df.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f5f5f5', '#e5e5e5', '#e5e5e5']
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.5)
    plt.title('Environment Parameter Verification', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/environment_parameters.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. DEMONSTRATE IMPROVED REWARDS WITH TRAINING
    print("\nEvaluating model performance...")
    
    # Function to evaluate a model
    def evaluate_model(model, env, n_episodes=100):
        rewards = []
        final_wealths = []
        goals_taken = []
        
        for i in range(n_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_goals = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                if reward > 0:
                    episode_goals += 1
            
            rewards.append(episode_reward)
            final_wealths.append(info['current_wealth'])
            goals_taken.append(episode_goals)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_final_wealth': np.mean(final_wealths),
            'mean_goals_taken': np.mean(goals_taken),
        }
    
    # Try to load checkpoints to demonstrate improvement
    checkpoint_dir = f"{log_dir}/checkpoints"
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for file in sorted(os.listdir(checkpoint_dir)):
            if file.endswith(".zip"):
                checkpoints.append(os.path.join(checkpoint_dir, file))
    
    # If no checkpoints found, evaluate random policy vs trained
    if not checkpoints:
        print("No checkpoints found. Comparing random policy to trained model.")
        
        # Evaluate random policy
        random_rewards = []
        for i in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            random_rewards.append(episode_reward)
        
        random_mean = np.mean(random_rewards)
        
        # Evaluate trained policy
        trained_results = evaluate_model(model, env, n_eval_episodes)
        
        # Create comparison plot
        plt.figure(figsize=(8, 6))
        
        labels = ['Random Policy', 'Trained Policy']
        means = [random_mean, trained_results['mean_reward']]
        
        bars = plt.bar(labels, means, width=0.6)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        # Calculate improvement percentage
        improvement = ((trained_results['mean_reward'] - random_mean) / abs(random_mean)) * 100 if random_mean != 0 else 0
        plt.title(f'Policy Comparison\nImprovement: {improvement:.1f}%')
        plt.ylabel('Average Reward')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f"{output_dir}/policy_comparison.png", dpi=300)
        plt.close()
        
    else:
        # Evaluate performance at different checkpoints
        print(f"Evaluating {len(checkpoints)} checkpoints...")
        checkpoint_results = []
        
        # Sample a subset of checkpoints if there are too many
        if len(checkpoints) > 5:
            indices = np.linspace(0, len(checkpoints)-1, 5, dtype=int)
            sampled_checkpoints = [checkpoints[i] for i in indices]
        else:
            sampled_checkpoints = checkpoints
        
        for checkpoint in sampled_checkpoints:
            cp_model = PPO.load(
                checkpoint,
                env=env,
                custom_objects={
                    'action_space': env.action_space,
                    'observation_space': env.observation_space
                }
            )
            
            # Extract timestep from filename (assuming format like model_50000_steps.zip)
            try:
                timestep = int(os.path.basename(checkpoint).split('_')[1])
            except:
                timestep = 0
            
            results = evaluate_model(cp_model, env, n_eval_episodes)
            results['timestep'] = timestep
            checkpoint_results.append(results)
        
        # Create DataFrame for easy plotting
        results_df = pd.DataFrame(checkpoint_results)
        
        # Plot reward improvement over timesteps
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            results_df['timestep'], 
            results_df['mean_reward'],
            yerr=results_df['std_reward'],
            marker='o',
            linestyle='-',
            capsize=4
        )
        
        plt.title('Reward Improvement During Training')
        plt.xlabel('Training Timesteps')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/reward_improvement.png", dpi=300)
        plt.close()
        
        # Plot goals taken improvement
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['timestep'], results_df['mean_goals_taken'], 
                marker='o', linestyle='-')
        
        plt.title(f'Average Goals Achieved (out of {num_goals})')
        plt.xlabel('Training Timesteps')
        plt.ylabel('Average Goals Taken')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/goals_improvement.png", dpi=300)
        plt.close()
    
    # 5. COMPARE TO PAPER'S REPORTED PERFORMANCE
    print("\nComparing to paper's reported performance...")
    final_results = evaluate_model(model, env, n_episodes=500)
    
    # Paper reports RL efficiency of 94-98% compared to optimal DP solution
    # We can show our achieved reward as an absolute measure
    plt.figure(figsize=(8, 6))
    plt.bar(['Trained Model'], [final_results['mean_reward']], width=0.4)
    plt.title(f'Model Performance\nAverage Reward: {final_results["mean_reward"]:.2f}')
    plt.ylabel('Average Reward')
    plt.ylim(bottom=0)
    plt.text(0, final_results['mean_reward'] * 0.5, 
             f'Goals Taken: {final_results["mean_goals_taken"]:.2f}/{num_goals}',
             ha='center')
    plt.savefig(f"{output_dir}/final_performance.png", dpi=300)
    plt.close()
    
    print(f"\nVerification complete. Plots saved to {output_dir}/")
    return True

if __name__ == "__main__":
    verify_model_replication(
        model_path="./ppo_gbwm_model.zip",
        log_dir="./ppo_gbwm_logs/",
        output_dir="./verification_plots"
    )
