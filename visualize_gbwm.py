import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from stable_baselines3 import PPO
from tqdm import tqdm
from collections import defaultdict

# Import the custom environment
from gbwm_env import GBWMEnv

class GBWMVisualizer:
    """
    Visualization tool for Goals-Based Wealth Management RL model evaluation.
    Generates plots for key performance indicators (KPIs) from the paper.
    """
    
    def __init__(self, 
                 model_path="./ppo_gbwm_model.zip",
                 num_goals=4,
                 time_horizon=16,
                 initial_wealth_factor=12.0,
                 initial_wealth_exponent=0.85,
                 w_max=1_000_000.0,
                 max_steps=None,
                 output_dir="./visualizations",
                 n_eval_episodes=1000,
                 seed=42):
        """Initialize the visualizer with model and environment parameters."""
        self.model_path = model_path
        self.num_goals = num_goals
        self.time_horizon = time_horizon
        self.initial_wealth_factor = initial_wealth_factor
        self.initial_wealth_exponent = initial_wealth_exponent
        self.w_max = w_max
        self.max_steps = max_steps if max_steps is not None else time_horizon + 5
        self.output_dir = output_dir
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
        
        # Initialize environment and load model
        self.setup_env_and_model()
        
    def setup_env_and_model(self):
        """Set up the environment and load the trained model."""
        print("Setting up environment...")
        self.env = GBWMEnv(
            num_goals=self.num_goals,
            time_horizon=self.time_horizon,
            initial_wealth_factor=self.initial_wealth_factor,
            initial_wealth_exponent=self.initial_wealth_exponent,
            w_max=self.w_max,
            max_steps=self.max_steps
        )
        
        # Load the trained model if it exists
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model = PPO.load(
                self.model_path, 
                env=self.env,
                custom_objects={'action_space': self.env.action_space, 
                               'observation_space': self.env.observation_space}
            )
            print("Model loaded successfully.")
        else:
            print(f"Model file not found at {self.model_path}")
            self.model = None
    
    def evaluate_policy(self, deterministic=True):
        """Evaluate the policy and collect data for visualization."""
        if self.model is None:
            print("No model loaded. Cannot evaluate.")
            return None
        
        print(f"Evaluating policy over {self.n_eval_episodes} episodes...")
        
        # Data structures to store results
        results = {
            'episode_rewards': [],
            'final_wealths': [],
            'episode_lengths': [],
            'goals_taken': [],
            'wealth_trajectories': [],
            'portfolio_choices': [],
            'goal_decisions': []
        }
        
        # Set seed for reproducibility
        self.env.reset(seed=self.seed)
        
        # Run evaluation episodes
        for episode in tqdm(range(self.n_eval_episodes)):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            # Track episode data
            wealth_trajectory = [info['current_wealth']]
            portfolio_choices = []
            goal_decisions = []
            goals_taken = 0
            
            # Run episode
            while not (done or truncated):
                action, _states = self.model.predict(obs, deterministic=deterministic)
                
                # Store action data
                goal_choice, portfolio_choice = action
                goal_decisions.append(goal_choice)
                portfolio_choices.append(portfolio_choice)
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Update tracking
                episode_reward += reward
                wealth_trajectory.append(info['current_wealth'])
                
                # Count taken goals
                if reward > 0:
                    goals_taken += 1
            
            # Store episode results
            results['episode_rewards'].append(episode_reward)
            results['final_wealths'].append(info['current_wealth'])
            results['episode_lengths'].append(info['step_count'])
            results['goals_taken'].append(goals_taken)
            results['wealth_trajectories'].append(wealth_trajectory)
            results['portfolio_choices'].append(portfolio_choices)
            results['goal_decisions'].append(goal_decisions)
        
        # Calculate summary statistics
        results['mean_reward'] = np.mean(results['episode_rewards'])
        results['std_reward'] = np.std(results['episode_rewards'])
        results['median_reward'] = np.median(results['episode_rewards'])
        results['mean_final_wealth'] = np.mean(results['final_wealths'])
        results['std_final_wealth'] = np.std(results['final_wealths'])
        results['median_final_wealth'] = np.median(results['final_wealths'])
        results['mean_goals_taken'] = np.mean(results['goals_taken'])
        
        print("Evaluation complete.")
        return results
    
    def plot_reward_distribution(self, results):
        """Plot the distribution of accumulated rewards."""
        plt.figure(figsize=(10, 6))
        sns.histplot(results['episode_rewards'], kde=True)
        plt.axvline(results['mean_reward'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_reward"]:.2f}')
        plt.axvline(results['median_reward'], color='g', linestyle='--', 
                   label=f'Median: {results["median_reward"]:.2f}')
        
        plt.title('Distribution of Accumulated Utility (Reward)')
        plt.xlabel('Total Utility')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/reward_distribution.png", dpi=300)
        plt.close()
    
    def plot_final_wealth_distribution(self, results):
        """Plot the distribution of final wealth."""
        plt.figure(figsize=(10, 6))
        sns.histplot(results['final_wealths'], kde=True)
        plt.axvline(results['mean_final_wealth'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_final_wealth"]:.2f}')
        plt.axvline(results['median_final_wealth'], color='g', linestyle='--', 
                   label=f'Median: {results["median_final_wealth"]:.2f}')
        
        plt.title('Distribution of Final Wealth')
        plt.xlabel('Final Wealth')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/final_wealth_distribution.png", dpi=300)
        plt.close()
    
    def plot_goals_taken_distribution(self, results):
        """Plot the distribution of goals taken."""
        plt.figure(figsize=(10, 6))
        
        # Count occurrences of each number of goals taken
        goals_counts = pd.Series(results['goals_taken']).value_counts().sort_index()
        
        # Plot as bar chart
        ax = goals_counts.plot(kind='bar')
        plt.title(f'Distribution of Goals Taken (out of {self.num_goals})')
        plt.xlabel('Number of Goals Taken')
        plt.ylabel('Frequency')
        
        # Add percentage labels
        total = len(results['goals_taken'])
        for p in ax.patches:
            percentage = 100 * p.get_height() / total
            ax.annotate(f'{percentage:.1f}%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/goals_taken_distribution.png", dpi=300)
        plt.close()
    
    def plot_average_wealth_trajectory(self, results):
        """Plot the average wealth trajectory over time."""
        # Find max length of wealth trajectories
        max_length = max(len(traj) for traj in results['wealth_trajectories'])
        
        # Pad shorter trajectories with NaN
        padded_trajectories = []
        for traj in results['wealth_trajectories']:
            padded = traj + [np.nan] * (max_length - len(traj))
            padded_trajectories.append(padded)
        
        # Convert to numpy array and calculate mean/std ignoring NaN values
        wealth_data = np.array(padded_trajectories)
        mean_wealth = np.nanmean(wealth_data, axis=0)
        std_wealth = np.nanstd(wealth_data, axis=0)
        
        # Create time axis
        time_steps = np.arange(max_length)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, mean_wealth, label='Mean Wealth', color='blue')
        plt.fill_between(time_steps, 
                         mean_wealth - std_wealth, 
                         mean_wealth + std_wealth, 
                         alpha=0.3, color='blue',
                         label='±1 Std Dev')
        
        # Mark goal times
        goal_times = np.linspace(self.time_horizon / self.num_goals,
                                self.time_horizon,
                                self.num_goals, dtype=int)
        for t in goal_times:
            plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        
        plt.title('Average Wealth Trajectory')
        plt.xlabel('Time Step')
        plt.ylabel('Wealth')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/average_wealth_trajectory.png", dpi=300)
        plt.close()
    
    def plot_portfolio_choice_heatmap(self, results):
        """Plot heatmap of portfolio choices over time."""
        # Initialize counts matrix (time x portfolio choice)
        n_portfolios = self.env.n_portfolios
        portfolio_counts = np.zeros((self.time_horizon, n_portfolios))
        
        # Count portfolio choices at each time step
        for episode in range(len(results['portfolio_choices'])):
            choices = results['portfolio_choices'][episode]
            for t, choice in enumerate(choices):
                if t < self.time_horizon:  # Ensure we're within bounds
                    portfolio_counts[t, choice] += 1
        
        # Convert to percentages
        portfolio_pcts = portfolio_counts / np.sum(portfolio_counts, axis=1, keepdims=True) * 100
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(portfolio_pcts, cmap="YlGnBu", annot=True, fmt=".1f",
                   xticklabels=range(1, n_portfolios+1),
                   yticklabels=range(self.time_horizon))
        
        plt.title('Portfolio Choice Distribution Over Time')
        plt.xlabel('Portfolio (1 = Conservative, 15 = Aggressive)')
        plt.ylabel('Time Step')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/portfolio_choice_heatmap.png", dpi=300)
        plt.close()
    
    def plot_goal_taking_behavior(self, results):
        """Plot goal-taking behavior at goal times."""
        goal_times = np.linspace(self.time_horizon / self.num_goals,
                                self.time_horizon,
                                self.num_goals, dtype=int)
        
        # Initialize data structure to track goal-taking percentages
        goal_decisions = {t: {'take': 0, 'skip': 0} for t in goal_times}
        
        # Count goal decisions at each goal time
        for episode_idx, decisions in enumerate(results['goal_decisions']):
            for t_idx, t in enumerate(goal_times):
                if t_idx < len(decisions):  # Check if we have a decision for this goal
                    if decisions[t_idx] == 1:  # Goal was taken
                        goal_decisions[t]['take'] += 1
                    else:  # Goal was skipped
                        goal_decisions[t]['skip'] += 1
        
        # Calculate percentages
        take_percentages = []
        skip_percentages = []
        
        for t in goal_times:
            total = goal_decisions[t]['take'] + goal_decisions[t]['skip']
            if total > 0:
                take_percentages.append(goal_decisions[t]['take'] / total * 100)
                skip_percentages.append(goal_decisions[t]['skip'] / total * 100)
            else:
                take_percentages.append(0)
                skip_percentages.append(0)
        
        # Plot stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.bar(goal_times, take_percentages, label='Take Goal', color='green')
        ax.bar(goal_times, skip_percentages, bottom=take_percentages, 
               label='Skip Goal', color='red')
        
        ax.set_xticks(goal_times)
        ax.set_xlabel('Goal Time')
        ax.set_ylabel('Percentage')
        ax.set_title('Goal-Taking Behavior at Each Goal Time')
        ax.legend()
        
        # Add percentage labels
        for i, t in enumerate(goal_times):
            if take_percentages[i] > 0:
                ax.text(t, take_percentages[i]/2, f"{take_percentages[i]:.1f}%", 
                       ha='center', va='center', color='white', fontweight='bold')
            
            if skip_percentages[i] > 0:
                ax.text(t, take_percentages[i] + skip_percentages[i]/2, 
                       f"{skip_percentages[i]:.1f}%", 
                       ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/goal_taking_behavior.png", dpi=300)
        plt.close()
    
    def plot_wealth_vs_utility(self, results):
        """Plot the relationship between final wealth and accumulated utility."""
        plt.figure(figsize=(10, 6))
        plt.scatter(results['final_wealths'], results['episode_rewards'], 
                   alpha=0.5, edgecolor='k', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(results['final_wealths'], results['episode_rewards'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(results['final_wealths']), 
                p(sorted(results['final_wealths'])), 
                "r--", linewidth=2)
        
        plt.title('Relationship Between Final Wealth and Accumulated Utility')
        plt.xlabel('Final Wealth')
        plt.ylabel('Accumulated Utility')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/wealth_vs_utility.png", dpi=300)
        plt.close()
    
    def simulate_random_policy(self):
        """Simulate a random policy as a benchmark."""
        print("Simulating random policy...")
        
        # Data structures to store results
        results = {
            'episode_rewards': [],
            'final_wealths': [],
            'episode_lengths': [],
            'goals_taken': [],
            'wealth_trajectories': []
        }
        
        # Set seed for reproducibility
        self.env.reset(seed=self.seed)
        
        # Run evaluation episodes
        for episode in tqdm(range(self.n_eval_episodes)):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            # Track episode data
            wealth_trajectory = [info['current_wealth']]
            goals_taken = 0
            
            # Run episode
            while not (done or truncated):
                # Random action
                action = self.env.action_space.sample()
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Update tracking
                episode_reward += reward
                wealth_trajectory.append(info['current_wealth'])
                
                # Count taken goals
                if reward > 0:
                    goals_taken += 1
            
            # Store episode results
            results['episode_rewards'].append(episode_reward)
            results['final_wealths'].append(info['current_wealth'])
            results['episode_lengths'].append(info['step_count'])
            results['goals_taken'].append(goals_taken)
            results['wealth_trajectories'].append(wealth_trajectory)
        
        # Calculate summary statistics
        results['mean_reward'] = np.mean(results['episode_rewards'])
        results['std_reward'] = np.std(results['episode_rewards'])
        results['median_reward'] = np.median(results['episode_rewards'])
        results['mean_final_wealth'] = np.mean(results['final_wealths'])
        results['std_final_wealth'] = np.std(results['final_wealths'])
        results['median_final_wealth'] = np.median(results['final_wealths'])
        results['mean_goals_taken'] = np.mean(results['goals_taken'])
        
        print("Random policy simulation complete.")
        return results
    
    def simulate_greedy_policy(self):
        """Simulate a greedy policy (always take goals when possible)."""
        print("Simulating greedy policy...")
        
        # Data structures to store results
        results = {
            'episode_rewards': [],
            'final_wealths': [],
            'episode_lengths': [],
            'goals_taken': [],
            'wealth_trajectories': []
        }
        
        # Set seed for reproducibility
        self.env.reset(seed=self.seed)
        
        # Run evaluation episodes
        for episode in tqdm(range(self.n_eval_episodes)):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            # Track episode data
            wealth_trajectory = [info['current_wealth']]
            goals_taken = 0
            
            # Run episode
            while not (done or truncated):
                # Greedy goal-taking: always choose to take goal (1) when at a goal time
                # Otherwise choose a random portfolio
                current_time = info['current_time']
                is_goal_time = current_time in self.env.goal_times
                
                if is_goal_time:
                    action = [1, np.random.randint(0, self.env.n_portfolios)]
                else:
                    action = [0, np.random.randint(0, self.env.n_portfolios)]
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Update tracking
                episode_reward += reward
                wealth_trajectory.append(info['current_wealth'])
                
                # Count taken goals
                if reward > 0:
                    goals_taken += 1
            
            # Store episode results
            results['episode_rewards'].append(episode_reward)
            results['final_wealths'].append(info['current_wealth'])
            results['episode_lengths'].append(info['step_count'])
            results['goals_taken'].append(goals_taken)
            results['wealth_trajectories'].append(wealth_trajectory)
        
        # Calculate summary statistics
        results['mean_reward'] = np.mean(results['episode_rewards'])
        results['std_reward'] = np.std(results['episode_rewards'])
        results['median_reward'] = np.median(results['episode_rewards'])
        results['mean_final_wealth'] = np.mean(results['final_wealths'])
        results['std_final_wealth'] = np.std(results['final_wealths'])
        results['median_final_wealth'] = np.median(results['final_wealths'])
        results['mean_goals_taken'] = np.mean(results['goals_taken'])
        
        print("Greedy policy simulation complete.")
        return results
    
    def simulate_buy_hold_policy(self, portfolio_idx=7):
        """Simulate a buy-and-hold policy (fixed portfolio)."""
        print(f"Simulating buy-and-hold policy with portfolio #{portfolio_idx+1}...")
        
        # Data structures to store results
        results = {
            'episode_rewards': [],
            'final_wealths': [],
            'episode_lengths': [],
            'goals_taken': [],
            'wealth_trajectories': []
        }
        
        # Set seed for reproducibility
        self.env.reset(seed=self.seed)
        
        # Run evaluation episodes
        for episode in tqdm(range(self.n_eval_episodes)):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            # Track episode data
            wealth_trajectory = [info['current_wealth']]
            goals_taken = 0
            
            # Run episode
            while not (done or truncated):
                # Buy-and-hold with random goal-taking
                current_time = info['current_time']
                is_goal_time = current_time in self.env.goal_times
                
                if is_goal_time:
                    # 50% chance to take goal when possible
                    goal_action = np.random.choice([0, 1])
                    action = [goal_action, portfolio_idx]
                else:
                    action = [0, portfolio_idx]  # Always same portfolio
                
                # Step environment
                obs, reward, done, truncated, info = self.env.step(action)
                
                # Update tracking
                episode_reward += reward
                wealth_trajectory.append(info['current_wealth'])
                
                # Count taken goals
                if reward > 0:
                    goals_taken += 1
            
            # Store episode results
            results['episode_rewards'].append(episode_reward)
            results['final_wealths'].append(info['current_wealth'])
            results['episode_lengths'].append(info['step_count'])
            results['goals_taken'].append(goals_taken)
            results['wealth_trajectories'].append(wealth_trajectory)
        
        # Calculate summary statistics
        results['mean_reward'] = np.mean(results['episode_rewards'])
        results['std_reward'] = np.std(results['episode_rewards'])
        results['median_reward'] = np.median(results['episode_rewards'])
        results['mean_final_wealth'] = np.mean(results['final_wealths'])
        results['std_final_wealth'] = np.std(results['final_wealths'])
        results['median_final_wealth'] = np.median(results['final_wealths'])
        results['mean_goals_taken'] = np.mean(results['goals_taken'])
        
        print("Buy-and-hold policy simulation complete.")
        return results
    
    def plot_policy_comparison(self, rl_results, random_results, 
                              greedy_results, buy_hold_results):
        """Plot comparison of different policy performances."""
        plt.figure(figsize=(12, 8))
        
        # Prepare data for comparison
        policies = ["RL (PPO)", "Greedy", "Buy & Hold", "Random"]
        mean_rewards = [
            rl_results['mean_reward'],
            greedy_results['mean_reward'],
            buy_hold_results['mean_reward'],
            random_results['mean_reward']
        ]
        std_rewards = [
            rl_results['std_reward'],
            greedy_results['std_reward'],
            buy_hold_results['std_reward'],
            random_results['std_reward']
        ]
        
        # Create bar plot
        x_pos = np.arange(len(policies))
        plt.bar(x_pos, mean_rewards, yerr=std_rewards, align='center', 
                alpha=0.7, ecolor='black', capsize=10)
        plt.xticks(x_pos, policies)
        
        # Add value labels on top of bars
        for i, v in enumerate(mean_rewards):
            plt.text(i, v + std_rewards[i] + 0.5, f"{v:.2f}", 
                    ha='center', fontweight='bold')
        
        plt.title('Comparison of Policy Performance')
        plt.xlabel('Policy')
        plt.ylabel('Mean Accumulated Utility (Reward)')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/policy_comparison.png", dpi=300)
        plt.close()
    
    def plot_wealth_trajectory_comparison(self, rl_results, random_results, 
                                         greedy_results, buy_hold_results):
        """Plot comparison of wealth trajectories under different policies."""
        plt.figure(figsize=(12, 8))
        
        # Function to calculate average wealth trajectory
        def get_avg_trajectory(results):
            # Find max length
            max_length = max(len(traj) for traj in results['wealth_trajectories'])
            
            # Pad shorter trajectories with NaN
            padded_trajectories = []
            for traj in results['wealth_trajectories']:
                padded = traj + [np.nan] * (max_length - len(traj))
                padded_trajectories.append(padded)
            
            # Convert to numpy array and calculate mean ignoring NaN values
            wealth_data = np.array(padded_trajectories)
            mean_wealth = np.nanmean(wealth_data, axis=0)
            
            return mean_wealth
        
        # Calculate and plot average wealth trajectories
        time_steps = np.arange(self.time_horizon + 2)  # +2 for initial and final
        
        # RL trajectory
        rl_traj = get_avg_trajectory(rl_results)
        plt.plot(time_steps[:len(rl_traj)], rl_traj, 
                'b-', linewidth=2, label='RL (PPO)')
        
        # Other trajectories
        greedy_traj = get_avg_trajectory(greedy_results)
        plt.plot(time_steps[:len(greedy_traj)], greedy_traj, 
                'm-', linewidth=2, label='Greedy')
        
        bh_traj = get_avg_trajectory(buy_hold_results)
        plt.plot(time_steps[:len(bh_traj)], bh_traj, 
                'r-', linewidth=2, label='Buy & Hold')
        
        random_traj = get_avg_trajectory(random_results)
        plt.plot(time_steps[:len(random_traj)], random_traj, 
                'k-', linewidth=2, label='Random')
        
        # Mark goal times
        goal_times = np.linspace(self.time_horizon / self.num_goals,
                                self.time_horizon,
                                self.num_goals, dtype=int)
        for t in goal_times:
            plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
        
        plt.title('Average Wealth Trajectory Comparison')
        plt.xlabel('Time Step')
        plt.ylabel('Wealth')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/wealth_trajectory_comparison.png", dpi=300)
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualizations including benchmark comparisons."""
        if self.model is None:
            print("Cannot generate visualizations: No model loaded.")
            return
        
        # Evaluate RL policy
        print("\n1. Evaluating trained RL policy...")
        rl_results = self.evaluate_policy(deterministic=True)
        if rl_results is None:
            return
        
        # Generate basic plots
        print("\n2. Generating basic RL policy visualizations...")
        self.plot_reward_distribution(rl_results)
        self.plot_final_wealth_distribution(rl_results)
        self.plot_goals_taken_distribution(rl_results)
        self.plot_average_wealth_trajectory(rl_results)
        self.plot_portfolio_choice_heatmap(rl_results)
        self.plot_goal_taking_behavior(rl_results)
        self.plot_wealth_vs_utility(rl_results)
        
        # Simulate benchmark policies
        print("\n3. Simulating benchmark policies...")
        random_results = self.simulate_random_policy()
        greedy_results = self.simulate_greedy_policy()
        buy_hold_results = self.simulate_buy_hold_policy(portfolio_idx=7)
        
        # Generate comparison visualizations
        print("\n4. Generating policy comparison visualizations...")
        self.plot_policy_comparison(
            rl_results, random_results, greedy_results, buy_hold_results
        )
        
        self.plot_wealth_trajectory_comparison(
            rl_results, random_results, greedy_results, buy_hold_results
        )
        
        # Save summary statistics to file
        summary = {
            'Mean Reward': rl_results['mean_reward'],
            'Std Reward': rl_results['std_reward'],
            'Median Reward': rl_results['median_reward'],
            'Mean Final Wealth': rl_results['mean_final_wealth'],
            'Std Final Wealth': rl_results['std_final_wealth'],
            'Median Final Wealth': rl_results['median_final_wealth'],
            'Mean Goals Taken': rl_results['mean_goals_taken'],
            'Number of Episodes': self.n_eval_episodes
        }
        
        with open(f"{self.output_dir}/summary_statistics.txt", 'w') as f:
            f.write("Summary Statistics\n")
            f.write("=================\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"\nAll visualizations saved to {self.output_dir}/")


# Example usage
if __name__ == "__main__":
    visualizer = GBWMVisualizer(
        model_path="./ppo_gbwm_model.zip",
        num_goals=4,
        time_horizon=16,
        n_eval_episodes=1000,
        output_dir="./visualizations"
    )
    
    visualizer.generate_all_visualizations()
