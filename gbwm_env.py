import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GBWMEnv(gym.Env):
    """
    Custom Environment for Goals-Based Wealth Management (GBWM).
    Follows the Gymnasium API and implements the environment as described in:
    'Reinforcement learning for Multiple Goals in Goals-Based Wealth Management'
    """

    metadata = {'render_modes': [], 'render_fps': 1}

    def __init__(self,
                 num_goals=16,
                 time_horizon=16,
                 initial_wealth_factor=12.0,
                 initial_wealth_exponent=0.85,
                 goal_cost_base=10.0,
                 goal_cost_growth=1.08,
                 goal_utility_base=10.0,
                 goal_utility_increment=1.0,
                 portfolio_means=None,
                 portfolio_stddevs=None,
                 max_steps=None):  # Removed w_max as we'll calculate it dynamically
        super().__init__()
        
        self.num_goals = num_goals
        self.time_horizon = time_horizon
        self.initial_wealth_factor = initial_wealth_factor
        self.initial_wealth_exponent = initial_wealth_exponent
        self.goal_cost_base = goal_cost_base
        self.goal_cost_growth = goal_cost_growth
        self.goal_utility_base = goal_utility_base
        self.goal_utility_increment = goal_utility_increment
        
        # Safety parameter to prevent infinite running
        self.max_steps = max_steps if max_steps is not None else time_horizon + 10
        self.step_count = 0
        
        # Calculate initial wealth
        self.initial_wealth = self.initial_wealth_factor * (self.num_goals ** self.initial_wealth_exponent)
        
        # Define goal times, costs, and utilities
        self.goal_times = np.linspace(self.time_horizon / self.num_goals,
                                       self.time_horizon,
                                       self.num_goals, dtype=int)
        self.goal_costs = {t: self.goal_cost_base * (self.goal_cost_growth ** t) for t in self.goal_times}
        self.goal_utilities = {t: self.goal_utility_base + t * self.goal_utility_increment for t in self.goal_times}
        
        # Define investment portfolios
        if portfolio_means is None or portfolio_stddevs is None:
            self.n_portfolios = 15
            min_mean, max_mean = 0.052632, 0.088636
            min_std, max_std = 0.037351, 0.195437
            
            # The paper specifies equally spaced mean returns
            self.portfolio_means = np.linspace(min_mean, max_mean, self.n_portfolios)
            
            # For stddevs, we use a mapping from the efficient frontier
            # In reality, these would follow a non-linear relationship
            # This is an approximation based on the paper's mentioned range
            self.portfolio_stddevs = np.array([
                0.037351, 0.045000, 0.055000, 0.065000, 0.075000,
                0.085000, 0.095000, 0.110000, 0.125000, 0.140000,
                0.155000, 0.170000, 0.180000, 0.190000, 0.195437
            ])
        else:
            self.n_portfolios = len(portfolio_means)
            self.portfolio_means = np.array(portfolio_means)
            self.portfolio_stddevs = np.array(portfolio_stddevs)
            assert len(portfolio_means) == len(portfolio_stddevs), "Means and stddevs must have same length"
        
        # Calculate w_max dynamically using the paper's approach
        self.w_max = self._compute_w_max()
            
        # Gymnasium Setup
        self.action_space = spaces.MultiDiscrete([2, self.n_portfolios])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # State variables
        self.current_time = 0
        self.current_wealth = self.initial_wealth
        self.total_reward = 0.0

    def _compute_w_max(self):
        """Compute maximum wealth bound based on GBM dynamics as described in the paper."""
        μ_max = np.max(self.portfolio_means)
        σ_max = np.max(self.portfolio_stddevs)
        return self.initial_wealth * np.exp(
            (μ_max - 0.5 * σ_max**2) * self.time_horizon + 3 * σ_max * np.sqrt(self.time_horizon)
        )

    def _get_obs(self):
        """Returns the current observation."""
        norm_time = self.current_time / self.time_horizon
        norm_wealth = np.clip(self.current_wealth / self.w_max, 0.0, 1.0)
        return np.array([norm_time, norm_wealth], dtype=np.float32)

    def _get_info(self):
        """Returns auxiliary information."""
        return {
            "current_time": self.current_time,
            "current_wealth": self.current_wealth,
            "total_reward": self.total_reward,
            "step_count": self.step_count,
            "w_max": self.w_max,
            "action_mask": self._get_action_mask()  # Add action masking
        }
    
    def _get_action_mask(self):
        """Generate a mask for valid actions at the current state."""
        mask = np.ones((2, self.n_portfolios), dtype=np.int8)
        
        # Mask portfolio choices at t=T (not needed)
        if self.current_time == self.time_horizon:
            mask[1, :] = 0
        
        # Mask goal choices in non-goal years
        if self.current_time not in self.goal_times:
            mask[0, :] = 0
            
        # Mask goal-taking if insufficient wealth
        if self.current_time in self.goal_times:
            if self.current_wealth < self.goal_costs[self.current_time]:
                mask[0, 1] = 0  # Mask the "take goal" action
            
        return mask

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        
        self.current_time = 0
        self.current_wealth = self.initial_wealth
        self.total_reward = 0.0
        self.step_count = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Executes one time step within the environment."""
        self.step_count += 1
        goal_choice, portfolio_choice = action
        step_reward = 0.0
        
        # --- Goal Decision Phase ---
        is_goal_time = self.current_time in self.goal_times
        if is_goal_time:
            goal_cost = self.goal_costs[self.current_time]
            goal_utility = self.goal_utilities[self.current_time]
            
            # Special handling for the final goal (t=T)
            if self.current_time == self.time_horizon:
                if self.current_wealth >= goal_cost:
                    self.current_wealth -= goal_cost
                    step_reward = goal_utility
            elif goal_choice == 1:  # Agent chose to take the goal
                if self.current_wealth >= goal_cost:
                    self.current_wealth -= goal_cost
                    step_reward = goal_utility
        
        # --- Investment Phase (only if t < T) ---
        if self.current_time < self.time_horizon:
            mu = self.portfolio_means[portfolio_choice]
            sigma = self.portfolio_stddevs[portfolio_choice]
            
            if self.current_wealth > 1e-6:
                drift = mu - 0.5 * (sigma ** 2)
                diffusion = sigma * self.np_random.standard_normal()
                growth_factor = np.exp(drift + diffusion)
                self.current_wealth *= growth_factor
            else:
                self.current_wealth = 1e-6
        
        # Increment time
        self.current_time += 1
        
        # --- Explicit Termination and Truncation Conditions ---
        # Termination: natural end of episode
        terminated = self.current_time > self.time_horizon
        
        # Truncation: forced end due to safety limit
        truncated = self.step_count >= self.max_steps
        
        # Update total reward
        self.total_reward += step_reward
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Ensure wealth doesn't exceed bounds unreasonably
        self.current_wealth = min(self.current_wealth, self.w_max * 1.5)
        
        return observation, step_reward, terminated, truncated, info

    def close(self):
        """Clean up any resources."""
        pass


# --- Example Usage ---
if __name__ == '__main__':
    env = GBWMEnv(num_goals=4, time_horizon=16)
    obs, info = env.reset()
    
    print("Initial Observation:", obs)
    print("Initial Info:", info)
    print(f"W_max calculated: {info['w_max']:.2f}")
    
    terminated = False
    truncated = False
    total_reward_run = 0
    
    # Run until either terminated or truncated
    while not (terminated or truncated):
        # In PPO implementation, you would use this mask to guide policy
        action_mask = info["action_mask"]
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_run += reward
        
        print(f"Step: {info['step_count']}, Action: {action}, Obs: {obs.round(3)}, "
              f"Reward: {reward:.2f}, Time: {info['current_time']}/{env.time_horizon}, "
              f"Terminated: {terminated}, Truncated: {truncated}")
        
        # Extra safety break
        if info['step_count'] > 100:
            print("WARNING: Breaking due to excessive steps!")
            break
    
    print(f"\nEpisode finished after {info['step_count']} steps.")
    print(f"Final Wealth: {info['current_wealth']:.2f}")
    print(f"Total Accumulated Utility: {total_reward_run:.2f}")
    
    env.close()

# For PPO training, use these hyperparameters from the paper:
"""
# PPO Training Code (pseudo-code)
from stable_baselines3 import PPO

# Hyperparameters from the paper
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.01,           # η = 0.01
    n_steps=16,                   # Time horizon steps
    batch_size=4800,              # M = 4800
    n_epochs=10,                  # Number of epochs per update
    gamma=0.99,                   # Discount factor
    clip_range=0.5,               # ε = 0.50
    policy_kwargs={
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])]  # N_neur = 64
    },
    verbose=1
)

# Train for N_traj = 50,000 trajectories
model.learn(total_timesteps=50000 * 16)
"""
