# Goals-Based Wealth Management using Reinforcement Learning

**Author**: Sanket Mishra, Roll Number 194190004,
**Course Instructor**: Prof. K. S. Mallikarjuna Rao, IEOR

## Introduction

This repository contains the implementation of "Reinforcement Learning for Multiple Goals in Goals-Based Wealth Management" based on the paper by Das et al. The project demonstrates how reinforcement learning can be applied to solve the goals-based wealth management (GBWM) problem, where investors seek to maximize the expected utility from multiple financial goals over time while making optimal investment decisions.

GBWM is an approach that focuses on helping investors achieve specific financial goals rather than just maximizing returns or meeting benchmarks. The implementation uses Proximal Policy Optimization (PPO) to learn optimal strategies for:

1. When to fulfill financial goals (goal-taking decisions)
2. How to allocate investments among different portfolios (portfolio selection)

## Repository Structure

The repository consists of four main Python files that should be executed in the following order:

1. **gbwm_env.py**: Custom Gymnasium environment for GBWM that models:
    - State space: Time and wealth
    - Action space: Goal-taking decisions and portfolio selection
    - Reward function: Utilities from fulfilled goals
    - Investment returns: Portfolio evolution using geometric Brownian motion
2. **train_ppo.py**: Trains the PPO reinforcement learning agent:
    - Parallel environment training for efficiency
    - Progress tracking and saving checkpoints
    - Hyperparameter configuration based on the paper
3. **evaluate_policy.py**: Evaluates the trained model:
    - Runs multiple episodes with the trained policy
    - Collects statistics on rewards and final wealth
    - Reports performance metrics
4. **visualize_gbwm.py**: Generates visualizations for key performance indicators:
    - Reward and wealth distributions
    - Wealth trajectories over time
    - Portfolio selection patterns
    - Goal-taking behavior
    - Comparison with benchmark strategies (random, greedy, buy-and-hold)

## Installation

```bash
# Clone the repository
git clone https://github.com/OResearcher/IE612 _FinEnggProject.git
cd IE612 _FinEnggProject

# Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```


## Usage

Execute the files in the following order:

### 1. Environment Testing

To verify that the environment works correctly:

```bash
python gbwm_env.py
```

This will run a single episode with random actions to ensure the environment functions as expected.

### 2. Training the Agent

```bash
python train_ppo.py
```

This will train the PPO agent for 50,000 trajectories (approximately 850,000 total timesteps). Training progress will be displayed, and checkpoints will be saved in the `ppo_gbwm_logs/checkpoints` directory. The final model will be saved as `ppo_gbwm_model.zip`.

### 3. Evaluating Performance

```bash
python evaluate_policy.py
```

This script evaluates the trained model over 1,000 episodes and reports statistics on accumulated utility (rewards) and final wealth.

### 4. Visualizing Results

```bash
python visualize_gbwm.py
```

This generates a comprehensive set of visualizations saved to the `visualizations` directory, including:

- Reward and wealth distributions
- Goal-taking behavior
- Portfolio selection heatmaps
- Performance comparison with benchmark strategies


## Key Findings

The implementation demonstrates that reinforcement learning can effectively solve the goals-based wealth management problem. As shown in the paper, the PPO algorithm achieves:

- 94-98% of the optimal expected utility compared to dynamic programming solutions
- Superior performance compared to benchmark strategies like greedy goal-taking and buy-and-hold
- Effective adaptation to different numbers of goals and time horizons

The visualizations help understand how the trained agent balances between current and future goals, and how it adjusts its investment strategy over time based on accumulated wealth.

## Further Research

Potential areas for extending this work include:

- Adding more state variables (inflation, interest rates)
- Incorporating investor risk preferences
- Considering tax-efficient investment strategies
- Testing with different market models beyond geometric Brownian motion


## Acknowledgments

This project was completed as part of the IE 612 Introduction to Financial Engineering course. The implementation is based on the paper ["Reinforcement Learning for Multiple Goals in Goals-Based Wealth Management" by Das, Mittal, Ostrov, Radhakrishnan, Srivastav, and Wang.](https://srdas.github.io/Papers/GBWM_RL_AIxB.pdf)

<div style="text-align: center">⁂</div>

