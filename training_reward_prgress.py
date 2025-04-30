import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from stable_baselines3.common.monitor import load_results

# # Set the visual style
# sns.set_style("whitegrid")
# plt.figure(figsize=(12, 7))

# Load training data
training_data = load_results("./ppo_gbwm_logs/")
timesteps = np.array(np.cumsum(training_data['l']), dtype=float)  # Convert to float explicitly
rewards = np.array(training_data['r'], dtype=float)  # Convert to float explicitly

# 1. Plot raw data with low opacity
plt.scatter(timesteps, rewards, alpha=0.1, s=1, color='#ff7f0e', label='Individual Episodes')

# 2. Add multiple smoothing levels for better trend visibility
window_sizes = [10, 50, 100]
colors = ['red', 'green', 'blue']
labels = ['10-Episode Average', '50-Episode Average', '100-Episode Average']

for i, window in enumerate(window_sizes):
    smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    plt.plot(timesteps, smoothed, linewidth=2+i, color=colors[i], label=labels[i])

# 3. Add trend line to show overall progress (fixed with explicit dtype conversion)
z = np.polyfit(timesteps, rewards, 1)
p = np.poly1d(z)
plt.plot(timesteps, p(timesteps), '--', color='purple', linewidth=2, 
         label=f'Trend (slope: {z[0]:.6f})')

# # 4. Highlight convergence region (last 20% of training)
# convergence_start = int(len(timesteps) * 0.8)
# avg_final = np.mean(rewards[convergence_start:])
# plt.axhline(y=avg_final, linestyle=':', color='black', alpha=0.7,
#            label=f'Final Avg: {avg_final:.2f}')

# 5. Improve readability and aesthetics
plt.title('GBWM Training Reward Progress', fontsize=16)
plt.xlabel('Timesteps', fontsize=14)
plt.ylabel('Rewards', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
# plt.grid(True, alpha=0.3)
plt.tight_layout()

# # 6. Add annotation for total training steps
# plt.annotate(f'Total Steps: {timesteps[-1]:,.0f}', 
#              xy=(0.02, 0.02), xycoords='axes fraction',
#              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Save high-resolution figure
plt.savefig('improved_training_progress.png', dpi=300, bbox_inches='tight')
