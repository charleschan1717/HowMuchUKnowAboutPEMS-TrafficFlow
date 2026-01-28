import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
import os

# Configuration
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'figure.autolayout': True,
    'grid.alpha': 0.3
})

def load_data(path):
    if not os.path.exists(path):
        print("Data not found, generating random data for testing.")
        return np.random.rand(17856, 170, 3) * 100
    return np.load(path)['data']

# Analysis 1: Global Distribution & Normality Check (Figure 1)
def plot_global_distribution(flow_data):
    # Flatten data to treat as a single population
    flat_data = flow_data.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Histogram & KDE
    sns.histplot(flat_data, bins=100, kde=True, stat="density", ax=ax1, 
                 color='#3498db', edgecolor='black', alpha=0.5)
    ax1.set_title('Global Flow Distribution (Histogram)', fontweight='bold')
    ax1.set_xlabel('Traffic Flow (Veh/5min)')
    ax1.set_ylabel('Density')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Subplot 2: Q-Q Plot
    stats.probplot(flat_data, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_markerfacecolor('#0000FF') # Data dots
    ax2.get_lines()[0].set_markeredgecolor('#0000FF')
    ax2.get_lines()[1].set_color('#FF0000')           # Theoretical line
    ax2.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_distribution.png', dpi=300)
    plt.show()

# Analysis 2: Spatio-Temporal Availability Matrix (Figure 2)
def plot_high_contrast_matrix(flow_data):
    num_steps, num_nodes = flow_data.shape
    steps_per_day = 288
    
    viz_matrix = np.zeros_like(flow_data)
    is_zero = (flow_data == 0)
    
    time_indices = np.arange(num_steps)
    hour_of_day = (time_indices % steps_per_day) * 5 / 60
    is_daytime = (hour_of_day >= 6) & (hour_of_day < 22)
    
    viz_matrix[is_zero] = 1 
    day_anomaly_mask = is_zero & is_daytime[:, None]
    viz_matrix[day_anomaly_mask] = 2
    
    fig, ax = plt.subplots(figsize=(24, 6)) 
    
    colors = ['#E0E0E0', '#2980B9', '#FF0000'] 
    cmap = ListedColormap(colors)
    
    ax.imshow(viz_matrix.T, cmap=cmap, aspect='auto', interpolation='nearest')
    
    ax.set_title('Figure 2: Spatio-Temporal Availability Matrix (Sensor Failures Highlighted)', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Sensor Node Index', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (Weeks)', fontsize=14, fontweight='bold')
    
    xticks = np.arange(0, num_steps, steps_per_day * 7)
    xticklabels = [f'Week {i//(steps_per_day*7) + 1}' for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    legend_patches = [
        mpatches.Patch(color=colors[0], label='Valid Flow'),
        mpatches.Patch(color=colors[1], label='Night Zero (Normal)'),
        mpatches.Patch(color=colors[2], label='Daytime Missing (Anomaly)')
    ]
    
    ax.legend(handles=legend_patches, loc='upper right', 
              fontsize=12, frameon=True, framealpha=1, edgecolor='black', facecolor='white')
    ax.grid(which='major', axis='x', color='black', alpha=0.1, linestyle='--')

    plt.tight_layout()
    plt.savefig('01_missing_matrix_high_contrast.png', dpi=300, bbox_inches='tight')
    plt.show()

# Execution
if __name__ == "__main__":
    NPZ_PATH = r'data/PEMS08.npz'
    data = load_data(NPZ_PATH)
    flow_data = data[:, :, 0]
    
    # 1. Statistics & Normality
    print(">>> Generating Figure 1 (Distribution)...")
    plot_global_distribution(flow_data)
    
    # 2. Dead Node Analysis
    node_zero_rates = np.sum(flow_data == 0, axis=0) / flow_data.shape[0]
    dead_nodes = np.where(node_zero_rates > 0.99)[0]
    print(f"Dead Nodes (Always 0): {dead_nodes}")
    
    # 3. Spatio-Temporal Matrix
    print("Generating Figure 2 (Availability Matrix)...")
    plot_high_contrast_matrix(flow_data)
