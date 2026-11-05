import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# Data from Table: Hybrid SuffixDecoding speedup on Spec-Bench
specbench_thresholds = np.array([0, 1, 2, 3, 4, 5, 6, 7])
specbench_data = {
    'coding': [1.83, 2.28, 2.73, 2.88, 3.07, 3.25, 3.19, 3.44],
    'qa': [1.77, 1.86, 2.04, 2.09, 2.22, 2.28, 2.22, 2.40],
    'math': [1.97, 2.01, 2.40, 2.54, 2.72, 2.70, 2.88, 3.14],
    'reasoning': [1.39, 1.69, 2.01, 2.10, 2.25, 2.35, 2.30, 2.50],
    'stem': [1.39, 1.90, 2.19, 2.39, 2.57, 2.74, 2.68, 2.87],
    'writing': [1.33, 2.12, 2.36, 2.50, 2.64, 2.70, 2.61, 2.83],
    'Overall': [1.66, 1.80, 2.03, 2.13, 2.25, 2.37, 2.31, 2.50]
}

# Data from Table: Hybrid SuffixDecoding speedup on AgenticSQL
agenticsql_thresholds = np.array([0, 1, 2, 3, 7])
agenticsql_data = {
    'Classify': [3.02, 2.31, 2.24, 2.22, 2.06],
    'Extract': [9.85, 7.03, 7.14, 5.51, 6.79],
    'Enrich': [10.41, 8.37, 7.97, 5.68, 7.10],
    'Combine': [4.85, 3.38, 3.33, 3.52, 3.44],
    'SQL': [3.09, 2.17, 2.29, 2.20, 2.29],
    'Overall': [5.35, 3.95, 3.80, 3.22, 3.66]
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# Color palette for different tasks
colors_specbench = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
colors_agenticsql = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Left subplot: Spec-Bench results
tasks_to_plot_specbench = ['coding', 'math', 'reasoning', 'stem', 'writing', 'qa']
for i, task in enumerate(tasks_to_plot_specbench):
    ax1.plot(specbench_thresholds, specbench_data[task], 
             marker='o', linewidth=2, markersize=5.5, label=task.capitalize(),
             color=colors_specbench[i], markeredgewidth=1, markeredgecolor='white')

# Plot overall with thicker line
ax1.plot(specbench_thresholds, specbench_data['Overall'], 
         marker='D', linewidth=3, markersize=6.5, label='Overall',
         color='#2C3E50', markeredgewidth=1.5, markeredgecolor='white', zorder=10)

ax1.set_xlabel('Threshold $\\tau$')
ax1.set_ylabel('Speedup ($\\times$)')
ax1.set_title('(a) Spec-Bench (Open-Ended)', loc='left', fontweight='normal')
ax1.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', 
           framealpha=0.95, ncol=2)
ax1.grid(linestyle=':', linewidth=0.8, alpha=0.4)
ax1.set_xticks(specbench_thresholds)
ax1.set_ylim(1.0, 3.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add annotation for optimal range
ax1.axvspan(5, 7, alpha=0.15, color='green', zorder=0)
ax1.text(6, 3.5, 'Optimal\nrange', ha='center', va='center', 
         fontsize=8.5, color='darkgreen', style='italic')

# Right subplot: AgenticSQL results
tasks_to_plot_agenticsql = ['Extract', 'Enrich', 'Combine', 'Classify', 'SQL']
for i, task in enumerate(tasks_to_plot_agenticsql):
    ax2.plot(agenticsql_thresholds, agenticsql_data[task], 
             marker='s', linewidth=2, markersize=5.5, label=task,
             color=colors_agenticsql[i], markeredgewidth=1, markeredgecolor='white')

# Plot overall with thicker line
ax2.plot(agenticsql_thresholds, agenticsql_data['Overall'], 
         marker='D', linewidth=3, markersize=6.5, label='Overall',
         color='#2C3E50', markeredgewidth=1.5, markeredgecolor='white', zorder=10)

ax2.set_xlabel('Threshold $\\tau$')
ax2.set_ylabel('Speedup ($\\times$)')
ax2.set_title('(b) AgenticSQL (Agentic)', loc='left', fontweight='normal')
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', 
           framealpha=0.95, ncol=2)
ax2.grid(linestyle=':', linewidth=0.8, alpha=0.4)
ax2.set_xticks(agenticsql_thresholds)
ax2.set_ylim(1.0, 11.0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add annotation for optimal value
ax2.axvspan(-0.5, 0.5, alpha=0.15, color='green', zorder=0)
ax2.text(0, 10.2, 'Best: $\\tau=0$', ha='center', va='center', 
         fontsize=8.5, color='darkgreen', style='italic')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('/Users/goliaro/Downloads/_Camera_Ready__SuffixDecoding/latex/figures/threshold_sensitivity.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/goliaro/Downloads/_Camera_Ready__SuffixDecoding/latex/figures/threshold_sensitivity.png', 
            dpi=300, bbox_inches='tight')

print("Plot saved as threshold_sensitivity.pdf and threshold_sensitivity.png in latex/figures/")

