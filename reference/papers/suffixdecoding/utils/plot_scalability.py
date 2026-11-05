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
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.alpha'] = 0.3

# Data from Table: Performance metrics for different entry counts
entries = np.array([1, 5, 10, 15, 20])  # in thousands
tokens = np.array([27, 143, 285, 429, 572])  # in millions
memory = np.array([1670, 2980, 4070, 5110, 6150])  # in MB
update_time = np.array([3.95, 4.06, 4.05, 4.07, 4.04])  # in microseconds
lookup_time = np.array([12.18, 10.93, 12.00, 11.62, 11.64])  # in microseconds

# Create figure with two subplots (adjusted for paper quality)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

# Left subplot: Memory footprint bar plot
ax1.bar(entries, memory, width=0.7, color='#4A90E2', alpha=0.85, 
        label='Measured', edgecolor='#2E5C8A', linewidth=1.2)

# Add linear growth line (fit through the data points)
# Linear fit: memory = slope * entries + intercept
coeffs = np.polyfit(entries, memory, 1)
linear_fit = coeffs[0] * entries + coeffs[1]
ax1.plot(entries, linear_fit, color='#E74C3C', linestyle='--', linewidth=2.5, 
         label='Linear fit $O(n)$', marker='', zorder=5)

ax1.set_xlabel('Number of Entries (thousands)')
ax1.set_ylabel('Memory Footprint (MB)')
ax1.set_title('(a) Memory Scalability', loc='left', fontweight='normal')
ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='black', framealpha=0.95)
ax1.grid(axis='y', linestyle=':', linewidth=0.8)
ax1.set_xticks(entries)
ax1.set_xticklabels([f'{int(e)}' for e in entries])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Right subplot: Update and lookup time curves
ax2.plot(entries, update_time, marker='o', linewidth=2.5, markersize=7, 
         label='Update time', color='#27AE60', markeredgewidth=1.5, 
         markeredgecolor='#1E8449', markerfacecolor='#27AE60')
ax2.plot(entries, lookup_time, marker='s', linewidth=2.5, markersize=7, 
         label='Lookup time', color='#E67E22', markeredgewidth=1.5,
         markeredgecolor='#CA6F1E', markerfacecolor='#E67E22')
ax2.set_xlabel('Number of Entries (thousands)')
ax2.set_ylabel('Time per Token ($\\mu$s)')
ax2.set_title('(b) Update and Lookup Performance', loc='left', fontweight='normal')
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black', framealpha=0.95)
ax2.grid(linestyle=':', linewidth=0.8)
ax2.set_xticks(entries)
ax2.set_xticklabels([f'{int(e)}' for e in entries])
ax2.set_ylim(0, 15)  # Set y-axis from 0 to 15
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('/Users/goliaro/Downloads/_Camera_Ready__SuffixDecoding/latex/figures/scalability_metrics.pdf', 
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/goliaro/Downloads/_Camera_Ready__SuffixDecoding/latex/figures/scalability_metrics.png', 
            dpi=300, bbox_inches='tight')

print("Plot saved as scalability_metrics.pdf and scalability_metrics.png in latex/figures/")

