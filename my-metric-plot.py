import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)


seg_metrics = {
    "Curve continuity": [1,0,1,0.5,0.5,0.5,0.5,0.5,1,1,0.5,1,1,0.5,1,1,0,0,1,1,1,0.5,1,1,1,1,0.5,0.5,1,0.8,0,1,0.5,0.5,0.5,0.2,0.8,1,0.2,0.2,0.2,0,0.8,0.8],
    "False positive rate": [1,0,0,0,0,0,0,0,0.5,1,0.5,1,0.5,1,0.5,1,0,0,1,0.8,0.5,1,0.5,1,0.9,0.5,1,1,0,0.5,0,0.5,0.5,0.5,0.5,0.2,0.4,0.2,0.2,0.3,0.5,0.5,0.5,0.6],
    "Header interferece": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
}
seg2_metrics = {
    "Curve continuity": [1,0,0.8,0.8,0.8,0.8,0.7,0.7,0.8,0.8,1,0.8,0.8,0.7,1,1,0.8,0.5,0.8,1,1,1,1,0.8,1,1,0.5,0.8,0.8,0.5,0,0.5,0,0.5,0.5,0,0.5,1,0,0,0.5,0,0.5,1],
    "False positive rate": [1,0.5,0.8,0.5,0.8,0.8,0.5,0.8,1,1,1,1,1,0.5,0.8,0.8,0.5,0.5,1,0.5,0.5,1,0.5,1,1,0.5,1,0.9,1,0,0,1,0.5,0.5,0.5,0,0,0.5,0.5,0.5,0,1,0.5,0],
    "Header interferece": [1,0.5,1,0.7,1,1,1,0.7,1,1,1,1,1,1,1,1,1,1,1,1,0.8,1,1,0.8,1,0.8,1,1,1,1,0.5,0,0.5,0.5,0.5,1,1,0.5,0.5,0,0,0,1,0]
}
noheader_andheader_metrics = {
    "Curve continuity": [0.5,0,0.5,0.5,0.5,0.5,0.5,1,0.7,1,1,1,1,1,1,1,0.5,0.5,1,1,1,1,1,1,0.9,1,0.8,0.9,1,1,0.5,1,0.7,0.8,0.5,0,1,1,0.5,0.5,0.4,0.2,1,1],
    "False positive rate": [1,0,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,0.8,1,0.9,1,1,0.5,0.5,1,1,0.5,1,0.6,1,1,0.5,1,1,1,0.8,0.5,1,1,0.5,0.5,0.5,0.8,0.8,0.7,0.5,0.6,0.5,0.5,0.8],
    "Header interferece": [1,0,1,0.2,0.5,0.5,1,0.8,1,1,0.5,0.8,1,0.5,0.5,1,0.8,0.9,1,1,0.5,0.5,0.8,1,1,0.8,1,1,0.7,1,0.5,0.5,0.8,0.5,0.5,0.5,1,0.6,0.8,0.5,0.5,0.8,0.2,0.5]
}
thisnew_lascurves_metrics = {
    "Curve continuity": [1,0.5,0,1,0.5,0.5,0.5,0.8,0.5,1,1,0.5,1,0.6,0.4,0.5,0.6,1,1,0.6,0.5,0.5,1,0.5,0.2,0.8,1,0.8,0,1,1,1,0.5,0.2,1,1,0,0,0,0.5,1,1],
    "False positive rate": [1,0.5,0.8,0.5,0.5,0.2,0,0.8,1,0.9,0.7,1,0.5,0.5,0.8,0.5,0.5,0.9,0.5,0,0.8,0.5,1,1,0.5,1,1,0.8,0.5,0.5,1,0,0.5,0,0.5,1,0.8,0.8,0.8,0.6,1,1,0.6],
    "Header interferece": [1,0.5,1,0.5,0.9,1,0.5,0.8,1,1,1,1,0.8,1,0.5,1,0.5,0.8,1,1,0.5,0.8,0.8,0.9,1,0.8,1,1,1,1,0.5,0,0.5,0.5,0.5,0.8,0.7,0.8,0.8,0.6,0.6,0.5,0.9,0]
}

seg_metrics_means = [np.mean(val) for val in seg_metrics.values()]
seg_metrics_overall_mean = np.mean(seg_metrics_means)

seg2_metrics_means = [np.mean(val) for val in seg2_metrics.values()]
seg2_metrics_overall_mean = np.mean(seg2_metrics_means)

noheader_metric_means = [np.mean(val) for val in noheader_andheader_metrics.values()]
noheader_metric_overall_mean = np.mean(noheader_metric_means)

thisnew_lascurves_metric_means = [np.mean(val) for val in thisnew_lascurves_metrics.values()]
thisnew_lascurves_metric_overall_mean = np.mean(thisnew_lascurves_metric_means)

# for key, values in seg_metrics.items():
#     plt.figure()
#     sns.set_style("white")
#     sns.histplot(values, kde=True, bins=len(set(values)))
#     plt.title(f"{key} Distribution")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.savefig(os.path.join("plots", f"seg_metrics_{key}_distribution.png"))
#     plt.close()

# for key, values in seg2_metrics.items():
#     plt.figure()
#     sns.set_style("white")
#     sns.histplot(values, kde=True, bins=len(set(values)))
#     plt.title(f"{key} Distribution")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.savefig(os.path.join("plots", f"seg2_metrics_{key}_distribution.png"))
#     plt.close()

# for key, values in noheader_andheader_metrics.items():
#     plt.figure()
#     sns.set_style("white")
#     sns.histplot(values, kde=True, bins=len(set(values)))
#     plt.title(f"{key} Distribution")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.savefig(os.path.join("plots", f"noheader_andheader_metrics_{key}_distribution.png"))
#     plt.close()

# for key, values in thisnew_lascurves_metrics.items():
#     plt.figure()
#     sns.set_style("white")
#     sns.histplot(values, kde=True, bins=len(set(values)))
#     plt.title(f"{key} Distribution")
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.savefig(os.path.join("plots", f"thisnew_lascurves_metrics_{key}_distribution.png"))
#     plt.close()

grand_means = [
   {"Experiment": "no annotated header", "Grand mean": seg_metrics_overall_mean},
   {"Experiment": "annotated header", "Grand mean": seg2_metrics_overall_mean},
   {"Experiment": "no header + header", "Grand mean": noheader_metric_overall_mean},
   {"Experiment": "header + las value generated", "Grand mean": thisnew_lascurves_metric_overall_mean},
]

data = []
for row in grand_means:
    data.append({"Experiment": row["Experiment"], "value": row["Grand mean"], "type": "actual"}),
    data.append({"Experiment": row["Experiment"], "value": 1.0, "type": "ideal"})


df_grand_means = pd.DataFrame(data)

sns.set_style("white")
sns.set_palette("magma")
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=df_grand_means, x="Experiment", y="value", hue="type", ax=ax)

# Get the bars after creating the plot
bars = ax.patches
n_experiments = len(grand_means)

# The bars are ordered: all "actual" bars first, then all "ideal" bars
actual_bars = bars[:n_experiments]
ideal_bars = bars[n_experiments:]

# Draw lines between actual and ideal bars
for i in range(n_experiments):
    # Get the center x-position of each bar
    actual_x = actual_bars[i].get_x() + actual_bars[i].get_width() / 2
    actual_y = actual_bars[i].get_height()
    ideal_x = ideal_bars[i].get_x() + ideal_bars[i].get_width() / 2
    ideal_y = ideal_bars[i].get_height()

    # Draw the connecting line
    ax.plot([actual_x, ideal_x], [actual_y, ideal_y],
            color='blue', linestyle='--', linewidth=2, alpha=0.7)

    # Add gap text
    gap = ideal_y - actual_y
    mid_x = (actual_x + ideal_x) / 2
    mid_y = (actual_y + ideal_y) / 2
    ax.text(mid_x, mid_y, f'{gap:.2f}',
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8))

# Add ideal threshold line
#ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Ideal threshold')

# Customize the plot
plt.title("Grand Mean Comparison vs. Ideal", fontsize=14, fontweight='bold')
plt.xlabel("Experiment", fontsize=12)
plt.ylabel("Grand Mean", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Type', loc='best')
plt.tight_layout()
plt.show()
