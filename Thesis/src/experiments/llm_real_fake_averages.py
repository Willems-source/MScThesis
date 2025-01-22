# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import ttest_ind
# from itertools import combinations

# # Load the dataset
# file_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis/llm_real_fake_metrics_general_50_topic_model.csv"
# data = pd.read_csv(file_path)

# # Specify the columns that represent metrics
# metric_columns = ["Mean JS", "Std JS", "First Order Diff JS", "Peak Ratio", "RMSE"]

# # Calculate the average values for specified metrics grouped by the label column
# averages = data.groupby("label")[metric_columns].mean()

# # Manually set the correct category order: humanreal, humanfake, llm
# categories = ["humanreal", "humanfake", "llm"]

# # Reindex the averages DataFrame to enforce the correct order of categories
# averages = averages.reindex(categories)

# # Debugging: Print the averages to ensure the correct order
# print("Corrected Grouped Averages for Metrics by Label:")
# print(averages)

# # Statistical significance testing within each metric
# significance_results = {}
# for metric in metric_columns:
#     pairs = list(combinations(categories, 2))  # All pairwise combinations
#     significance_results[metric] = {}
#     for cat1, cat2 in pairs:
#         # Perform a t-test for each pair
#         t_stat, p_value = ttest_ind(
#             data[data["label"] == cat1][metric],
#             data[data["label"] == cat2][metric],
#             equal_var=False  # Welch's t-test for unequal variances
#         )
#         significance_results[metric][f"{cat1} vs {cat2}"] = p_value

# # Create the bar plots for each metric
# fig, axes = plt.subplots(1, len(metric_columns), figsize=(20, 5), sharey=False)
# fig.suptitle("Average Values of Metrics for Human Real, Human Fake, and LLM (Corrected)")

# for i, metric in enumerate(metric_columns):
#     ax = axes[i]
#     sns.barplot(x=categories, y=averages[metric], ax=ax, palette="Set1")
#     ax.set_title(metric)
#     ax.set_xlabel("Category")
#     ax.set_ylabel("Average Value")

# # Adjust layout and save the plot
# output_path = "C:/Thesis/MScThesis/gc_experiments_results/plots/average_metrics_bar_chart_corrected.png"
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
# plt.savefig(output_path, dpi=300)
# plt.show()

# print(f"Bar chart saved to: {output_path}")

# # Print statistical significance results
# print("\nStatistical Significance Results (Pairwise t-tests):")
# for metric, results in significance_results.items():
#     print(f"\n{metric}:")
#     for pair, p_value in results.items():
#         print(f"  {pair}: p = {p_value:.4f}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from itertools import combinations
import math
from matplotlib.ticker import FormatStrFormatter

# Load the dataset
file_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis/llm_real_fake_metrics_general_50_topic_model.csv"
data = pd.read_csv(file_path)

# Specify the columns that represent metrics
metric_columns = ["Mean JS", "Std JS", "First Order Diff JS", "Peak Ratio", "RMSE"]

# Rename the metrics for the updated display
renamed_metrics = {
    "Mean JS": "Mean",
    "Std JS": "Standard Deviation",
    "First Order Diff JS": "Oscillation",
    "Peak Ratio": "Peak Ratio",
    "RMSE": "RMSE"
}

# Calculate the average values for specified metrics grouped by the label column
averages = data.groupby("label")[metric_columns].mean()

# Manually set the correct category order: humanreal, humanfake, llm
categories = ["humanreal", "humanfake", "llm"]

# Reindex the averages DataFrame to enforce the correct order of categories
averages = averages.reindex(categories)

# Statistical significance testing
significance_results = {}
all_pairs = list(combinations(categories, 2))  # (humanreal-humanfake, humanreal-llm, humanfake-llm)
for metric in metric_columns:
    significance_results[metric] = {}
    for cat1, cat2 in all_pairs:
        # Perform a t-test for each pair (Welch's t-test)
        x = data[data["label"] == cat1][metric]
        y = data[data["label"] == cat2][metric]
        
        # Handle exceptions for NaN or small sample sizes
        try:
            _, p_value = ttest_ind(x.dropna(), y.dropna(), equal_var=False)
        except Exception:
            p_value = float('nan')
        
        # Store the p-value
        significance_results[metric][f"{cat1} vs {cat2}"] = p_value

# Print the average metric values
print("\nAverage Metric Values:")
print(averages.rename(columns=renamed_metrics).round(3))

# Print statistical significance results
print("\nStatistical Significance Results (Pairwise t-tests):")
for metric, results in significance_results.items():
    print(f"\n{renamed_metrics[metric]}:")
    for pair, p_value in results.items():
        if not math.isnan(p_value):
            print(f"  {pair}: p = {p_value:.4f}")
        else:
            print(f"  {pair}: p = NaN")

# Helper function to annotate significance
def significance_annotation(ax, x1, x2, y, p_value, fontsize=10):
    """
    Draw a line between x1 and x2 at height y, labeling with 
    significance level determined by p_value. If p_value is NaN,
    label is 'ns'.
    """
    # Determine the label based on the p-value
    if math.isnan(p_value):
        label = "ns"
    elif p_value < 0.01:
        label = "***"
    elif p_value < 0.05:
        label = "**"
    elif p_value < 0.1:
        label = "*"
    else:
        label = "ns"
    
    # Arc parameters
    h = 0.005
    # Draw the connecting arcs/lines
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c='k')
    # Place the text in the middle
    ax.text((x1 + x2) * 0.5, y + h, label, ha='center', va='bottom', color='k', fontsize=fontsize)

# === Create the bar plots in a two-row layout ===
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=False)
plt.subplots_adjust(wspace=0.5, hspace=0.4)  # Increase space between plots

# Increase the font size of the overall title
#fig.suptitle("Average Values of Metrics for Human Real, Human Fake, and LLM", fontsize=22)

for i, metric in enumerate(metric_columns):
    row = i // 3
    col = i % 3
    ax = axes[row, col] if len(metric_columns) > 3 else axes[i]  # Handle row layout

    sns.barplot(
        x=categories,
        y=averages[metric],
        ax=ax,
        palette="Set1",
        edgecolor="black",
        linewidth=1.2,
        dodge=False,
        width=0.6
    )

    # Add values tightly above the bars
    for j, val in enumerate(averages[metric]):
        ax.text(j, val + 0.005, f"{val:.2f}", ha='center', va='bottom', fontsize=12)  # Adjusted closer to the bar

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(renamed_metrics[metric], fontsize=18)
    ax.set_xlabel("")
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=14)
    ax.set_ylabel("", fontsize=16)

    # Add the significance arcs higher than before
    y_max = averages[metric].max()
    offset_step = 0.12 * (y_max if y_max != 0 else 1.0)  # Increase the offset step for more space
    for idx, (cat1, cat2) in enumerate(all_pairs):
        x1, x2 = categories.index(cat1), categories.index(cat2)
        p_value = significance_results[metric][f"{cat1} vs {cat2}"]
        offset = 0.02 + idx * offset_step  # Start higher and increase step size
        significance_annotation(ax, x1, x2, y_max + offset, p_value, fontsize=12)

# Hide unused subplots if any
if len(metric_columns) < len(axes.flatten()):
    for j in range(len(metric_columns), len(axes.flatten())):
        axes.flatten()[j].set_visible(False)

# Save and show the plot
output_path = "C:/Thesis/MScThesis/gc_experiments_results/plots/improved_metrics_bar_chart.png"
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Bar chart saved to: {output_path}")


from numpy import mean, std

# Function to calculate Cohen's d
def compute_cohens_d(group1, group2):
    """Calculate Cohen's d effect size for two groups."""
    try:
        if len(group1) == 0 or len(group2) == 0:  # Check for empty groups
            return float('nan')
        mean_diff = mean(group1) - mean(group2)
        pooled_std = std(group1, ddof=1)**2 + std(group2, ddof=1)**2
        pooled_std = (pooled_std / 2)**0.5
        return mean_diff / pooled_std if pooled_std > 0 else 0
    except Exception as e:
        print(f"[WARN] Error calculating Cohen's d: {e}")
        return float('nan')

# Add Cohen's d effect size computation to significance testing
effect_sizes = {}
for metric in metric_columns:
    effect_sizes[metric] = {}
    for cat1, cat2 in all_pairs:
        x = data[data["label"] == cat1][metric]
        y = data[data["label"] == cat2][metric]
        # Compute Cohen's d and handle potential issues with NaN or empty groups
        try:
            cohens_d = compute_cohens_d(x.dropna(), y.dropna())
        except Exception as e:
            print(f"[WARN] Error processing {metric} for {cat1} vs {cat2}: {e}")
            cohens_d = float('nan')
        effect_sizes[metric][f"{cat1} vs {cat2}"] = cohens_d

# Print statistical significance and Cohen's d results
print("\nStatistical Significance Results (Pairwise t-tests and Cohen's d):")
for metric, results in significance_results.items():
    print(f"\n{renamed_metrics[metric]}:")
    for pair, p_value in results.items():
        d_value = effect_sizes[metric][pair]
        if not math.isnan(p_value):
            print(f"  {pair}: p = {p_value:.4f}, Cohen's d = {d_value:.4f}")
        else:
            print(f"  {pair}: p = NaN, Cohen's d = NaN")




