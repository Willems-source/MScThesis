# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define the directory for saving plots
# output_base_dir = r"C:\Thesis\MScThesis\gc_experiments_results\plots"
# os.makedirs(output_base_dir, exist_ok=True)  # Create the base directory if it doesn't exist

# # Load data
# file_path = r"C:/Thesis/MScThesis/gc_experiments_results/plots/final_replacement_accuracies.csv"
# #file_path = r"C:/Thesis/MScThesis/gc_experiments_results/replacement/50art_comparison_metrics.csv"
# data = pd.read_csv(file_path)

# # Dictionary to store output paths
# output_paths = {}

# # Modularized function for plotting
# def plot_metric(
#     data, x_axis, group_by, y_axis, output_dir, sentences_per_chunk, title_prefix=""
# ):
#     """
#     Creates a plot for the specified metric (y_axis) against the x_axis variable,
#     with multiple lines grouped by the group_by variable.

#     Args:
#         data (pd.DataFrame): The dataset to use.
#         x_axis (str): The variable for the X-axis.
#         group_by (str): The variable to group by (each group gets its own line).
#         y_axis (str): The metric to plot on the Y-axis.
#         output_dir (str): Directory where the plots will be saved.
#         sentences_per_chunk (int): Value for sentences per chunk.
#         title_prefix (str): Prefix for the plot title (e.g., "Effect of").

#     Returns:
#         str: The path to the saved plot.
#     """
#     # Filter data for the specified Sentences Per Chunk
#     filtered_data = data[data["Sentences Per Chunk"] == sentences_per_chunk]

#     # Group and average over trials
#     grouped_data = (
#         filtered_data
#         .groupby([x_axis, group_by])
#         .mean(numeric_only=True)  # Only calculate averages for numeric columns
#         .reset_index()
#     )

#     # Automatically detect unique values of `group_by`
#     group_values = sorted(filtered_data[group_by].dropna().unique())  

#     # Dynamic labels for title and y-axis
#     metric_label = y_axis.replace("_", " ")  # Replace underscores with spaces for better readability
#     title = f"{title_prefix} {metric_label}\n(Sentences Per Chunk = {sentences_per_chunk})"
#     y_label = metric_label

#     # Create a figure
#     plt.figure(figsize=(10, 6))

#     # Plot for all unique group values
#     for group_value in group_values:
#         subset = grouped_data[grouped_data[group_by] == group_value]
#         if not subset.empty:
#             plt.plot(
#                 subset[x_axis], 
#                 subset[y_axis], 
#                 marker='o', 
#                 label=f"{group_by} {group_value}"
#             )

#     # Add labels, title, and legend
#     plt.xlabel(x_axis.replace("_", " ").capitalize())
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.legend(title=group_by.replace("_", " ").capitalize())
#     plt.grid()

#     # Save the plot dynamically based on the selected variables
#     plot_file = os.path.join(
#         output_dir,
#         f"{x_axis}_vs_{y_axis.lower().replace(' ', '_')}_grouped_by_{group_by.lower().replace(' ', '_')}_sent{sentences_per_chunk}.png"
#     )
#     plt.savefig(plot_file, dpi=300)  # Save with high resolution
#     plt.close()  # Close the figure to avoid overlap in memory

#     print(f"Plot saved to {plot_file}")
#     return plot_file  # Return the path to the saved plot


# # Metrics to analyze
# metrics_to_plot = [
#     "Avg Mean JS", 
#     "Avg Std JS", 
#     "Avg CV JS", 
#     "Avg First Order Diff JS", 
#     "Avg Num Peaks", 
#     "Avg Peak Ratio",
#     "Avg RMSE" 
# ]

# # Accuracy metrics to analyze
# accuracy_metrics_to_plot = [
#     "GC Acc Mean JS",
#     "GC Acc Std JS",
#     "GC Acc CV JS",
#     "GC Acc First Order Diff JS",
#     "GC Acc Second Order Diff JS",
#     "GC Acc Num Peaks",
#     "GC Acc Peak Ratio",
#     "GC Acc RMSE"
# ]


# # Sentences Per Chunk values to analyze
# sentences_per_chunk_values = [5]  

# # Function to generate plots for a specific metric type (averages or accuracies)
# def generate_plots(metric_list, prefix, sentences_per_chunk_values):
#     for sentences_per_chunk in sentences_per_chunk_values:
#         # Define output directories dynamically
#         output_dir_chunks = os.path.join(output_base_dir, f"final_{prefix}_chunks_affected_sent{sentences_per_chunk}")
#         os.makedirs(output_dir_chunks, exist_ok=True)

#         # Effect of % Chunks Affected, grouped by Sentences to Replace
#         for metric in metric_list:
#             path = plot_metric(
#                 data=data,
#                 x_axis="% Chunks Affected",
#                 group_by="Sentences to Replace",
#                 y_axis=metric,
#                 output_dir=output_dir_chunks,
#                 sentences_per_chunk=sentences_per_chunk,
#                 title_prefix=f"Effect of % Chunks Affected on"
#             )
#             # Store the path in the dictionary
#             output_paths[f"% Chunks Affected on {metric} (Sent {sentences_per_chunk})"] = path

#         output_dir_sentences = os.path.join(output_base_dir, f"final_{prefix}_sentences_replaced_sent{sentences_per_chunk}")
#         os.makedirs(output_dir_sentences, exist_ok=True)

#         # Effect of Sentences to Replace, grouped by % Chunks Affected
#         for metric in metric_list:
#             path = plot_metric(
#                 data=data,
#                 x_axis="Sentences to Replace",
#                 group_by="% Chunks Affected",
#                 y_axis=metric,
#                 output_dir=output_dir_sentences,
#                 sentences_per_chunk=sentences_per_chunk,
#                 title_prefix=f"Effect of Sentences to Replace on"
#             )
#             # Store the path in the dictionary
#             output_paths[f"Sentences to Replace on {metric} (Sent {sentences_per_chunk})"] = path


# # # Generate plots for averages
# generate_plots(metrics_to_plot, prefix="avgs", sentences_per_chunk_values=sentences_per_chunk_values)

# # Generate plots for accuracies
# generate_plots(accuracy_metrics_to_plot, prefix="acc", sentences_per_chunk_values=sentences_per_chunk_values)

# # Print all saved paths
# print("\nSaved Plot Paths:")
# for description, path in output_paths.items():
#     print(f"{description}: {path}")


import matplotlib.pyplot as plt

# Data from the table
chunks_affected = [10, 20, 30, 50, 75, 100]  # % chunks affected
values = [
    [0.599, 0.669, 0.701, 0.752, 0.782, 0.811],  # 1 Sentence
    [0.632, 0.701, 0.741, 0.781, 0.806, 0.831],  # 2 Sentences
    [0.660, 0.716, 0.776, 0.807, 0.824, 0.847],  # 3 Sentences
    [0.687, 0.764, 0.798, 0.845, 0.871, 0.892],  # 4 Sentences
    [0.706, 0.785, 0.846, 0.870, 0.890, 0.920]   # All Sentences
]

# Labels for the lines
labels = [
    "1 Sentence",
    "2 Sentences",
    "3 Sentences",
    "4 Sentences",
    "All Sentences"
]

# Plotting
plt.figure(figsize=(10, 6))

for i, value in enumerate(values):
    plt.plot(chunks_affected, value, marker='o', label=labels[i])

plt.title("Accuracy (%) in distinguishing original from modified articles")
plt.xlabel("% Chunks Affected")
plt.ylabel(" % Correct classifications")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
