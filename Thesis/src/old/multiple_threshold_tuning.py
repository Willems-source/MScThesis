import pandas as pd
import numpy as np
from itertools import product

# Load the dataset
file_path = "C:/Thesis/MScThesis/Thesis/src/data/combined_real_fake_with_metrics.csv"
data = pd.read_csv(file_path)

# Define ranges for thresholds based on common sense or data analysis
mean_js_range = np.arange(0.35, 0.45, 0.005)  # Threshold for Mean JS
std_js_range = np.arange(0.1, 0.3, 0.01)     # Threshold for Std JS
rmse_range = np.arange(0.3, 0.6, 0.005)       # Threshold for RMSE
peak_ratio_range = np.arange(0.0, 0.2, 0.01) # Threshold for Peak Ratio

# Define rules for classification
def classify(row, mean_js_threshold, std_js_threshold, rmse_threshold, peak_ratio_threshold):
    """
    Classify an article as 'real' or 'fake' based on the thresholds.
    """
    if (row["Mean JS"] > mean_js_threshold and
        row["Std JS"] < std_js_threshold and
        row["RMSE"] < rmse_threshold and
        row["Peak Ratio"] < peak_ratio_threshold):
        return "fake"
    else:
        return "real"

# Iterate over all combinations of thresholds
results = []
for mean_js, std_js, rmse, peak_ratio in product(mean_js_range, std_js_range, rmse_range, peak_ratio_range):
    # Apply classification for the current combination of thresholds
    data["predicted_label"] = data.apply(
        classify, 
        axis=1, 
        args=(mean_js, std_js, rmse, peak_ratio)
    )
    # Calculate accuracy
    accuracy = (data["predicted_label"] == data["label"]).mean()
    
    # Store results
    results.append({
        "Mean JS Threshold": mean_js,
        "Std JS Threshold": std_js,
        "RMSE Threshold": rmse,
        "Peak Ratio Threshold": peak_ratio,
        "Accuracy": accuracy
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Find the best combination
best_combination = results_df.loc[results_df["Accuracy"].idxmax()]
print("Best Combination of Thresholds:")
print(best_combination)

# Save results to a CSV file
output_file = "C:/Thesis/MScThesis/gc_experiments_results/decision_boundary_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")




