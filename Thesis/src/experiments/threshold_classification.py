import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split

# 1) Read the data
isot_health_file_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/fakehealth_metrics_finetuned_isot_50_topics_model_fulldataset.csv"
data = pd.read_csv(isot_health_file_path)

# Convert labels to binary
#   'fake' -> 1, 'real' -> 0
data['label_binary'] = data['label'].apply(lambda x: 1 if x == 'fake' else 0)

# 2) Split into train/test (70/30)
train_data, test_data = train_test_split(
    data, test_size=0.3, random_state=42, stratify=data['label_binary']
)

# 3) Function to compute accuracy given a threshold on a dataset
def compute_accuracy(dataset, metric_column, threshold):
    dataset['predicted_label'] = dataset[metric_column].apply(lambda x: 'fake' if x > threshold else 'real')
    accuracy = (dataset['predicted_label'] == dataset['label']).mean()
    return accuracy

# 4) Find best threshold on the training set
metric_column = "RMSE"
thresholds = np.arange(0.0, 1.0, 0.005)  # or whatever range you want
best_threshold = None
best_train_accuracy = 0.0

for t in thresholds:
    train_acc = compute_accuracy(train_data, metric_column, t)
    if train_acc > best_train_accuracy:
        best_train_accuracy = train_acc
        best_threshold = t

print(f"Best threshold found on training set: {best_threshold:.3f} (Train Accuracy={best_train_accuracy:.4f})")

# 5) Evaluate final performance on the test set
test_accuracy = compute_accuracy(test_data, metric_column, best_threshold)
print(f"Test Accuracy using best threshold: {test_accuracy:.4f}")

# 6) Statistical significance test (comparing to random guessing)
n_total = len(test_data)
n_correct = int(test_accuracy * n_total)
majority_class_prob = test_data['label'].value_counts(normalize=True).max()
binomial_res = binomtest(n_correct, n_total, majority_class_prob, alternative='greater')

print("\nBinomial Test Results (Test set):")
print(f"  p-value: {binomial_res.pvalue:.4f}")
if binomial_res.pvalue < 0.05:
    print(" => Statistically better than random guessing.")
else:
    print(" => Not significantly better than random guessing.")

# 7) Calculate Cohen's d
random_guess_accuracy = majority_class_prob
pooled_std = np.sqrt(((test_accuracy * (1 - test_accuracy)) + (random_guess_accuracy * (1 - random_guess_accuracy))) / 2)
cohen_d = (test_accuracy - random_guess_accuracy) / pooled_std
print(f"\nCohen's d: {cohen_d:.4f}")

# 8) Plot Training Accuracies vs. Threshold
train_accuracies = []

for t in thresholds:
    train_acc = compute_accuracy(train_data, metric_column, t)
    train_accuracies.append(train_acc)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, train_accuracies, label="Training Accuracy", color="blue")
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f"Best Threshold ({best_threshold:.3f})")
plt.scatter(best_threshold, best_train_accuracy, color='red', label=f"Train Acc={best_train_accuracy:.4f}")
plt.title("Training Accuracy vs. Threshold")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
output_plot_path = "training_accuracy_threshold_plot.png"
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"Training accuracy plot saved to {output_plot_path}")
