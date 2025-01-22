# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE
# from scipy.stats import binomtest

# ### ----------------------- THRESHOLD -----------------------

# # File path
# file_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis/llm_human_metrics_general_50_topic_model.csv"

# # Load dataset
# data = pd.read_csv(file_path)

# # Function to calculate accuracy for a given threshold
# def calculate_accuracy(data, column, threshold):
#     data["predicted"] = data[column] > threshold
#     data["predicted_label"] = data["predicted"].apply(lambda x: "llm" if x else "human")
#     accuracy = (data["predicted_label"] == data["label"]).mean()
#     return accuracy

# # Define the metric and its threshold range
# metric = "RMSE"
# threshold_range = np.arange(0.2, 1.0, 0.005)

# # Balance the dataset using SMOTE
# X = data[[metric]].values
# y = data["label"].apply(lambda x: 1 if x == "llm" else 0).values
# smote = SMOTE(random_state=42)
# X_balanced, y_balanced = smote.fit_resample(X, y)
# data_balanced = pd.DataFrame(X_balanced, columns=[metric])
# data_balanced["label"] = y_balanced
# data_balanced["label"] = data_balanced["label"].apply(lambda x: "llm" if x == 1 else "human")

# # Initialize variables to store results
# accuracies = []

# for threshold in threshold_range:
#     accuracy = calculate_accuracy(data_balanced, metric, threshold)
#     accuracies.append((threshold, accuracy))

# thresholds, accuracy_values = zip(*accuracies)
# best_index = np.argmax(accuracy_values)
# best_threshold = thresholds[best_index]
# best_accuracy = accuracy_values[best_index]

# # Perform a significance test
# n_total = len(data_balanced)
# n_correct = int(best_accuracy * n_total)
# random_guess_accuracy = data_balanced["label"].value_counts(normalize=True).max()  # Baseline accuracy (random guessing)
# binomial_test = binomtest(n_correct, n_total, random_guess_accuracy, alternative='greater')

# # Print results
# print(f"Best RMSE Threshold: {best_threshold:.3f}")
# print(f"Best RMSE Accuracy: {best_accuracy:.4f}")
# print(f"P-value: {binomial_test.pvalue:.4f}")
# if binomial_test.pvalue < 0.05:
#     print("The accuracy is statistically significantly greater than random guessing.")
# else:
#     print("The accuracy is NOT statistically significantly greater than random guessing.")

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, accuracy_values, label=f"Accuracy (Best: {best_accuracy:.4f})", color="blue")
# plt.axvline(x=best_threshold, linestyle="--", color="red", label=f"Best Threshold ({best_threshold:.3f})")
# plt.title("Accuracy vs. Threshold for LLM vs Human Classification")
# plt.xlabel("Threshold")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Save the plot
# output_plot_path = "C:/Thesis/MScThesis/gc_experiments_results/plots/threshold_classification/rmse_llm_human_threshold_classification.png"
# plt.savefig(output_plot_path, dpi=300)
# plt.show()

# print(f"Plot saved to {output_plot_path}")

#### ----------------------- LR CLASSIFICATION -----------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import binomtest, chi2

# Define the file path for the dataset
file_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis/llm_human_metrics_general_50_topic_model.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Debugging Check 1: Print unique labels before processing
print("Unique labels in 'label' column:", data["label"].unique())

# Debugging Check 2: Print counts of each label
print("Counts of each label:")
print(data["label"].value_counts())

# Standardize the label column (strip whitespace, make lowercase)
data["label"] = data["label"].str.strip().str.lower()

# Debugging Check 3: Print unique labels after formatting
print("Unique labels after formatting:", data["label"].unique())

# Debugging Check 4: Print the shape of the dataset after dropping NaN
print("Shape of dataset after dropping NaN values:", data.shape)

# Define features and target
features = ["RMSE"]
X = data[features]
y = data["label"].apply(lambda x: 1 if x == "llm" else 0)  # Encode labels: llm=1, human=0

# Debugging Check 5: Print the distribution of labels after mapping
print("Label distribution after mapping (1: llm, 0: human):")
print(y.value_counts())

# Oversample the dataset
ros = RandomOverSampler(random_state=42)

# Debugging Check 6: Check the shape of X and y before oversampling
print("Shape of X and y before oversampling:", X.shape, y.shape)

# Perform oversampling
X_resampled, y_resampled = ros.fit_resample(X, y)

# Debugging Check 7: Print the distribution of labels after oversampling
print("Label distribution after oversampling (1: llm, 0: human):")
print(pd.Series(y_resampled).value_counts())

# Split the resampled dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create output directory
output_dir = "C:/Thesis/MScThesis/gc_experiments_results/plots/lr/llm_human"
os.makedirs(output_dir, exist_ok=True)

confusion_matrix_path = os.path.join(output_dir, "llm_human_confusion_matrix_lr_rmse_only.png")
feature_importance_path = os.path.join(output_dir, "llm_human_feature_importance_lr_rmse_only.png")

# Save the confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Human", "LLM"], yticklabels=["Human", "LLM"])
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(confusion_matrix_path, dpi=300)
plt.close()
print(f"Confusion matrix plot saved to {confusion_matrix_path}")

# Analyze feature importance
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Importance:")
print(coefficients)

# Save the feature importance as a bar plot
plt.figure(figsize=(6, 4))
plt.barh(coefficients["Feature"], coefficients["Coefficient"])
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.savefig(feature_importance_path, dpi=300)
plt.close()
print(f"Feature importance plot saved to {feature_importance_path}")

# Test if RMSE is a significant predictor using the Wald test
rmse_coefficient = model.coef_[0][0]
standard_error = (X_train.std(axis=0) / len(X_train))  # Estimate standard error
wald_statistic = (rmse_coefficient / standard_error[0]) ** 2  # Wald statistic
p_value_rmse = chi2.sf(wald_statistic, 1)  # 1 degree of freedom

print(f"\nWald Test for RMSE Coefficient:")
print(f"  RMSE Coefficient: {rmse_coefficient:.4f}")
print(f"  P-value: {p_value_rmse:.4f}")
if p_value_rmse < 0.05:
    print("RMSE is a statistically significant predictor of the label (reject H0).")
else:
    print("RMSE is NOT a statistically significant predictor of the label (fail to reject H0).")

# Test if the accuracy is significantly greater than random guessing
correct_predictions = sum(y_test == y_pred)
total_predictions = len(y_test)
baseline_accuracy = 0.5  # Null hypothesis: random guessing
binomial_test = binomtest(correct_predictions, total_predictions, p=baseline_accuracy, alternative='greater')

print(f"\nBinomial Test for Accuracy:")
print(f"  P-value: {binomial_test.pvalue:.4f}")
if binomial_test.pvalue < 0.05:
    print("The observed accuracy is statistically significantly greater than random guessing.")
else:
    print("The observed accuracy is not statistically significantly greater than random guessing.")
