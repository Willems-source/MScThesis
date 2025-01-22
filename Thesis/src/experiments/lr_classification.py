# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import os
# from scipy.stats import binomtest, chi2

# datasets = {
#     "isot_isot": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/isot_metrics_finetuned_isot_50_topics_model_fulldataset.csv",
#     "health_isot": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/fakehealth_metrics_finetuned_isot_50_topics_model_fulldataset.csv",
#     "health_general": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/fakehealth_metrics_general_50_topic_model_fulldataset.csv",
#     "isot_general": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/isot_metrics_general_50_topic_model_fulldataset.csv"
# }

# selected_dataset = "health_isot" #change where needed
# file_path = datasets[selected_dataset] 
# data = pd.read_csv(file_path)

# dataset_type = "isot" if "isot" in selected_dataset else "health"
# topic_source = "generic" if "general" in selected_dataset else "isot"

# # Drop rows with NaN values
# data = data.dropna()

# # Define features and target
# features = ["RMSE"]
# X = data[features]
# y = data["label"].apply(lambda x: 1 if x == "fake" else 0)  # Encode labels: fake=1, real=0

# # Split the dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# # Initialize and train the logistic regression model
# model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Create output directory
# output_dir = f"C:/Thesis/MScThesis/gc_experiments_results/plots/lr/{dataset_type}_{topic_source}"
# os.makedirs(output_dir, exist_ok=True)

# confusion_matrix_path = os.path.join(output_dir, f"{dataset_type}_{topic_source}_confusion_matrix_lr_rmse_only.png")
# feature_importance_path = os.path.join(output_dir, f"{dataset_type}_{topic_source}_feature_importance_lr_rmse_only.png")

# # Save the confusion matrix plot
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.savefig(confusion_matrix_path, dpi=300)
# plt.close()
# print(f"Confusion matrix plot saved to {confusion_matrix_path}")

# # Analyze feature importance
# coefficients = pd.DataFrame({
#     "Feature": features,
#     "Coefficient": model.coef_[0]
# }).sort_values(by="Coefficient", ascending=False)

# print("\nFeature Importance:")
# print(coefficients)

# # Save the feature importance as a bar plot
# plt.figure(figsize=(6, 4))
# plt.barh(coefficients["Feature"], coefficients["Coefficient"])
# plt.title("Feature Importance (Logistic Regression Coefficients)")
# plt.xlabel("Coefficient Value")
# plt.savefig(feature_importance_path, dpi=300)
# plt.close()
# print(f"Feature importance plot saved to {feature_importance_path}")

# # Test if RMSE is a significant predictor using the Wald test
# rmse_coefficient = model.coef_[0][0]
# standard_error = (X_train.std(axis=0) / len(X_train))  # Estimate standard error
# wald_statistic = (rmse_coefficient / standard_error[0]) ** 2  # Wald statistic
# p_value_rmse = chi2.sf(wald_statistic, 1)  # 1 degree of freedom

# print(f"\nWald Test for RMSE Coefficient:")
# print(f"  RMSE Coefficient: {rmse_coefficient:.4f}")
# print(f"  P-value: {p_value_rmse:.4f}")
# if p_value_rmse < 0.05:
#     print("RMSE is a statistically significant predictor of the label (reject H0).")
# else:
#     print("RMSE is NOT a statistically significant predictor of the label (fail to reject H0).")

# # Test if the accuracy is significantly greater than random guessing
# correct_predictions = sum(y_test == y_pred)
# total_predictions = len(y_test)
# baseline_accuracy = 0.5  # Null hypothesis: random guessing
# binomial_test = binomtest(correct_predictions, total_predictions, p=baseline_accuracy, alternative='greater')

# print(f"\nBinomial Test for Accuracy:")
# print(f"  P-value: {binomial_test.pvalue:.4f}")
# if binomial_test.pvalue < 0.05:
#     print("The observed accuracy is statistically significantly greater than random guessing.")
# else:
#     print("The observed accuracy is not statistically significantly greater than random guessing.")
# # Save the confusion matrix plot
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# # Adjust output path based on dataset
# if file_path == isot_file_path:
#     confusion_matrix_path = os.path.join(output_dir, "isot_confusion_matrix_lr_rmse_only.png")
# else:
#     confusion_matrix_path = os.path.join(output_dir, "health_confusion_matrix_lr_rmse_only.png")

# plt.savefig(confusion_matrix_path, dpi=300)
# plt.close()
# print(f"Confusion matrix plot saved to {confusion_matrix_path}")

# # Analyze feature importance
# coefficients = pd.DataFrame({
#     "Feature": features,
#     "Coefficient": model.coef_[0]
# }).sort_values(by="Coefficient", ascending=False)

# print("\nFeature Importance:")
# print(coefficients)

# # Test if RMSE is a significant predictor using the Wald test
# rmse_coefficient = model.coef_[0][0]
# standard_error = (X_train.std(axis=0) / len(X_train))  # Estimate standard error
# wald_statistic = (rmse_coefficient / standard_error[0]) ** 2  # Wald statistic
# p_value_rmse = chi2.sf(wald_statistic, 1)  # 1 degree of freedom

# print(f"\nWald Test for RMSE Coefficient:")
# print(f"  RMSE Coefficient: {rmse_coefficient:.4f}")
# print(f"  P-value: {p_value_rmse:.4f}")
# if p_value_rmse < 0.05:
#     print("RMSE is a statistically significant predictor of the label (reject H0).")
# else:
#     print("RMSE is NOT a statistically significant predictor of the label (fail to reject H0).")

# # Test if the accuracy is significantly greater than random guessing
# correct_predictions = sum(y_test == y_pred)
# total_predictions = len(y_test)
# baseline_accuracy = 0.5  # Null hypothesis: random guessing
# binomial_test = binomtest(correct_predictions, total_predictions, p=baseline_accuracy, alternative='greater')

# print(f"\nBinomial Test for Accuracy:")
# print(f"  P-value: {binomial_test.pvalue:.4f}")
# if binomial_test.pvalue < 0.05:
#     print("The observed accuracy is statistically significantly greater than random guessing.")
# else:
#     print("The observed accuracy is not statistically significantly greater than random guessing.")

# correct_predictions = 5475
# total_predictions = 9677
# baseline_accuracy = 0.5 #H0
# binomial_test = binomtest(correct_predictions, total_predictions, p=baseline_accuracy, alternative='greater')
# print(f"Binomial test p-value: {binomial_test.pvalue:.4f}")
# if binomial_test.pvalue < 0.05:
#     print("The observed accuracy is statistically significantly greater than random guessing.")
# else:
#     print("The observed accuracy is not statistically significantly greater than random guessing.")


# ------------------------ START OF THE SCRIPT ------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import binomtest, chi2

datasets = {
    "isot_isot": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/isot_metrics_finetuned_isot_50_topics_model_fulldataset.csv",
    "health_isot": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/fakehealth_metrics_finetuned_isot_50_topics_model_fulldataset.csv",
    "health_general": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/fakehealth_metrics_general_50_topic_model_fulldataset.csv",
    "isot_general": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/isot_metrics_general_50_topic_model_fulldataset.csv",
    "health_health": "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final/health_real_fake_metrics_health_50_topic_model_fulldataset.csv"
}

selected_dataset = "isot_isot" #change where needed
file_path = datasets[selected_dataset] 
data = pd.read_csv(file_path)

dataset_type = "isot" if "isot" in selected_dataset else "health"
topic_source = "generic" if "general" in selected_dataset else "isot"

# Drop rows with NaN values
data = data.dropna()

# Define features and target
features = ["RMSE"]
X = data[features]
y = data["label"].apply(lambda x: 1 if x == "fake" else 0)  # Encode labels: fake=1, real=0

# Oversample the dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

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
output_dir = f"C:/Thesis/MScThesis/gc_experiments_results/plots/lr/{dataset_type}_{topic_source}"
os.makedirs(output_dir, exist_ok=True)

confusion_matrix_path = os.path.join(output_dir, f"{dataset_type}_{topic_source}_confusion_matrix_lr_rmse_only.png")
feature_importance_path = os.path.join(output_dir, f"{dataset_type}_{topic_source}_feature_importance_lr_rmse_only.png")

# Save the confusion matrix plot
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(confusion_matrix_path, dpi=300)
plt.show()
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
#plt.savefig(feature_importance_path, dpi=300)
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

