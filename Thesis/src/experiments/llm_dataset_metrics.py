# import os
# import pandas as pd

# # Define the directory containing the CSV files
# directory = r"C:\Thesis\MScThesis\data\raw\mage_used"

# # Define the output file path
# output_file = os.path.join(directory, "llm_generated_text.csv")

# # Initialize an empty list to store DataFrames
# dataframes = []

# # Iterate through the files in the directory
# for file in os.listdir(directory):
#     if file.endswith(".csv"):
#         # Extract prompt strategy and source model from the file name
#         file_name_parts = file.split("_")
#         prompt_strategy = file_name_parts[2]  # continuation, specified, topical
#         source_model = file_name_parts[-1].replace(".csv", "")  # model part

#         # Read the CSV file
#         file_path = os.path.join(directory, file)
#         df = pd.read_csv(file_path)

#         # Add the new columns
#         df["prompt strategy"] = prompt_strategy
#         df["source model"] = source_model

#         # Append the DataFrame to the list
#         dataframes.append(df)

# # Concatenate all DataFrames into one
# final_df = pd.concat(dataframes, ignore_index=True)

# # Save the final DataFrame to a CSV file
# final_df.to_csv(output_file, index=False)

# print(f"Concatenated CSV saved to {output_file}")


# import pandas as pd

# # Load the dataset
# file_path = "C:/Thesis/MScThesis/data/raw/mage_used/llm_generated_text.csv"  # Replace with the actual path to your file
# data = pd.read_csv(file_path)

# # Calculate sentence count for each article based on the number of periods as a rough estimate
# data['sentence_count'] = data['text'].apply(lambda x: x.count('.') + x.count('!') + x.count('?'))

# # Group by source model and prompt strategy to compute the required values
# summary = (
#     data.groupby(["source model", "prompt strategy"])
#     .agg(
#         Articles=("text", "count"),  # Count the number of articles
#         Avg_Sentences_Per_Article=("sentence_count", "mean")  # Average sentences per article
#     )
#     .reset_index()
# )

# # Rename columns for clarity
# summary.rename(columns={
#     "source model": "Model Variant",
#     "prompt strategy": "Prompt Strategy",
#     "Articles": "Number of Articles",
#     "Avg_Sentences_Per_Article": "Avg. Sentences/article"
# }, inplace=True)

# # Save to a CSV or print
# summary.to_csv("summary_results.csv", index=False)  # Save the results
# print(summary)  # Print the summary

import os
import pandas as pd
from bertopic import BERTopic
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics
from src.data.preprocessing import split_into_chunks
import numpy as np

# Load the pre-trained BERTopic model
topic_model_path = "C:/Thesis/MScThesis/Thesis/topic_models/general_50_topic_model" 
topic_model = BERTopic.load(topic_model_path)

# Extract the topic model name for dynamic naming
topic_model_name = os.path.splitext(os.path.basename(topic_model_path))[0]  

# Function to compute metrics for each article
def compute_metrics_for_dataset(data, sentences_per_chunk=5, min_chunks=3):
    metrics = []
    for index, row in data.iterrows():
        article_text = row['text']

        if not isinstance(article_text, str) or not article_text.strip():
            continue  # Skip invalid rows

        try:
            _, original_probs = topic_model.transform([article_text])
            original_probs = np.clip(original_probs[0], 0, 1)  # Clip probabilities to [0, 1]

            chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)
            if len(chunks) == 0:
                continue  # Skip insufficient chunks

            # Compute JS values with checks
            js_values = compute_js_values_for_chunks(chunks, original_probs, topic_model)
            if not js_values or np.isnan(js_values).any() or np.isinf(js_values).any():
                print(f"Skipping invalid JS values at index {index}")
                with open("invalid_js_values.log", "a") as log_file:
                    log_file.write(f"Index {index}: {js_values}\n")
                continue

            coherence_metrics = compute_coherence_metrics(js_values)

            metrics.append({
                "Mean JS": coherence_metrics.get("Mean JS", np.nan),
                "Std JS": coherence_metrics.get("Std JS", np.nan),
                "CV JS": coherence_metrics.get("CV JS", np.nan),
                "First Order Diff JS": coherence_metrics.get("First Order Diff JS", np.nan),
                "Num Peaks": coherence_metrics.get("Num Peaks", np.nan),
                "Peak Ratio": coherence_metrics.get("Peak Ratio", np.nan),
                "RMSE": coherence_metrics.get("RMSE", np.nan)
            })
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            continue

    return pd.DataFrame(metrics)

# Function to process dataset and save dynamically
def process_llm_generated_text(input_path, output_dir, model_filters=None, sentences_per_chunk=5, min_chunks=3, test_mode=True):
    combined_data = pd.read_csv(input_path)
    combined_data['text'] = combined_data['text'].astype(str).str.strip()
    combined_data = combined_data[combined_data['text'].astype(bool)]

    # Filter by models if filters are provided
    if model_filters:
        combined_data = combined_data[combined_data['source model'].str.contains('|'.join(model_filters), case=False, na=False)]

    if test_mode:
        combined_data = combined_data.head(100)  # Use a smaller subset for testing

    # Compute metrics
    metrics_df = compute_metrics_for_dataset(combined_data, sentences_per_chunk, min_chunks)

    # Merge metrics with input data
    combined_data_with_metrics = pd.concat([combined_data.reset_index(drop=True), metrics_df], axis=1)

    # Dynamically generate filename
    model_filters_name = "all_models" if not model_filters else "_".join(model_filters)
    filename = f"{model_filters_name}_llm_metrics_{topic_model_name}.csv"
    output_path = os.path.join(output_dir, filename)

    # Save to CSV
    combined_data_with_metrics.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

# Main execution
if __name__ == "__main__":
    input_path = "C:/Thesis/MScThesis/data/raw/mage_used/llm_generated_text.csv"
    output_dir = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis"

    os.makedirs(output_dir, exist_ok=True)

    # Specify the models to filter (e.g., ["13B", "7B", "65B"] for specific models or [] for all models)
    model_filters = None #["13B", "7B", "65B"]  # None for all models, otherwise a list of model names

    process_llm_generated_text(
        input_path=input_path,
        output_dir=output_dir,
        model_filters=None, # set to None for all
        sentences_per_chunk=5,
        min_chunks=3,
        test_mode=False  # Set to False for processing the full dataset
    )

# import pandas as pd
# from scipy.stats import ttest_ind

# # Load the dataset
# file_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis/llm_human_metrics_general_50_topic_model.csv"
# data = pd.read_csv(file_path)

# # Columns to analyze
# metrics = ['Mean JS', 'Std JS', 'CV JS', 'First Order Diff JS', 'Num Peaks', 'Peak Ratio', 'RMSE']

# # Group data by label
# llm_data = data[data['label'] == 'llm']
# human_data = data[data['label'] == 'human']

# # Calculate averages and perform t-tests
# for metric in metrics:
#     llm_avg = llm_data[metric].mean()
#     human_avg = human_data[metric].mean()
#     t_stat, p_value = ttest_ind(llm_data[metric], human_data[metric], equal_var=False)
    
#     print(f"Metric: {metric}")
#     print(f"LLM Average: {llm_avg:.4f}")
#     print(f"Human Average: {human_avg:.4f}")
#     print(f"P-Value: {p_value:.4e}")
#     print(f"Statistically Significant: {'Yes' if p_value < 0.05 else 'No'}")
#     print("-" * 50)



### ----------------- WITH JS VALUES SAVED -------------
