import os
import pandas as pd
from bertopic import BERTopic
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics
from src.data.preprocessing import split_into_chunks
import numpy as np

# Load the pre-trained BERTopic model
topic_model_path = "C:/Thesis/MScThesis/Thesis/topic_models/finetuned_isot_50_topics_model"  # Update path to model
#topic_model_path = "C:/Thesis/MScThesis/Thesis/topic_models/general_50_topic_model"  # Update path to model
#topic_model_path = "C:/Thesis/MScThesis/Thesis/topic_models/health_50_topic_model" 

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
                #"Second Order Diff JS": coherence_metrics.get("Second Order Diff JS", np.nan),
                "Num Peaks": coherence_metrics.get("Num Peaks", np.nan),
                "Peak Ratio": coherence_metrics.get("Peak Ratio", np.nan),
                "RMSE": coherence_metrics.get("RMSE", np.nan)
            })
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            continue

    return pd.DataFrame(metrics)

# Function to process dataset and save dynamically
def process_combined_dataset(input_path, output_dir, article_size=100, sentences_per_chunk=5, min_chunks=3, test_mode=True):
    combined_data = pd.read_csv(input_path)
    combined_data['text'] = combined_data['text'].astype(str).str.strip()
    combined_data = combined_data[combined_data['text'].astype(bool)]

    if test_mode:
        real_subset = combined_data[combined_data['label'] == 'real'].head(article_size)
        fake_subset = combined_data[combined_data['label'] == 'fake'].head(article_size)
        combined_data = pd.concat([real_subset, fake_subset], ignore_index=True)

    # Compute metrics
    metrics_df = compute_metrics_for_dataset(combined_data, sentences_per_chunk, min_chunks)

    # Merge metrics with input data
    combined_data_with_metrics = pd.concat([combined_data.reset_index(drop=True), metrics_df], axis=1)

    # Dynamically generate filename
    filename = f"ISOTv2_real_fake_metrics_{topic_model_name}_{article_size}.csv"
    output_path = os.path.join(output_dir, filename)

    # Save to CSV
    combined_data_with_metrics.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

# Main execution
if __name__ == "__main__":
    #input_path = "C:/Thesis/MScThesis/data/FakeHealthdata.csv"
    input_path = "C:/Thesis/MScThesis/Thesis/src/data/new_combined_real_fake.csv"
    output_dir = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/final"

    os.makedirs(output_dir, exist_ok=True)
    process_combined_dataset(
        input_path=input_path,
        output_dir=output_dir,
        article_size=500,  # Number of articles
        sentences_per_chunk=6,
        min_chunks=2,
        test_mode=True  # Set to False for full dataset
    )

# #2612
# import os
# import pandas as pd
# from bertopic import BERTopic
# from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics
# from src.data.preprocessing import split_into_chunks
# import numpy as np

# # Load the pre-trained BERTopic model
# topic_model_path = "C:/Thesis/MScThesis/Thesis/topic_models/finetuned_isot_50_topics_model"  # Update path to model
# topic_model = BERTopic.load(topic_model_path)

# # Extract the topic model name for dynamic naming
# topic_model_name = os.path.splitext(os.path.basename(topic_model_path))[0]

# # Function to compute metrics for each article
# def compute_metrics_for_dataset(data, sentences_per_chunk=5, min_chunks=3):
#     metrics = []
#     for index, row in data.iterrows():
#         article_text = row['text']
#         try:
#             if not isinstance(article_text, str) or not article_text.strip():
#                 raise ValueError("Invalid or empty article text")

#             # Log article details
#             with open("error_details.log", "a") as log_file:
#                 log_file.write(f"\n\nProcessing Index {index}:\n")
#                 log_file.write(f"Article Length: {len(article_text)}\n")
#                 log_file.write(f"Article Text: {article_text[:500]}...\n")

#             # Transform full article
#             _, original_probs = topic_model.transform([article_text])
#             original_probs = np.clip(original_probs[0], 0, 1)  # Clip probabilities to [0, 1]
#             if np.isnan(original_probs).any():
#                 raise ValueError("Original probabilities contain NaN")

#             # Log probabilities
#             with open("error_details.log", "a") as log_file:
#                 log_file.write(f"Original Probs: {original_probs}\n")

#             # Split article into chunks
#             chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)
#             if len(chunks) == 0:
#                 raise ValueError("No valid chunks generated")

#             # Log chunks
#             with open("error_details.log", "a") as log_file:
#                 log_file.write(f"Generated Chunks: {chunks}\n")

#             # Compute JS values
#             js_values = compute_js_values_for_chunks(chunks, original_probs, topic_model)
#             if not js_values or np.isnan(js_values).any() or np.isinf(js_values).any():
#                 raise ValueError(f"Invalid JS Values: {js_values}")

#             # Log JS values
#             with open("error_details.log", "a") as log_file:
#                 log_file.write(f"JS Values: {js_values}\n")

#             # Compute coherence metrics
#             coherence_metrics = compute_coherence_metrics(js_values)

#             metrics.append({
#                 "Mean JS": coherence_metrics.get("Mean JS", np.nan),
#                 "Std JS": coherence_metrics.get("Std JS", np.nan),
#                 "CV JS": coherence_metrics.get("CV JS", np.nan),
#                 "First Order Diff JS": coherence_metrics.get("First Order Diff JS", np.nan),
#                 "Num Peaks": coherence_metrics.get("Num Peaks", np.nan),
#                 "Peak Ratio": coherence_metrics.get("Peak Ratio", np.nan),
#                 "RMSE": coherence_metrics.get("RMSE", np.nan)
#             })

#         except Exception as e:
#             # Log error details for debugging
#             with open("error_details.log", "a") as log_file:
#                 log_file.write(f"Error at Index {index}: {e}\n")
#             metrics.append({
#                 "Mean JS": 0,
#                 "Std JS": 0,
#                 "CV JS": 0,
#                 "First Order Diff JS": 0,
#                 "Num Peaks": 0,
#                 "Peak Ratio": 0,
#                 "RMSE": 0
#             })

#     return pd.DataFrame(metrics)

# # Function to process dataset and save dynamically
# def process_combined_dataset(input_path, output_dir, article_size=100, sentences_per_chunk=5, min_chunks=3, test_mode=True):
#     combined_data = pd.read_csv(input_path)
#     combined_data['text'] = combined_data['text'].astype(str).str.strip()
#     combined_data = combined_data[combined_data['text'].astype(bool)]  # Remove empty text rows

#     if test_mode:
#         real_subset = combined_data[combined_data['label'] == 'real'].head(article_size)
#         fake_subset = combined_data[combined_data['label'] == 'fake'].head(article_size)
#         combined_data = pd.concat([real_subset, fake_subset], ignore_index=True)

#     # Compute metrics
#     metrics_df = compute_metrics_for_dataset(combined_data, sentences_per_chunk, min_chunks)

#     # Merge metrics with input data
#     combined_data_with_metrics = pd.concat([combined_data.reset_index(drop=True), metrics_df], axis=1)

#     # Save to CSV
#     filename = f"2912_real_fake_metrics_{topic_model_name}_{article_size}.csv"
#     output_path = os.path.join(output_dir, filename)
#     combined_data_with_metrics.to_csv(output_path, index=False)

# # Main execution
# if __name__ == "__main__":
#     input_path = "C:/Thesis/MScThesis/Thesis/src/data/new_combined_real_fake.csv"
#     output_dir = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison"

#     os.makedirs(output_dir, exist_ok=True)

#     process_combined_dataset(
#         input_path=input_path,
#         output_dir=output_dir,
#         article_size=100,  # Number of articles
#         sentences_per_chunk=5,
#         min_chunks=3,
#         test_mode=True  # Set to False for full dataset
#     )

