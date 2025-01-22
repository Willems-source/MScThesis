import pandas as pd
from bertopic import BERTopic
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics
from src.data.preprocessing import split_into_chunks
import numpy as np

# Load the pre-trained BERTopic model
topic_model = BERTopic.load("isot_topic_model")
#topic_model = BERTopic.load("fitted_topic_model")

# Function to compute metrics for each article
def compute_metrics_for_dataset(data, sentences_per_chunk=5, min_chunks=3):
    # Initialize lists to store metrics
    metrics = []

    for index, row in data.iterrows():
        article_text = row['text']

        # Skip rows with invalid or missing text
        if not isinstance(article_text, str) or not article_text.strip():
            print(f"Skipping invalid text at index {index}")
            continue

        # Get topic probabilities
        try:
            _, original_probs = topic_model.transform([article_text])
        except Exception as e:
            print(f"Error transforming text at index {index}: {e}")
            continue

        # Split text into chunks
        chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)
        if len(chunks) == 0:
            print(f"Skipping article {index} due to insufficient chunks.")
            continue

        # Compute JS values and coherence metrics
        try:
            js_values = compute_js_values_for_chunks(chunks, original_probs[0], topic_model)
            # Skip if JS values are empty or contain NaNs
            if not js_values or np.isnan(js_values).any():
                print(f"Skipping article {index} due to invalid JS values.")
                continue
            coherence_metrics = compute_coherence_metrics(js_values)
        except Exception as e:
            print(f"Error computing metrics at index {index}: {e}")
            continue

        # Add metrics to the list
        metrics.append({
            "Mean JS": coherence_metrics.get("Mean JS", np.nan),
            "Std JS": coherence_metrics.get("Std JS", np.nan),
            "CV JS": coherence_metrics.get("CV JS", np.nan),
            "First Order Diff JS": coherence_metrics.get("First Order Diff JS", np.nan),
            "Second Order Diff JS": coherence_metrics.get("Second Order Diff JS", np.nan),
            "Num Peaks": coherence_metrics.get("Num Peaks", np.nan),
            "Peak Ratio": coherence_metrics.get("Peak Ratio", np.nan),
            "RMSE": coherence_metrics.get("RMSE", np.nan)
        })

    return pd.DataFrame(metrics)

# Function to process combined dataset
def process_combined_dataset(input_path, output_path_with_metrics, sentences_per_chunk=5, min_chunks=3, threshold=0.39, test_mode=False):
    # Load the combined dataset
    combined_data = pd.read_csv(input_path)

    # Ensure the text column is a string and drop rows with missing text
    combined_data['text'] = combined_data['text'].astype(str)
    combined_data = combined_data[combined_data['text'].str.strip().astype(bool)]

    # Select a subset of articles if in test mode
    if test_mode:
        real_subset = combined_data[combined_data['label'] == 'real'].head(1000)
        fake_subset = combined_data[combined_data['label'] == 'fake'].head(1000)
        combined_data = pd.concat([real_subset, fake_subset], ignore_index=True)
        print("Running in test mode: processing X articles from each category.")

    # Compute global coherence metrics
    global_metrics = compute_metrics_for_dataset(combined_data, sentences_per_chunk=sentences_per_chunk, min_chunks=min_chunks)

    # Merge metrics with the combined dataset
    combined_data_with_metrics = pd.concat([combined_data.reset_index(drop=True), global_metrics.reset_index(drop=True)], axis=1)

    # Threshold-based classification
    combined_data_with_metrics['predicted_label'] = combined_data_with_metrics['Mean JS'].apply(lambda x: 'real' if x <= threshold else 'fake')

    # Compute overall accuracy
    combined_data_with_metrics['correct'] = combined_data_with_metrics['label'] == combined_data_with_metrics['predicted_label']
    accuracy = combined_data_with_metrics['correct'].mean()

    print(f"Overall classification accuracy: {accuracy:.2f}")

    # Save the final dataset with metrics and predictions
    combined_data_with_metrics.to_csv(output_path_with_metrics, index=False)
    print(f"Dataset with metrics and predictions saved to {output_path_with_metrics}.")

# Paths for the combined dataset and output
input_path = "C:/Thesis/MScThesis/Thesis/src/data/combined_real_fake.csv"
output_path_with_metrics = "C:/Thesis/MScThesis/Thesis/src/data/isot_combined_real_fake_with_metrics.csv"

# Main execution
if __name__ == "__main__":
    # Set to True for testing the first 50 articles, False for the entire dataset
    test_mode = True  # Change to False to process the full dataset

    process_combined_dataset(
        input_path=input_path,
        output_path_with_metrics=output_path_with_metrics,
        sentences_per_chunk=5,
        min_chunks=3,
        threshold=0.38,
        test_mode=test_mode
    )

