import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from src.data.load_data import load_preprocessed_wsj_data
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics, create_topic_based_permutations, load_sentence_pool
import numpy as np

# Load preprocessed WSJ data
data_path = "C:/Thesis/MScThesis/Thesis/src/data/wsj_cleaned_min_sent_length_6.csv"
corpus = load_preprocessed_wsj_data(data_path)

# Convert to a list of documents
documents = corpus["text"].tolist()

# Load the pre-trained BERTopic model
topic_model = BERTopic.load("fitted_topic_model")

# Load the precomputed sentence pool
sentence_pool_path = "C:/Thesis/MScThesis/Thesis/src/data/full_sentence_pool_tpds.csv"
sentence_pool_df = load_sentence_pool(sentence_pool_path)

# Define the parameter grids
chunks_affected_values = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0] #[0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
sentences_to_replace_per_chunk_values = [1, 2, 3, 4, 'all']
sentences_per_chunk_values = [5] #[3,5,7] # Loop over these values

def valid_js_values(js_values):
    """ Check if JS values are valid and do not contain NaN or inf. """
    return js_values and not np.isnan(js_values).any() and not np.isinf(js_values).any()

# Start runs for all parameter combinations
for sentences_per_chunk in sentences_per_chunk_values:
    print(f"Running for sentences_per_chunk={sentences_per_chunk}")

    for chunks_affected in chunks_affected_values:
        for sentences_to_replace_per_chunk in sentences_to_replace_per_chunk_values:
            # Skip cases where sentences_to_replace is greater than sentences_per_chunk
            if sentences_to_replace_per_chunk != 'all' and sentences_to_replace_per_chunk > sentences_per_chunk:
                continue  # Skip invalid configurations

            # Skip numerical cases equivalent to full replacement
            if sentences_to_replace_per_chunk == sentences_per_chunk:
                continue  # Skip this case and only process 'all'

            # Dynamically adjust sentences_to_replace for 'all'
            if sentences_to_replace_per_chunk == 'all':
                actual_sentences_to_replace = sentences_per_chunk
            else:
                actual_sentences_to_replace = sentences_to_replace_per_chunk

            print(f"Running for chunks_affected={chunks_affected}, sentences_to_replace_per_chunk={sentences_to_replace_per_chunk}")

            # Experiment Configuration
            experiment_config = {
                "dataset": "test",  # Options: "test" (first X articles) or "full" (entire dataset)
                "chunks_affected": chunks_affected,
                "sentences_to_replace_per_chunk": sentences_to_replace_per_chunk,
                "num_trials": 3,  # Number of trials for randomness
                "min_chunks": 3,
                "sentences_per_chunk": sentences_per_chunk,  # Use the current value from the loop
            }

            if experiment_config["dataset"] == "test":
                selected_documents = documents[:100]  # First X articles
                dataset_description = len(selected_documents)  # Use the number of selected articles
            elif experiment_config["dataset"] == "full":
                selected_documents = documents
                dataset_description = "full"  # Indicate that the entire dataset is analyzed

            # Process articles for each trial
            for trial in range(1, experiment_config["num_trials"] + 1):
                print(f"Starting Trial {trial}")

                # Initialize global metrics container
                gc_metrics = {
                    "Mean JS": [],
                    "Std JS": [],
                    "CV JS": [],
                    "First Order Diff JS": [],
                    "Second Order Diff JS": [],
                    "Num Peaks": [],
                    "Peak Ratio": [],
                    "RMSE": [], 
                    "GC Accuracies": {
                        "Mean JS": 0,
                        "Std JS": 0,
                        "CV JS": 0,
                        "First Order Diff JS": 0,
                        "Second Order Diff JS": 0,
                        "Num Peaks": 0,
                        "Peak Ratio": 0,
                        "RMSE": 0,  
                    },
                    "Total Permutations": {
                        "Mean JS": 0,
                        "Std JS": 0,
                        "CV JS": 0,
                        "First Order Diff JS": 0,
                        "Second Order Diff JS": 0,
                        "Num Peaks": 0,
                        "Peak Ratio": 0,
                        "RMSE": 0,  
                    },
                    "Total Articles": 0
                }

                for article_id, article_text in enumerate(selected_documents):
                    # Get full article topic probabilities
                    article_topic, original_probs = topic_model.transform([article_text])

                    # Split original article into chunks
                    original_chunks = split_into_chunks(
                        article_text, experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
                    )
                    num_chunks = len(original_chunks)
                    original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)

                    if not valid_js_values(original_js_values):
                        print(f"Skipping Article {article_id} due to invalid JS values.")
                        continue

                    # Compute metrics for the original article
                    original_metrics = compute_coherence_metrics(original_js_values)

                    # Add original metrics to the global container
                    for metric in ["Mean JS", "Std JS", "CV JS", "First Order Diff JS", "Second Order Diff JS", "Num Peaks", "Peak Ratio", "RMSE"]:
                        gc_metrics[metric].append(original_metrics[metric])

                    # Initialize counters for GC accuracies for this article
                    article_gc_accuracies = {
                        "Mean JS": 0, "Std JS": 0, "CV JS": 0,
                        "First Order Diff JS": 0, "Second Order Diff JS": 0,
                        "Num Peaks": 0, "Peak Ratio": 0, "RMSE": 0
                    }

                    # Run the experiment for the current article with permutations tied to the number of chunks
                    permutations, all_affected_chunks = create_topic_based_permutations(
                        article_text,
                        article_id,
                        article_topic[0],  # Topic ID of the article
                        sentence_pool_df,
                        num_permutations=num_chunks,  # Number of permutations equals the number of chunks
                        sentences_per_chunk=experiment_config["sentences_per_chunk"],
                        chunks_affected=experiment_config["chunks_affected"],
                        sentences_to_replace_per_chunk=experiment_config["sentences_to_replace_per_chunk"],
                    )

                    for perm in permutations:
                        perm_chunks = split_into_chunks(
                            perm["Text"], experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
                        )

                        perm_js_values = compute_js_values_for_chunks(perm_chunks, original_probs[0], topic_model)
                        perm_metrics = compute_coherence_metrics(perm_js_values)

                        # Update global metrics with permutation metrics
                        for metric in ["Mean JS", "Std JS", "CV JS", "First Order Diff JS", "Second Order Diff JS", "Num Peaks", "Peak Ratio", "RMSE"]:
                            gc_metrics[metric].append(perm_metrics[metric])

                        # Update GC accuracy counters for this article
                        for metric in article_gc_accuracies.keys():
                            article_gc_accuracies[metric] += int(original_metrics[metric] < perm_metrics[metric])
                            gc_metrics["Total Permutations"][metric] += 1  # Increment total permutations

                    # Aggregate GC accuracies for this article
                    for metric in article_gc_accuracies.keys():
                        gc_metrics["GC Accuracies"][metric] += article_gc_accuracies[metric]

                    gc_metrics["Total Articles"] += 1

                # Compute overall metrics (including both originals and permutations)
                total_metrics = len(gc_metrics["Mean JS"])  # Includes both originals and permutations
                overall_metrics = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Dataset": dataset_description,
                    "% Chunks Affected": experiment_config["chunks_affected"],
                    "Sentences Per Chunk": experiment_config["sentences_per_chunk"],
                    "Sentences to Replace": experiment_config["sentences_to_replace_per_chunk"],
                    "Trials": trial,  # Trial number
                    "Avg Mean JS": sum(gc_metrics["Mean JS"]) / total_metrics,
                    "Avg Std JS": sum(gc_metrics["Std JS"]) / total_metrics,
                    "Avg CV JS": (sum(gc_metrics["Std JS"]) / total_metrics) / (sum(gc_metrics["Mean JS"]) / total_metrics),
                    "Avg First Order Diff JS": sum(gc_metrics["First Order Diff JS"]) / total_metrics,
                    "Avg Second Order Diff JS": sum(gc_metrics["Second Order Diff JS"]) / total_metrics,
                    "Avg Num Peaks": sum(gc_metrics["Num Peaks"]) / total_metrics,
                    "Avg Peak Ratio": sum(gc_metrics["Peak Ratio"]) / total_metrics,
                    "Avg RMSE": sum(gc_metrics["RMSE"]) / total_metrics,
                }

                # Add GC Accuracy for all metrics
                for metric, count in gc_metrics["GC Accuracies"].items():
                    total_permutations = gc_metrics["Total Permutations"][metric]
                    overall_metrics[f"GC Acc {metric}"] = count / total_permutations if total_permutations > 0 else 0

                # Save trial-specific metrics to CSV
                # summary_output_path = f"C:/Thesis/MScThesis/gc_experiments_results/replacement/comparison_summary_metrics.csv"
                summary_output_path = f"C:/Thesis/MScThesis/gc_experiments_results/replacement/final_replacement_metrics.csv"

                # Check if the file exists to determine whether to include the header
                try:
                    with open(summary_output_path, 'r') as file:
                        file_exists = True
                except FileNotFoundError:
                    file_exists = False

                # Write to the CSV file in append mode
                pd.DataFrame([overall_metrics]).to_csv(
                    summary_output_path,
                    mode='a',  # Append mode
                    header=not file_exists,  # Write header only if the file does not exist
                    index=False
                )

                print(f"Results appended for Trial {trial}, chunks_affected={chunks_affected}, sentences_to_replace_per_chunk={sentences_to_replace_per_chunk}, sentences_per_chunk={sentences_per_chunk}")



