import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from src.data.load_data import load_preprocessed_wsj_data
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, create_topic_based_permutations, load_sentence_pool

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
chunks_affected_values = [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
sentences_to_replace_per_chunk_values = [1, 2, 3, 5, 'all']
sentences_per_chunk_values = [3, 5]
num_articles = 50  # Process 50 articles
summary_output_path = "C:/Thesis/MScThesis/gc_experiments_results/replacement/rmse_summary_metrics.csv"

# Start runs for all parameter combinations
for sentences_per_chunk in sentences_per_chunk_values:
    print(f"Running RMSE for sentences_per_chunk={sentences_per_chunk}")

    for chunks_affected in chunks_affected_values:
        for sentences_to_replace_per_chunk in sentences_to_replace_per_chunk_values:
            if sentences_to_replace_per_chunk != 'all' and sentences_to_replace_per_chunk > sentences_per_chunk:
                continue
            if sentences_to_replace_per_chunk == sentences_per_chunk:
                continue

            actual_sentences_to_replace = (
                sentences_per_chunk if sentences_to_replace_per_chunk == 'all'
                else sentences_to_replace_per_chunk
            )

            print(f"Running for chunks_affected={chunks_affected}, sentences_to_replace_per_chunk={sentences_to_replace_per_chunk}")
            experiment_config = {
                "chunks_affected": chunks_affected,
                "sentences_to_replace_per_chunk": sentences_to_replace_per_chunk,
                "num_trials": 3,
                "min_chunks": 3,
                "sentences_per_chunk": sentences_per_chunk,
            }

            selected_documents = documents[:num_articles]

            for trial in range(1, experiment_config["num_trials"] + 1):
                print(f"Starting RMSE Trial {trial} for {num_articles} articles")

                # Initialize RMSE-specific metrics container
                rmse_metrics = {
                    "RMSE": [],
                    "GC Accuracies": {"RMSE": 0},
                    "Total Permutations": {"RMSE": 0},
                }

                for article_id, article_text in enumerate(selected_documents):
                    article_topic, original_probs = topic_model.transform([article_text])
                    original_chunks = split_into_chunks(
                        article_text, experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
                    )
                    original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)

                    # Calculate RMSE directly
                    rmse_value = (sum(value**2 for value in original_js_values) / len(original_js_values))**0.5
                    rmse_metrics["RMSE"].append(rmse_value)

                    # Process permutations for this article
                    permutations, _ = create_topic_based_permutations(
                        article_text, article_id, article_topic[0], sentence_pool_df,
                        num_permutations=len(original_chunks),
                        sentences_per_chunk=experiment_config["sentences_per_chunk"],
                        chunks_affected=experiment_config["chunks_affected"],
                        sentences_to_replace_per_chunk=experiment_config["sentences_to_replace_per_chunk"],
                    )

                    for perm in permutations:
                        perm_chunks = split_into_chunks(
                            perm["Text"], experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
                        )
                        perm_js_values = compute_js_values_for_chunks(perm_chunks, original_probs[0], topic_model)

                        # Calculate RMSE for permutation
                        perm_rmse = (sum(value**2 for value in perm_js_values) / len(perm_js_values))**0.5
                        rmse_metrics["RMSE"].append(perm_rmse)

                        # Update GC accuracy counters
                        rmse_metrics["GC Accuracies"]["RMSE"] += int(rmse_value < perm_rmse)
                        rmse_metrics["Total Permutations"]["RMSE"] += 1

                # Compute overall RMSE metrics (including both originals and permutations)
                total_metrics = len(rmse_metrics["RMSE"])
                overall_metrics = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Dataset": num_articles,
                    "% Chunks Affected": experiment_config["chunks_affected"],
                    "Sentences Per Chunk": experiment_config["sentences_per_chunk"],
                    "Sentences to Replace": experiment_config["sentences_to_replace_per_chunk"],
                    "Trials": trial,
                    "Avg RMSE": sum(rmse_metrics["RMSE"]) / total_metrics,
                    "GC Acc RMSE": (
                        rmse_metrics["GC Accuracies"]["RMSE"] /
                        rmse_metrics["Total Permutations"]["RMSE"]
                        if rmse_metrics["Total Permutations"]["RMSE"] > 0 else 0
                    ),
                }

                # Save RMSE-specific metrics to CSV
                try:
                    with open(summary_output_path, 'r') as file:
                        file_exists = True
                except FileNotFoundError:
                    file_exists = False

                pd.DataFrame([overall_metrics]).to_csv(
                    summary_output_path, mode='a', header=not file_exists, index=False
                )

                print(f"RMSE results appended for Trial {trial}, chunks_affected={chunks_affected}, sentences_to_replace_per_chunk={sentences_to_replace_per_chunk}, sentences_per_chunk={sentences_per_chunk}")
