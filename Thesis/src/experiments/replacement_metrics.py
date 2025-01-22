import pandas as pd
from datetime import datetime
from bertopic import BERTopic
from src.data.load_data import load_preprocessed_wsj_data
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics, create_topic_based_permutations, load_sentence_pool

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

# Experiment Configuration
experiment_config = {
    "dataset": "test",  # Options: "test" (first X articles) or "full" (entire dataset)
    "chunks_affected": 0.75,  # Percentage of chunks to affect, 0 to 1
    "sentences_to_replace_per_chunk": 'all',  # Number of sentences to replace per chunk (from 1,2,3,4 to "all")
    "num_trials": 3,  # Number of trials for randomness
    "min_chunks": 3,
    "sentences_per_chunk": 5,
}

# Select subset of articles based on dataset configuration
if experiment_config["dataset"] == "test":
    documents = documents[:100]  # First X articles
elif experiment_config["dataset"] == "full":
    pass  # Use all articles

# Initialize global metrics container
# Initialize global metrics container
gc_metrics = {
    "Mean JS": [],
    "Std JS": [],
    "CV JS": [],
    "First Order Diff JS": [],
    "Second Order Diff JS": [],
    "Num Peaks": [],
    "Peak Ratio": [],
    "GC Accuracies": {
        "Mean JS": 0,
        "Std JS": 0,
        "CV JS": 0,
        "First Order Diff JS": 0,
        "Second Order Diff JS": 0,
        "Num Peaks": 0,
        "Peak Ratio": 0,
    },
    "Total Articles": 0
}

# Start processing articles
for article_id, article_text in enumerate(documents):
    # Get full article topic probabilities
    article_topic, original_probs = topic_model.transform([article_text])

    # Split original article into chunks
    original_chunks = split_into_chunks(
        article_text, experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
    )
    num_chunks = len(original_chunks)
    original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)

    # Compute metrics for the original article
    original_metrics = compute_coherence_metrics(original_js_values)

    # Add original metrics to the global container
    for metric in ["Mean JS", "Std JS", "CV JS", "First Order Diff JS", "Second Order Diff JS", "Num Peaks", "Peak Ratio"]:
        gc_metrics[metric].append(original_metrics[metric])

    # Initialize counters for GC accuracies
    gc_accuracies = {
        "Mean JS": 0, "Std JS": 0, "CV JS": 0,
        "First Order Diff JS": 0, "Second Order Diff JS": 0,
        "Num Peaks": 0, "Peak Ratio": 0
    }

    # Run multiple trials
    for trial in range(experiment_config["num_trials"]):
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

        # Process permutations
        for perm in permutations:
            perm_chunks = split_into_chunks(
                perm["Text"], experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
            )

            perm_js_values = compute_js_values_for_chunks(perm_chunks, original_probs[0], topic_model)
            perm_metrics = compute_coherence_metrics(perm_js_values)

            # Add permutation metrics to the global container
            for metric in ["Mean JS", "Std JS", "CV JS", "First Order Diff JS", "Second Order Diff JS", "Num Peaks", "Peak Ratio"]:
                gc_metrics[metric].append(perm_metrics[metric])

            # Update GC accuracy counters
            for metric in gc_accuracies.keys():
                gc_accuracies[metric] += int(original_metrics[metric] < perm_metrics[metric])

    # Aggregate GC accuracies for this article
    for metric in gc_accuracies.keys():
        gc_metrics["GC Accuracies"][metric] += gc_accuracies[metric]

    gc_metrics["Total Articles"] += 1

# Compute overall metrics (including both originals and permutations)
total_metrics = len(gc_metrics["Mean JS"])  # Total metrics collected (originals + permutations)
overall_metrics = {
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Dataset": experiment_config["dataset"],
    "% Chunks Affected": experiment_config["chunks_affected"],
    "Sentences Per Chunk": experiment_config["sentences_per_chunk"],
    "Sentences to Replace": experiment_config["sentences_to_replace_per_chunk"],
    "Trials": experiment_config["num_trials"],
    "Avg Mean JS": sum(gc_metrics["Mean JS"]) / total_metrics,
    "Avg Std JS": sum(gc_metrics["Std JS"]) / total_metrics,
    "Avg CV JS": (sum(gc_metrics["Std JS"]) / total_metrics) / (sum(gc_metrics["Mean JS"]) / total_metrics),
    "Avg First Order Diff JS": sum(gc_metrics["First Order Diff JS"]) / total_metrics,
    "Avg Second Order Diff JS": sum(gc_metrics["Second Order Diff JS"]) / total_metrics,
    "Avg Num Peaks": sum(gc_metrics["Num Peaks"]) / total_metrics,
    "Avg Peak Ratio": sum(gc_metrics["Peak Ratio"]) / total_metrics,
}

# Add GC Accuracy for all metrics
total_permutations = gc_metrics["Total Articles"] * experiment_config["num_trials"] * num_chunks
for metric, count in gc_metrics["GC Accuracies"].items():
    overall_metrics[f"GC Acc {metric}"] = count / total_permutations

# Save overall metrics to CSV
summary_output_path = f"C:/Thesis/MScThesis/gc_experiments_results/replacement/summary_metrics.csv"

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

print(f"Results appended to {summary_output_path}")
