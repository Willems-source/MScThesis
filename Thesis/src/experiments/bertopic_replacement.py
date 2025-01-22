import pandas as pd
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
    "chunks_affected": 0.25,  # Percentage of chunks to affect (0 to 1)
    "sentences_to_replace_per_chunk": 'all',  # Number of sentences to replace per chunk (1,2,..., "all")
    "num_trials": 3,  # Number of trials to account for randomness
    "min_chunks": 3,
    "sentences_per_chunk": 5,
}

# Initialize results container
all_results = []

# Iterate over the first X articles
for article_id, article_text in enumerate(documents[:10]):  # Processing first 10 articles
    # Get the article's topic probabilities
    article_topic, original_probs = topic_model.transform([article_text])

    # Split original article into chunks
    original_chunks = split_into_chunks(
        article_text, experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
    )
    num_chunks = len(original_chunks)

    # Compute JS values and metrics for the original article
    original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)
    original_metrics = compute_coherence_metrics(original_js_values)

    # Add original article results
    all_results.append({
        "Article_ID": article_id + 1,
        "Type": "Original",
        "Text": article_text,
        **{f"Chunk_{i + 1}_T": chunk for i, chunk in enumerate(original_chunks)},
        **{f"Chunk_{i + 1}_J": js for i, js in enumerate(original_js_values)},
        **original_metrics,
    })

    # Run multiple trials
    for trial in range(experiment_config["num_trials"]):
        # Create permutations for the article
        permutations, all_affected_chunks = create_topic_based_permutations(
            article_text,
            article_id,
            article_topic[0],  # Topic ID of the article
            sentence_pool_df,
            num_permutations=num_chunks,  # Number of permutations = number of chunks
            sentences_per_chunk=experiment_config["sentences_per_chunk"],
            chunks_affected=experiment_config["chunks_affected"],
            sentences_to_replace_per_chunk=experiment_config["sentences_to_replace_per_chunk"],
        )

        # Analyze the permutations
        for perm_idx, perm in enumerate(permutations):
            perm_chunks = split_into_chunks(
                perm["Text"], experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
            )

            # Ensure perm_chunks and original_chunks have the same length
            if len(perm_chunks) != len(original_chunks):
                if len(perm_chunks) > len(original_chunks):
                    perm_chunks = perm_chunks[:len(original_chunks)]  # Truncate extra chunks
                else:
                    perm_chunks.extend([""] * (len(original_chunks) - len(perm_chunks)))  # Pad with empty chunks

            # Get affected chunks for this permutation
            affected_chunks = all_affected_chunks[perm_idx]

            # Validate unaffected chunks and enforce alignment
            for i in range(len(perm_chunks)):
                if i not in affected_chunks and perm_chunks[i] != original_chunks[i]:
                    perm_chunks[i] = original_chunks[i]  # Restore original chunk

            # Compute JS values and metrics for the permutation
            perm_js_values = compute_js_values_for_chunks(perm_chunks, original_probs[0], topic_model)
            perm_metrics = compute_coherence_metrics(perm_js_values)

            # Add permutation results
            all_results.append({
                "Article_ID": article_id + 1,
                "Type": f"Permutation_{perm['Permutation_Index']} (Trial {trial + 1}, Chunks Affected: {experiment_config['chunks_affected']}, Sentences Replaced: {experiment_config['sentences_to_replace_per_chunk']})",
                "Text": " ".join(perm_chunks),
                **{f"Chunk_{i + 1}_T": chunk for i, chunk in enumerate(perm_chunks)},
                **{f"Chunk_{i + 1}_J": js for i, js in enumerate(perm_js_values)},
                **perm_metrics,
            })

# Save all results to a CSV file
output_path = f"C:/Thesis/MScThesis/gc_experiments_results/replacement/rmse{experiment_config['chunks_affected']}chunks_{experiment_config['sentences_to_replace_per_chunk']}sentences.csv"
pd.DataFrame(all_results).to_csv(output_path, index=False)
print(f"Results saved to {output_path}")
