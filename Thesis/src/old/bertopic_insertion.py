import pandas as pd
from bertopic import BERTopic
from src.data.load_data import load_preprocessed_wsj_data
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics, create_topic_based_permutations, load_unrelated_sentence_pool

# Load preprocessed WSJ data
data_path = "C:\Thesis\MScThesis\Thesis\src\data\wsj_cleaned_min_sent_length_6.csv"
corpus = load_preprocessed_wsj_data(data_path)

# Convert to a list of documents
documents = corpus["text"].tolist()

# Load the pre-trained BERTopic model
topic_model = BERTopic.load("fitted_topic_model")

# Load the precomputed sentence pool
sentence_pool_path = "C:\Thesis\MScThesis\Thesis\src\data\sentence_pool.csv"
sentence_pool_df = load_unrelated_sentence_pool(sentence_pool_path)

# Experiment Configuration
experiment_config = {
    "swap_type": "single_chunk",  # Options: single_sentence_one_chunk, single_sentence_all_chunks, single_chunk
    "num_permutations": 20,
    "min_chunks": 3,
    "sentences_per_chunk": 5,
}

# Initialize results container
all_results = []

# Iterate over the first X articles
for article_id in range(1, 11):
    article_text = documents[article_id - 1]  # Get the article text
    article_topic, original_probs = topic_model.transform([article_text])

    # Run the experiment for the current article
    permutations = create_topic_based_permutations(
        article_text,
        article_id,
        article_topic[0],  # Topic ID of the article
        sentence_pool_df,
        num_permutations=experiment_config["num_permutations"],
        sentences_per_chunk=experiment_config["sentences_per_chunk"],
        swap_type=experiment_config["swap_type"],
    )

    # Process the original article
    original_chunks = split_into_chunks(
        article_text, experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
    )
    original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)

    all_results.append({
        "Article_ID": article_id,
        "Type": "Original",
        "Text": article_text,
        **{f"Chunk_{i + 1}_T": chunk for i, chunk in enumerate(original_chunks)},
        **{f"Chunk_{i + 1}_J": js for i, js in enumerate(original_js_values)},
        **compute_coherence_metrics(original_js_values),
    })

    # Analyze the permutations
    for perm in permutations:
        perm_chunks = split_into_chunks(
            perm["Text"], experiment_config["sentences_per_chunk"], experiment_config["min_chunks"]
        )
        perm_js_values = compute_js_values_for_chunks(perm_chunks, original_probs[0], topic_model)

        all_results.append({
            "Article_ID": article_id,
            "Type": f"Permutation_{perm['Permutation_Index']} ({experiment_config['swap_type']})",
            "Text": perm["Text"],
            **{f"Chunk_{i + 1}_T": chunk for i, chunk in enumerate(perm_chunks)},
            **{f"Chunk_{i + 1}_J": js for i, js in enumerate(perm_js_values)},
            **compute_coherence_metrics(perm_js_values),
        })

# Save all results to a CSV file
output_path = f"C:/Thesis/MScThesis/gc_experiments_results/insertion/TRIALTWO_insertion_experiment_{experiment_config['swap_type']}_20articles.csv"
pd.DataFrame(all_results).to_csv(output_path, index=False)
print(f"Results saved to {output_path}")


