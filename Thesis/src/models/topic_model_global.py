import import_ipynb
import re
import ftfy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import random
import gensim.corpora as corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
import spacy
from gensim import corpora, models
import gensim
from gensim.matutils import cossim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import seaborn as sns
from scipy.spatial.distance import cosine, jensenshannon
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import find_peaks

from Thesis.src.data.preprocessing import preprocess_text, split_into_chunks

true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

# STILL TO FIX: MAKE THIS MORE ROBUST
min_length_true_df = true_df[true_df['word_count'] > 400 ]
min_length_fake_df = fake_df[fake_df['word_count'] > 400 ]

sample_article =min_length_true_df.sample(n=1).iloc[0, 1]
sample_article = preprocess_text(sample_article, fix_encoding=True, is_fake=False)

article_chunks = split_into_chunks(sample_article, num_chunks=5)

# --- SETUP --- 
model_name = "valurank/distilroberta-topic-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
topic_model = AutoModelForSequenceClassification.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

# Check if label mappings are defined in the configuration
if hasattr(config, 'id2label'):
    label_mapping = config.id2label  # id2label maps index to class labels
else:
    raise ValueError("The model does not have label mappings defined.")

max_tokens = 512

# --- HELPER FUNCTIONS ---
def display_top_n_topics(prob_vector, labels, num_topics):
    """
    Display the top N topics and their probabilities.
    Parameters:
        prob_vector: Full probability vector for a given chunk or article.
        labels: Corresponding labels for the probability vector.
        num_display_topics: Number of top topics to display.
    """
    # Sort the indices based on probabilities in descending order
    sorted_indices = np.argsort(prob_vector)[::-1]
    sorted_probs = prob_vector[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]

    # Display only the top-N results
    # for i in range(min(num_topics, len(sorted_probs))):
    #     print(f"  - {sorted_labels[i]}: {sorted_probs[i]:.4f}")

def get_alphabetically_ordered_probabilities(prob_vector, label_mapping):
    """
    Order the probability vector alphabetically by the labels in label_mapping.
    Returns:
        ordered_probs: Probability vector ordered alphabetically.
        ordered_labels: Corresponding labels in alphabetical order.
    """
    # Create an ordered list of labels based on their alphabetic position
    ordered_labels = sorted(label_mapping.values())
    # Create a mapping from label to the index in the probability vector
    index_mapping = {label_mapping[idx]: idx for idx in range(len(label_mapping))}
    
    # Use the index mapping to order probabilities
    ordered_probs = [prob_vector[index_mapping[label]] for label in ordered_labels]

    return np.array(ordered_probs), ordered_labels

# --- DEFINE FUNCTIONS ---
def get_topic_probabilities_article(full_article, num_topics):
    """
    Get the full probability distribution for the entire article, ordered alphabetically by topics.
    Parameters:
        full_article: String text of the article.
        num_display_topics: Number of top topics to display.
    """
    # Tokenize and process the full article through the model
    inputs = tokenizer(full_article, return_tensors="pt", truncation=True, max_length=max_tokens)
    with torch.no_grad():
        outputs = topic_model(**inputs)

    # Get the full probability distribution for the entire article
    full_article_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # Order alphabetically by topic labels
    ordered_probs, ordered_labels = get_alphabetically_ordered_probabilities(full_article_probabilities, label_mapping)

    # # Display the top N topics
    # print("Full article topics:")
    # display_top_n_topics(ordered_probs, ordered_labels, num_topics)

    return ordered_probs

def get_topic_probabilities_chunks(article_chunks, num_topics):
    """
    Get the full probability distribution for each chunk, ordered alphabetically by topics.
    Parameters:
        article_chunks: List of strings, each representing a chunk.
        num_display_topics: Number of top topics to display for each chunk.
    """
    chunk_results = []

    # Process each chunk separately
    for idx, chunk in enumerate(article_chunks):
        # Tokenize and process the chunk through the model
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_tokens)
        with torch.no_grad():
            outputs = topic_model(**inputs)

        # Get the full probability distribution for the chunk
        chunk_probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Order alphabetically by topic labels
        ordered_probs, ordered_labels = get_alphabetically_ordered_probabilities(chunk_probabilities, label_mapping)

        # Store results for each chunk
        chunk_results.append({
            "chunk_index": idx + 1,
            "prob_distribution": ordered_probs,
            "labels": ordered_labels
        })

        # # Display the top N topics
        # print(f"Chunk {idx + 1}:")
        # display_top_n_topics(ordered_probs, ordered_labels, num_topics)

    return chunk_results


full_article = sample_article
full_article_probabilities = get_topic_probabilities_article(full_article, num_topics=5)
chunk_probabilities = get_topic_probabilities_chunks(article_chunks, num_topics=5)


def calculate_js_for_chunks(full_article_probs, chunk_results, num_topics):
    """
    Compare the full article's topic distribution with each chunk's distribution using Jensen-Shannon Divergence.
    
    Parameters:
    - full_article_probs: Probability vector for the full article (alphabetically ordered).
    - chunk_results: List of chunk results from get_topic_probabilities_chunks().
    - top_n_topics: Number of top topics to display and focus on.

    Returns:
    - js_divergences: List of JS Divergence values for each chunk.
    """
    js_divergences = []

    # Print header for readability
    print(f"Comparing Full Article to Each Chunk:\n")

    # Compare each chunk's probability distribution with the full article's distribution
    for chunk in chunk_results:
        chunk_index = chunk["chunk_index"]
        chunk_probs = chunk["prob_distribution"]

        # Calculate Jensen-Shannon Divergence
        js_divergence = jensenshannon(full_article_probs, chunk_probs)

        # Store the result
        js_divergences.append(js_divergence)

        # Print the divergence score along with top topics for clarity
        print(f"Chunk {chunk_index} to Full Article: JS Divergence = {js_divergence:.4f}")

    return js_divergences

# Compare the chunk distributions to the full article
js_divergences = calculate_js_for_chunks(full_article_probabilities, chunk_probabilities, num_topics=5)

# Step 1: Calculate Topic Embeddings
topics = sorted(label_mapping.values())  # Get the list of all topics in alphabetical order
model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a lightweight sentence transformer model
topic_embeddings = model.encode(topics)

# Step 2: Compute Similarity Matrix
similarity_matrix = cosine_similarity(topic_embeddings)

# Create a dictionary to access topic similarity scores
topic_sim_dict = {}
for i, topic1 in enumerate(topics):
    topic_sim_dict[topic1] = {}
    for j, topic2 in enumerate(topics):
        topic_sim_dict[topic1][topic2] = similarity_matrix[i, j]

# Step 3: Define a Weighted JS Divergence Function
def weighted_js_divergence(p, q, labels, similarity_threshold=0.7):
    """
    Compute JS divergence between two probability vectors, weighted by topic similarity.
    
    Parameters:
    - p: Probability vector for the full article
    - q: Probability vector for the chunk
    - labels: List of topic labels corresponding to the probability vectors
    - similarity_threshold: Threshold to decide if topics are similar enough to influence divergence

    Returns:
    - Weighted JS divergence score
    """
    # Compute average distribution
    m = 0.5 * (p + q)
    
    # Calculate the standard KL divergence for each pair (p||m) and (q||m)
    kl_pm = np.where(p != 0, p * np.log(p / m), 0)
    kl_qm = np.where(q != 0, q * np.log(q / m), 0)

    # Apply similarity-based weighting
    for i, label_p in enumerate(labels):
        for j, label_q in enumerate(labels):
            if i != j and p[i] > 0 and q[j] > 0:  # Skip zero entries
                similarity = topic_sim_dict[label_p][label_q]
                if similarity > similarity_threshold:
                    # Reduce KL divergence if topics are highly similar
                    kl_pm[i] *= (1 - similarity)
                    kl_qm[j] *= (1 - similarity)

    # Return the weighted JS divergence
    return 0.5 * (kl_pm.sum() + kl_qm.sum())

# Step 4: Calculate Weighted JS Divergences for All Chunks
def calculate_weighted_js_for_chunks(full_article_probs, chunk_results, labels, similarity_threshold=0.7):
    """
    Calculate weighted JS divergence for each chunk compared to the full article.
    
    Parameters:
    - full_article_probs: Probability vector for the full article
    - chunk_results: List of chunk results from get_topic_probabilities_chunks()
    - labels: Alphabetically ordered labels for topics
    - similarity_threshold: Threshold for considering topic similarity

    Returns:
    - weighted_js_divergences: List of weighted JS divergence values for each chunk
    """
    weighted_js_divergences = []  # Store the weighted JS divergence for each chunk

    # print("\nWeighted Jensen-Shannon Divergence Scores for Each Chunk vs. Full Article (Using Semantic Similarity):")

    # Calculate weighted JS divergence for each chunk
    for chunk in chunk_results:
        chunk_index = chunk["chunk_index"]
        chunk_probs = chunk["prob_distribution"]

        # Calculate the weighted JS divergence
        weighted_js_score = weighted_js_divergence(full_article_probs, chunk_probs, labels, similarity_threshold)
        weighted_js_divergences.append(weighted_js_score)

        # Print the result for each chunk
        # print(f"Chunk {chunk_index} to Full Article: Weighted JS Divergence = {weighted_js_score:.4f}")

    return weighted_js_divergences

# Step 5: Run the calculation for all chunks and print the results
weighted_js_divergences = calculate_weighted_js_for_chunks(full_article_probabilities, chunk_probabilities, topics)

def plot_js_divergences(js_divergences, num_chunks, weighted_js_divergences=None):

    plt.figure(figsize=(8, 5))
    plt.bar([f"Chunk {i+1}" for i in range(num_chunks)], js_divergences, color='skyblue', label='Unweighted JS Divergence')
    if weighted_js_divergences is not None:
        plt.plot([f"Chunk {i+1}" for i in range(num_chunks)], weighted_js_divergences, color='orange', marker='o', linestyle='--', label='Weighted JS Divergence')
    plt.xlabel("Chunks")
    plt.ylabel("Jensen-Shannon Divergence")
    plt.title("Jensen-Shannon Divergence between Each Chunk and the Full Article Topics")
    plt.xticks(rotation=45)
    # Add a legend if weighted JS is included
    if weighted_js_divergences is not None:
        plt.legend()
    plt.show()

# Plot the results using your already defined variables
plot_js_divergences(js_divergences, len(js_divergences), weighted_js_divergences=weighted_js_divergences)
#out comment if you want to exclude weighted values
#plot_js_divergences(js_divergences, len(js_divergences))

js_values = np.array(js_divergences)
weighted_js_values = np.array(weighted_js_divergences)

def compute_peak_analysis(divergence_values, prominence_threshold=0.10):
    """
    Perform peak analysis on a given set of JS divergence values.
    Parameters:
        divergence_values: List or array of JS divergence values for each chunk.
        prominence_threshold: Minimum prominence of peaks to be considered significant.

    Returns:
        peaks: Indices of the identified peaks.
        peak_heights: Heights of the identified peaks.
        peak_score: Aggregate score based on the prominence and height of identified peaks.
    """
    divergence_values = np.array(divergence_values)

    # Detect peaks and their properties
    peaks, properties = find_peaks(divergence_values, prominence=prominence_threshold)
    
    # Calculate peak heights and prominence
    peak_heights = divergence_values[peaks]
    peak_prominences = properties['prominences']
    
    # Calculate a peak score as the sum of the prominences (weighted by height)
    peak_score = np.sum(peak_heights * peak_prominences)
    num_peaks = int(len(peaks))

    # Return the peaks along with the other values
    return peaks, num_peaks, peak_heights, peak_score, properties

# Run peak analysis on your JS divergence values (unweighted)
peaks, num_peaks, peak_heights, peak_score, properties = compute_peak_analysis(js_values)
peaks_weighted, num_peaks_weighted, peak_heights_weighted, peak_score_weighted, properties_weighted = compute_peak_analysis(weighted_js_values)

# Output the results for unweighted
print(f"Unweighted JS Divergence:")
print(f"Identified Peaks: {num_peaks}")
print(f"Peak Heights: {peak_heights}")
print(f"Peak Score: {peak_score:.4f}")
print(f"Peak Prominences: {properties['prominences']}")

# Output the results for weighted
print(f"Weighted JS Divergence:")
print(f"Identified Peaks: {num_peaks_weighted}")
print(f"Peak Heights: {peak_heights_weighted}")
print(f"Peak Score: {peak_score_weighted:.4f}")
print(f"Peak Prominences: {properties_weighted['prominences']}")

# Plot the results with integer chunk index starting from 1
plt.figure(figsize=(8, 6))

# Plot for unweighted JS divergence
plt.plot(np.arange(1, len(js_values) + 1), js_values, marker='o', linestyle='-', label='Unweighted JS Divergence', color='blue')
plt.scatter(peaks + 1, js_values[peaks], color='red', label='Peaks (Unweighted)', zorder=5)  # Adjust peak indices to start from 1
for i, height in zip(peaks, peak_heights):
    plt.annotate(f"{height:.2f}", (i + 1, height), textcoords="offset points", xytext=(0, 5), ha='center')

# Plot for weighted JS divergence
plt.plot(np.arange(1, len(weighted_js_values) + 1), weighted_js_values, marker='o', linestyle='--', label='Weighted JS Divergence', color='green')
plt.scatter(peaks_weighted + 1, weighted_js_values[peaks_weighted], color='orange', label='Peaks (Weighted)', zorder=5)  # Adjust peak indices to start from 1
for i, height in zip(peaks_weighted, peak_heights_weighted):
    plt.annotate(f"{height:.2f}", (i + 1, height), textcoords="offset points", xytext=(0, 5), ha='center')

# Customize x-axis to show only integers starting from 1
plt.xticks(np.arange(1, len(js_values) + 1))
plt.xlabel('Chunk')
plt.ylabel('JS Divergence')
plt.title('Peak Analysis on JS Divergence (Unweighted vs Weighted)')
plt.legend()
plt.show()


def compute_coherence_metrics(js_values, num_peaks, peak_score, baseline=0):
    """
    Compute stability and peak analysis metrics for a given list of JS divergence values.
    Parameters:
        js_values: List of JS divergence values for each chunk.
        num_peaks: Number of peaks identified in JS values.
        peak_score: Peak score from peak analysis.
        baseline: Baseline value for computing deviations (default is 0).
        
    Returns:
        stability_metrics: Dictionary containing mean deviation, standard deviation, oscillation scores,
                           number of peaks, and peak score.
    """
    # Calculate absolute deviations from baseline
    divergence = np.abs(js_values - baseline)
    mean_divergence = np.mean(divergence)

    # Standard deviation of JS values (for overall fluctuation)
    std_dev = np.std(js_values)

    # Calculate first-order differences (consecutive changes)
    first_order_diffs = np.diff(js_values)
    mean_oscillation = np.mean(np.abs(first_order_diffs))

    # Calculate second-order differences (acceleration in change)
    second_order_diffs = np.diff(first_order_diffs)
    mean_second_order_change = np.mean(np.abs(second_order_diffs))

    # Compile results into a dictionary
    stability_metrics = {
        "Mean Divergence": mean_divergence,
        "Standard Deviation": std_dev,
        "Mean Oscillation": mean_oscillation,
        "Mean Second-order Change": mean_second_order_change,
        "Num Peaks": num_peaks,
        "Peak Score": peak_score
    }

    return stability_metrics


js_values = js_divergences
coherence_scores = compute_coherence_metrics(np.array(js_values), num_peaks=num_peaks, peak_score=peak_score, baseline=0)
print("Stability Scores (Unweighted):", coherence_scores)

weighted_js_values = weighted_js_divergences
weighted_coherence_scores = compute_coherence_metrics(np.array(weighted_js_values), num_peaks=num_peaks_weighted, peak_score=peak_score_weighted, baseline=0)
print("Weighted Stability Scores:", weighted_coherence_scores)


# --- COMPARISON FUNCTION UNWEIGHTED--- #
def compare_chunks_to_article(full_article_probs, chunk_results, num_topics):
    js_divergences = []
    for chunk in chunk_results:
        chunk_probs = chunk["prob_distribution"]
        js_divergence = float(jensenshannon(full_article_probs, chunk_probs))
        js_divergences.append(js_divergence)
    return js_divergences

def calculate_global_coherence(article_text, config, use_weighted=False):
    article_chunks = split_into_chunks(article_text, num_chunks=config["num_chunks"])
    full_article_probs = get_topic_probabilities_article(article_text, config["num_topics"])
    chunk_probabilities = get_topic_probabilities_chunks(article_chunks, config["num_topics"])

    if use_weighted:
        js_values = calculate_weighted_js_for_chunks(full_article_probs, chunk_probabilities, topics)
    else:
        js_values = compare_chunks_to_article(full_article_probs, chunk_probabilities, config["num_topics"])

    # Compute Peak metrics
    peaks, num_peaks, peak_heights, peak_score, _ = compute_peak_analysis(js_values, prominence_threshold=config["prominence_threshold"])

    return {
        "mean_js_divergence": np.mean(js_values),
        "std_js_divergence": np.std(js_values),
        "mean_oscillation": np.mean(np.abs(np.diff(js_values))),
        "mean_second_order_change": np.mean(np.abs(np.diff(np.diff(js_values)))),
        "peak_score": peak_score,
        "num_peaks": num_peaks
    }

real_sample = min_length_true_df.sample(n=200, random_state=config["random_seed"])
fake_sample = min_length_fake_df.sample(n=200, random_state=config["random_seed"])

def process_articles(sample_df, config, use_weighted=False):
    results = []
    for label, df in zip(["real", "fake"], sample_df):
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {label} news"):
            text = row['text']
            if pd.isna(text) or len(text.split()) < 150:
                continue

            # Calculate coherence metrics
            coherence_metrics = calculate_global_coherence(text, config, use_weighted=use_weighted)

            # Store the results
            coherence_metrics['label'] = label
            results.append(coherence_metrics)

    return pd.DataFrame(results)

# Process articles for unweighted JS divergence
results_unweighted = process_articles([real_sample, fake_sample], config, use_weighted=False)

# Process articles for weighted JS divergence
results_weighted = process_articles([real_sample, fake_sample], config, use_weighted=True)

# Combine both results
results_unweighted['analysis_type'] = 'Unweighted'
results_weighted['analysis_type'] = 'Weighted'
final_results_df = pd.concat([results_unweighted, results_weighted])

# Group by label and analysis type for final summary
final_summary = final_results_df.groupby(['label', 'analysis_type']).mean()
print(final_summary)

# --- Visualization with Real/Fake next to each other grouped by Unweighted/Weighted --- #
metrics = [
    'mean_js_divergence', 'std_js_divergence', 'mean_oscillation', 'mean_second_order_change',
    'peak_score', 'num_peaks'
]

# Create subplots for each metric
fig, axes = plt.subplots(3, 2, figsize=(12, 12))  # Adjust to 3x2 grid to fit all 6 metrics
fig.suptitle("Global Coherence and Peak Metrics Comparison: Real vs Fake News (Unweighted vs Weighted)", fontsize=16)

# Flatten axes for easier indexing
axes = axes.ravel()

# Plot each metric for real vs fake with grouped bars by unweighted/weighted
for i, metric in enumerate(metrics):
    # Create a grouped DataFrame with 'label' (real/fake) and 'analysis_type' (unweighted/weighted)
    sns.barplot(
        x='analysis_type', 
        y=metric, 
        hue='label', 
        data=final_results_df, 
        ax=axes[i], 
        palette='Set2'
    )
    
    axes[i].set_title(f"{metric.replace('_', ' ').title()}")
    axes[i].set_xlabel("Analysis Type")
    axes[i].set_ylabel(metric.replace('_', ' ').title())
    axes[i].legend(title="News Type", loc="upper right")

# Adjust layout and show plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
