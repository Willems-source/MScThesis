import random
from nltk.tokenize import sent_tokenize
import torch
import numpy as np
from scipy.spatial import distance
import random
import pandas as pd
import os
import pickle  # To save and load the sentence pool
from bertopic import BERTopic
from src.data.preprocessing import split_into_chunks
import spacy

def create_so_permutations(article_text, num_permutations=20):
    sentences = sent_tokenize(article_text)
    permutations = []
    while len(permutations) < num_permutations:
        permuted_sentences = sentences.copy()
        random.shuffle(permuted_sentences)
        if permuted_sentences != sentences and permuted_sentences not in permutations:
            permutations.append(permuted_sentences)
    return permutations

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")


def split_into_chunks(text, sentences_per_chunk=5, min_chunks=3):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    total_sentences = len(sentences)

    # Adjust chunk sizes for short articles
    if total_sentences < sentences_per_chunk * min_chunks:
        chunk_sizes = [total_sentences // min_chunks] * min_chunks
        for i in range(total_sentences % min_chunks):
            chunk_sizes[i] += 1
    else:
        chunk_sizes = [sentences_per_chunk] * (total_sentences // sentences_per_chunk)
        remainder = total_sentences % sentences_per_chunk
        if remainder:
            chunk_sizes.append(remainder)

    # Redistribute small chunks only if it won't violate the min_chunks constraint
    if chunk_sizes[-1] <= 3 and len(chunk_sizes) > min_chunks:  # Only adjust if it meets the criteria
        small_chunk = chunk_sizes.pop()  # Remove the small chunk
        for i in range(small_chunk):  # Distribute its sentences to previous chunks
            chunk_sizes[i % len(chunk_sizes)] += 1

    # If the number of chunks falls below min_chunks, adjust to enforce min_chunks
    while len(chunk_sizes) < min_chunks:
        chunk_sizes[-1] -= 1  # Take one sentence from the last chunk
        chunk_sizes.append(1)  # Create a new chunk with 1 sentence

    # Create the chunks
    chunks = []
    start_idx = 0
    for size in chunk_sizes:
        chunks.append(" ".join(sentences[start_idx:start_idx + size]))
        start_idx += size

    return chunks





def get_topic_probabilities(text, tokenizer, model):
    if not text.strip():
        raise ValueError("Input text is empty.")
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    except Exception as e:
        raise RuntimeError(f"Error during model inference: {e}")
    return probabilities


def calculate_js_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    Args:
        p (array-like): First probability distribution (full article)
        q (array-like): Second probability distribution (chunks)
    Returns:
        float: Jensen-Shannon divergence.
    """
    p, q = np.array(p) / np.sum(p), np.array(q) / np.sum(q)
    return distance.jensenshannon(p, q)


def compute_js_values_for_chunks(chunks, original_probs, model):
    js_values = []
    for chunk in chunks:
        chunk_probs = model.transform([chunk])[1][0]  # Get chunk probabilities
        js_value = calculate_js_divergence(original_probs, chunk_probs)
        if not np.isnan(js_value):  # Skip invalid JS values
            js_values.append(js_value)
    return js_values


def compute_coherence_metrics(js_values, peak_threshold = 0.15, peak_amplification_epsilon=1e-6, weights = (0.5, 0.3, 0.2), gamma=0.1, alpha=0.1):
    w1, w2, w3 = weights
    num_chunks = len(js_values)
    mean_js = np.mean(js_values)
    std_js = np.std(js_values)
    first_order_diff_js = np.mean(np.abs(np.diff(js_values)))
    #second_order_diff_js = np.mean(np.abs(np.diff(np.diff(js_values))))
    num_peaks = sum(
        (js_values[i] > js_values[i - 1] + peak_threshold and 
         js_values[i] > js_values[i + 1] + peak_threshold)
        for i in range(1, num_chunks - 1)
    )
    rmse = np.sqrt(max(0, np.mean(np.square(js_values))))
    peak_ratio = num_peaks / num_chunks if num_chunks > 0 else 0
    cv_js = std_js / mean_js if mean_js != 0 else 0  # Coefficient of Variation
    max_js = np.max(js_values)
    min_js = np.min(js_values)
    max_difference = max_js - min_js
    # # --- New Metrics ---
    # # Adjusted CV with Peak Amplification
    # acvp = cv_js * (peak_ratio + peak_amplification_epsilon)
    # # Weighted JS
    # weighted_js = w1 * mean_js + w2 * std_js
    # # Log-Weighted CV
    # log_wcv = cv_js * np.log(1 + gamma * mean_js)
    # weighted_mean_std_oscillation = w1 * mean_js + w2 * std_js * (1+w3*first_order_diff_js)

    # # COMBINED MEASURE
    # combined_js_measure_raw = w1*mean_js + w2*first_order_diff_js + w3*std_js
    # cv_penalty = alpha * cv_js if cv_js > 1 else 0 # ARBITRARY THRESHOLD --> to CHECK
    # combined_js_measure_raw += cv_penalty
    # combined_js_measure = 1 / (1 + np.exp(-combined_js_measure_raw))


    return {
        "Mean JS": mean_js,
        "Std JS": std_js,
        "CV JS": cv_js,
        "First Order Diff JS": first_order_diff_js,
        #"Second Order Diff JS": second_order_diff_js,
        "Num Peaks": num_peaks,
        "Peak Ratio": peak_ratio,
        "RMSE": rmse,
        "Max JS": max_js,
        "Min JS": min_js,
        "Max Difference": max_difference,
        # "Adjusted CV with Peak Amplification": acvp,
        # "Weighted JS": weighted_js,
        # "Log-Weighted CV": log_wcv,
        # "Measure RW": weighted_mean_std_oscillation,
        # "Combined JS Measure": combined_js_measure
    }


def get_unrelated_sentence(sentence_pool_df, article_topic, used_sentences):
    """
    Retrieve an unrelated sentence from the pool, avoiding sentences related to the article's topic.

    Args:
        sentence_pool_df (pd.DataFrame): DataFrame with columns 'Topic' and 'Sentence'.
        article_topic (int): Topic of the current article.
        used_sentences (set): Set of already used sentences to avoid repetition.

    Returns:
        str: An unrelated sentence.
    """
    unrelated_sentences = sentence_pool_df[sentence_pool_df["Topic"] != article_topic]["Sentence"].tolist()
    unrelated_sentence = None

    while unrelated_sentence is None:
        candidate_sentence = random.choice(unrelated_sentences)
        if candidate_sentence not in used_sentences:
            unrelated_sentence = candidate_sentence
            used_sentences.add(unrelated_sentence)
            # Clear used sentences if the pool is exhausted
            if len(used_sentences) >= len(unrelated_sentences):
                used_sentences.clear()

    return unrelated_sentence





def load_sentence_pool(file_path):
    """Load the sentence pool from the specified CSV file."""
    return pd.read_csv(file_path)


def create_topic_based_permutations(article_text, article_id, article_topic, sentence_pool_df, 
                                    num_permutations=20, sentences_per_chunk=5, 
                                    chunks_affected=0.25, sentences_to_replace_per_chunk="all"):
    """
    Create permutations of an article by replacing sentences or chunks.
    Affects a percentage of chunks, ensuring at least 1 chunk is always affected.
    """
    # Split the article into chunks
    original_chunks = split_into_chunks(article_text, sentences_per_chunk)
    num_chunks = len(original_chunks)
    permutations = []

    used_sentences = set()  # Track unrelated sentences used
    all_affected_chunks = []  # Collect affected chunks for all permutations
    perm_idx = 0

    while perm_idx < num_permutations:
        # Start with a clean copy of the original chunks
        permuted_chunks = list(original_chunks)  # Copy all chunks to ensure integrity

        # Compute the number of chunks to affect based on percentage
        if chunks_affected == 1:  # All chunks are affected
            affected_chunks = list(range(num_chunks))
        elif 0 < chunks_affected < 1:  # Affect a percentage of chunks
            num_chunks_to_affect = max(1, round(num_chunks * chunks_affected))  # At least 1 chunk
            affected_chunks = random.sample(range(num_chunks), num_chunks_to_affect)
        else:
            raise ValueError("Invalid value for chunks_affected. Must be between 0 and 1.")

        # Store affected chunks for this permutation
        all_affected_chunks.append(affected_chunks)

        # Modify only the selected chunks
        for chunk_idx in affected_chunks:
            if sentences_to_replace_per_chunk == "all":
                permuted_chunks[chunk_idx] = ""  # Remove all content from the chunk
                # Completely clear the text and replace with new sentences
                unrelated_sentences = [
                    get_unrelated_sentence(sentence_pool_df, article_topic, used_sentences)
                    for _ in range(sentences_per_chunk)
                ]
                permuted_chunks[chunk_idx] = " ".join(unrelated_sentences)  # Full replacement
            else:
                # Partially replace specified number of sentences
                chunk_sentences = sent_tokenize(permuted_chunks[chunk_idx])  # Tokenize sentences in the chunk
                num_sentences_to_replace = min(sentences_to_replace_per_chunk, len(chunk_sentences))
                indices_to_replace = random.sample(range(len(chunk_sentences)), num_sentences_to_replace)
                for idx in indices_to_replace:
                    unrelated_sentence = get_unrelated_sentence(sentence_pool_df, article_topic, used_sentences)
                    chunk_sentences[idx] = unrelated_sentence
                # Update the chunk with modified sentences
                permuted_chunks[chunk_idx] = " ".join(chunk_sentences)

        # Force unaffected chunks to match the original exactly if sentencizer causes spilling
        for idx in range(num_chunks):
            if idx not in affected_chunks:
                permuted_chunks[idx] = original_chunks[idx]  # Restore original chunks exactly

        # Rejoin the permuted chunks into text (to prevent chunk splitting issues)
        permuted_text = " ".join(permuted_chunks)

        # Add the permutation to the results
        permutations.append({
            "Permutation_Index": perm_idx + 1,
            "Text": permuted_text  # Store the full article text as a single string
        })
        perm_idx += 1

    return permutations, all_affected_chunks  # Return all affected chunks as a list



# Load the new sentence pool
sentence_pool_path = "C:/Thesis/MScThesis/Thesis/src/data/full_sentence_pool_tpds.csv"
sentence_pool_df = load_sentence_pool(sentence_pool_path)


# Select the replacement sentence with maximal JS divergence
def get_max_js_divergence_sentence(sentence_pool_df, sentence_tpd, used_sentences):
    """
    Finds the sentence in the pool with the maximum JS divergence compared to the given TPD.
    Args:
        sentence_pool_df (pd.DataFrame): DataFrame containing 'Sentence' and 'TPD' columns.
        sentence_tpd (list): TPD of the sentence to compare against.
        used_sentences (set): Sentences already used in the replacements.

    Returns:
        tuple: Best sentence and its JS divergence.
    """
    if not isinstance(sentence_pool_df, pd.DataFrame):
        raise ValueError("sentence_pool_df must be a pandas DataFrame with 'Sentence' and 'TPD' columns.")

    max_js = -1
    best_sentence = None

    for _, row in sentence_pool_df.iterrows():
        # Ensure the sentence hasn't been used already
        if row["Sentence"] in used_sentences:
            continue

        # Calculate JS divergence
        try:
            js_divergence = calculate_js_divergence(sentence_tpd, eval(row["TPD"]))  # Assuming TPD is stored as a stringified list
        except Exception as e:
            print(f"Error evaluating TPD for sentence: {row['Sentence']}. Skipping.")
            continue

        # Find the sentence with the highest JS divergence
        if js_divergence > max_js:
            max_js = js_divergence
            best_sentence = row["Sentence"]

    return best_sentence, max_js
