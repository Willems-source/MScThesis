# import pandas as pd
# import os
# import random
# from datetime import datetime
# from bertopic import BERTopic
# from src.data.preprocessing import split_into_chunks
# from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics

# # Load the pre-trained BERTopic model
# topic_model = BERTopic.load("fitted_topic_model")

# # Compute metrics for the entire corpus without alterations
# def compute_corpus_metrics(documents, topic_model, dataset_name, sentences_per_chunk=5, min_chunks=3, max_articles=None):
#     # Initialize global metrics container
#     corpus_metrics = {
#         "Mean JS": [], "Std JS": [], "CV JS": [],
#         "First Order Diff JS": [], "Second Order Diff JS": [],
#         "Num Peaks": [], "Peak Ratio": [], "RMSE": []
#     }

#     # Limit the number of articles analyzed
#     documents = random.sample(documents, max_articles) if len(documents) > max_articles else documents

#     for article_id, article_text in enumerate(documents):
#         # Get full article topic probabilities
#         article_topic, original_probs = topic_model.transform([article_text])

#         # Split original article into chunks
#         original_chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)
#         if len(original_chunks) == 0:
#             print(f"Skipping Article {article_id + 1}: No chunks generated")
#             continue
#         num_chunks = len(original_chunks)
#         original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)
#         if not original_js_values or pd.isna(original_js_values).any():
#             print(f"Skipping Article {article_id + 1}: Invalid JS values")
#             continue

#         # Compute metrics for the original article
#         original_metrics = compute_coherence_metrics(original_js_values)

#         # Add original metrics to the global container
#         for metric in ["Mean JS", "Std JS", "CV JS", "First Order Diff JS", "Second Order Diff JS", "Num Peaks", "Peak Ratio", "RMSE"]:
#             corpus_metrics[metric].append(original_metrics[metric])

#     # Compute average metrics across the entire corpus
#     total_articles = len(corpus_metrics["Mean JS"])  # Use processed articles count
#     corpus_summary = {
#         "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Dataset": dataset_name,
#         "Sentences Per Chunk": sentences_per_chunk,
#         "Total Articles": total_articles,
#         "Avg Mean JS": sum(corpus_metrics["Mean JS"]) / total_articles,
#         "Avg Std JS": sum(corpus_metrics["Std JS"]) / total_articles,
#         "Avg CV JS": (sum(corpus_metrics["Std JS"]) / total_articles) / (sum(corpus_metrics["Mean JS"]) / total_articles),
#         "Avg First Order Diff JS": sum(corpus_metrics["First Order Diff JS"]) / total_articles,
#         "Avg Second Order Diff JS": sum(corpus_metrics["Second Order Diff JS"]) / total_articles,
#         "Avg Num Peaks": sum(corpus_metrics["Num Peaks"]) / total_articles,
#         "Avg Peak Ratio": sum(corpus_metrics["Peak Ratio"]) / total_articles,
#         "Avg RMSE": sum(corpus_metrics["RMSE"]) / total_articles,
#     }

#     return corpus_summary


# # Process dataset and append results
# def process_dataset(data_path, dataset_name, output_path, sentences_per_chunk=5, min_chunks=3, max_articles=50):
#     # Load dataset
#     corpus = pd.read_csv(data_path)
#     documents = corpus["text"].tolist()

#     # Compute metrics
#     corpus_summary = compute_corpus_metrics(
#         documents, 
#         topic_model, 
#         dataset_name=dataset_name, 
#         sentences_per_chunk=sentences_per_chunk, 
#         min_chunks=min_chunks,
#         max_articles=max_articles
#     )

#     # Append results to the CSV file
#     if pd.io.common.file_exists(output_path):
#         pd.DataFrame([corpus_summary]).to_csv(output_path, mode="a", header=False, index=False)
#     else:
#         pd.DataFrame([corpus_summary]).to_csv(output_path, mode="w", header=True, index=False)

#     print(f"Results for {dataset_name} appended to {output_path}.")


# # Process multiple LLM-generated datasets
# def process_llm_generated_datasets(folder_path, output_path, sentences_per_chunk=5, min_chunks=3, max_articles=None):
#     # Loop through all files in the directory
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".csv"):  # Ensure we only process CSV files
#             dataset_name = file_name.replace(".csv", "")  # Use file name without extension as dataset name
#             data_path = os.path.join(folder_path, file_name)
#             print(f"Processing dataset: {dataset_name}")

#             process_dataset(
#                 data_path=data_path,
#                 dataset_name=dataset_name,
#                 output_path=output_path,
#                 sentences_per_chunk=sentences_per_chunk,
#                 min_chunks=min_chunks,
#                 max_articles=max_articles
#             )


# # Main function
# if __name__ == "__main__":
#     output_path = "C:/Thesis/MScThesis/gc_experiments_results/corpus_summary_metrics.csv"

#     # # Real news
#     # true_data_path = "C:/Thesis/MScThesis/Thesis/src/data/true_processed.csv"
#     # process_dataset(
#     #     data_path=true_data_path,
#     #     dataset_name="Real news",
#     #     output_path=output_path,
#     #     sentences_per_chunk=5,
#     #     min_chunks=3,
#     #     max_articles=2000
#     # )

#     # # Fake news
#     # fake_data_path = "C:/Thesis/MScThesis/Thesis/src/data/fake_processed.csv"
#     # process_dataset(
#     #     data_path=fake_data_path,
#     #     dataset_name="Fake news",
#     #     output_path=output_path,
#     #     sentences_per_chunk=5,
#     #     min_chunks=3,
#     #     max_articles=2000
#     # )

#     # LLM-news
#     llm_folder_path = "C:/Thesis/MScThesis/data/raw/mage"  
#     process_llm_generated_datasets(
#         folder_path=llm_folder_path,
#         output_path=output_path,
#         sentences_per_chunk=5,
#         min_chunks=3,
#         max_articles=10  
#     )


# AGGREGATING DIFFERENT LLM DATASET METRICS
import pandas as pd
import os
import random
from datetime import datetime
from bertopic import BERTopic
from src.data.preprocessing import split_into_chunks
from src.utils.helper import compute_js_values_for_chunks, compute_coherence_metrics

# Load the pre-trained BERTopic model
topic_model = BERTopic.load("fitted_topic_model")

# Compute metrics for the entire corpus without alterations
def compute_corpus_metrics(documents, topic_model, dataset_name, sentences_per_chunk=5, min_chunks=3, max_articles=None):
    # Initialize global metrics container
    corpus_metrics = {
        "Mean JS": [], "Std JS": [], "CV JS": [],
        "First Order Diff JS": [], "Second Order Diff JS": [],
        "Num Peaks": [], "Peak Ratio": [], "RMSE": []
    }

    # Limit the number of articles analyzed
    #documents = random.sample(documents, max_articles) if len(documents) > max_articles else documents

    for article_id, article_text in enumerate(documents):
        # Get full article topic probabilities
        article_topic, original_probs = topic_model.transform([article_text])

        # Split original article into chunks
        original_chunks = split_into_chunks(article_text, sentences_per_chunk, min_chunks)
        if len(original_chunks) == 0:
            print(f"Skipping Article {article_id + 1}: No chunks generated")
            continue
        num_chunks = len(original_chunks)
        original_js_values = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)
        if not original_js_values or pd.isna(original_js_values).any():
            print(f"Skipping Article {article_id + 1}: Invalid JS values")
            continue

        # Compute metrics for the original article
        original_metrics = compute_coherence_metrics(original_js_values)

        # Add original metrics to the global container
        for metric in ["Mean JS", "Std JS", "CV JS", "First Order Diff JS", "Second Order Diff JS", "Num Peaks", "Peak Ratio", "RMSE"]:
            corpus_metrics[metric].append(original_metrics[metric])

    # Compute average metrics across the entire corpus
    total_articles = len(corpus_metrics["Mean JS"])  # Use processed articles count
    corpus_summary = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset": dataset_name,
        "Sentences Per Chunk": sentences_per_chunk,
        "Total Articles": total_articles,
        "Avg Mean JS": sum(corpus_metrics["Mean JS"]) / total_articles,
        "Avg Std JS": sum(corpus_metrics["Std JS"]) / total_articles,
        "Avg CV JS": (sum(corpus_metrics["Std JS"]) / total_articles) / (sum(corpus_metrics["Mean JS"]) / total_articles),
        "Avg First Order Diff JS": sum(corpus_metrics["First Order Diff JS"]) / total_articles,
        "Avg Second Order Diff JS": sum(corpus_metrics["Second Order Diff JS"]) / total_articles,
        "Avg Num Peaks": sum(corpus_metrics["Num Peaks"]) / total_articles,
        "Avg Peak Ratio": sum(corpus_metrics["Peak Ratio"]) / total_articles,
        "Avg RMSE": sum(corpus_metrics["RMSE"]) / total_articles,
    }

    return corpus_summary


# Process dataset and append results
def process_dataset(data_path, dataset_name, output_path, sentences_per_chunk=5, min_chunks=3, max_articles=None):
    # Load dataset
    corpus = pd.read_csv(data_path)
    documents = corpus["text"].tolist()

    # Compute metrics
    corpus_summary = compute_corpus_metrics(
        documents, 
        topic_model, 
        dataset_name=dataset_name, 
        sentences_per_chunk=sentences_per_chunk, 
        min_chunks=min_chunks,
        max_articles=max_articles
    )

    # Append results to the CSV file
    if pd.io.common.file_exists(output_path):
        pd.DataFrame([corpus_summary]).to_csv(output_path, mode="a", header=False, index=False)
    else:
        pd.DataFrame([corpus_summary]).to_csv(output_path, mode="w", header=True, index=False)

    print(f"Results for {dataset_name} appended to {output_path}.")


# Process multiple LLM-generated datasets
def process_llm_generated_datasets(folder_path, output_path, sentences_per_chunk=5, min_chunks=3, max_articles=None):
    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):  # Ensure we only process CSV files
            dataset_name = file_name.replace(".csv", "")  # Use file name without extension as dataset name
            data_path = os.path.join(folder_path, file_name)
            print(f"Processing dataset: {dataset_name}")

            process_dataset(
                data_path=data_path,
                dataset_name=dataset_name,
                output_path=output_path,
                sentences_per_chunk=sentences_per_chunk,
                min_chunks=min_chunks,
                max_articles=max_articles
            )


# Main function
if __name__ == "__main__":
    output_path = "C:/Thesis/MScThesis/gc_experiments_results/corpus_summary_metrics.csv"

    # # Real news
    # true_data_path = "C:/Thesis/MScThesis/Thesis/src/data/true_processed.csv"
    # process_dataset(
    #     data_path=true_data_path,
    #     dataset_name="Real news",
    #     output_path=output_path,
    #     sentences_per_chunk=5,
    #     min_chunks=3,
    #     max_articles=2000
    # )

    # # Fake news
    # fake_data_path = "C:/Thesis/MScThesis/Thesis/src/data/fake_processed.csv"
    # process_dataset(
    #     data_path=fake_data_path,
    #     dataset_name="Fake news",
    #     output_path=output_path,
    #     sentences_per_chunk=5,
    #     min_chunks=3,
    #     max_articles=2000
    # )

    # # LLM-news
    # llm_folder_path = "C:/Thesis/MScThesis/data/raw/mage_processed"  
    # process_llm_generated_datasets(
    #     folder_path=llm_folder_path,
    #     output_path=output_path,
    #     sentences_per_chunk=5,
    #     min_chunks=3,
    #     max_articles=60  
    # )


    llm_folder_path_aggr = "C:/Thesis/MScThesis/data/raw/xsum_gpt_llama.csv"
    process_dataset(
        data_path=llm_folder_path_aggr,
        dataset_name="LLM Aggregated",
        output_path=output_path,
        sentences_per_chunk=5,
        min_chunks=3,
        #max_articles=60
    )

    

