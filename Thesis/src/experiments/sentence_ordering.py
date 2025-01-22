

# ---------------------------- WITH TRIALS -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from src.data.preprocessing import split_into_chunks
from src.utils.helper import (
    compute_js_values_for_chunks,
    compute_coherence_metrics,
    create_so_permutations,
)

def run_sentence_ordering_experiment_table(
    input_path,
    output_path,
    plot_path,
    num_permutations=20,
    sample_count=None,
    num_trials=3
):
    """
    Sentence ordering experiment that stores each article's original + permutations in a long-form CSV.
    Now supports multiple trials. For each (article, trial):
      1) We add a row for the original.
      2) Generate permutations, store each with metrics.
      3) Aggregate "accuracy" across trials for a final summary.

    Args:
        input_path      (str):  Path to the input dataset (CSV).
        output_path     (str):  Path to save the long-form CSV with all rows.
        plot_path       (str):  Path to save the accuracy plot.
        num_permutations(int):  Number of permutations to generate per trial.
        sample_count    (int):  If set, only process that many articles from the top of the file.
        num_trials      (int):  Number of trials to run per article (default=3).
    """
    # 1) Load data
    data = pd.read_csv(input_path)
    if sample_count:
        data = data.head(sample_count)
        print(f"[INFO] Only processing first {sample_count} articles for inspection.")

    model_path = r"C:\Thesis\MScThesis\Thesis\topic_models\wsj_topic_model"
    topic_model = BERTopic.load(model_path)

    # Define which metrics to compute
    metrics = [
        "Mean JS", "Std JS", "CV JS", "First Order Diff JS",
        "Peak Ratio","RMSE"]

    # Prepare counters for final overall accuracy
    overall_counters = {m: 0 for m in metrics}
    total_permutations = 0

    # DataFrame rows for storing each article/trial/permutation's data
    all_rows = []

    # 3) Iterate over articles
    for idx, row in data.iterrows():
        article_text = row["text"]

        # Skip invalid articles
        if not isinstance(article_text, str) or not article_text.strip():
            print(f"[WARN] Skipping invalid article at index {idx}.")
            continue

        # Original article -> topic probabilities
        _, original_probs = topic_model.transform([article_text])

        # Split into chunks
        original_chunks = split_into_chunks(article_text)
        if len(original_chunks) < 2:
            print(f"[WARN] Article {idx} has fewer than 2 chunks, skipping.")
            continue

        # Compute JS + metrics for the original
        orig_js_vals = compute_js_values_for_chunks(original_chunks, original_probs[0], topic_model)
        orig_metrics = compute_coherence_metrics(orig_js_vals)

        # 4) Multiple trials
        for trial_id in range(1, num_trials + 1):
            # Always record a row for the original (so each trial has a copy of it)
            row_dict = {
                "Article Index": idx,
                "Trial ID": trial_id,
                "Permutation ID": 0,
                "IsOriginal": True,
                "Text": article_text[:300] + ("..." if len(article_text) > 300 else "")
            }
            for m in metrics:
                row_dict[m] = orig_metrics[m]
            all_rows.append(row_dict)

            # Generate permutations for this trial
            permutations = create_so_permutations(article_text, num_permutations)
            # Prepare counters
            art_counters = {m: 0 for m in metrics}

            # Evaluate permutations
            for perm_id, perm_sentences in enumerate(permutations, start=1):
                perm_text = " ".join(perm_sentences)
                perm_chunks = split_into_chunks(perm_text)
                if len(perm_chunks) < 2:
                    continue

                perm_js_vals = compute_js_values_for_chunks(perm_chunks, original_probs[0], topic_model)
                perm_metrics = compute_coherence_metrics(perm_js_vals)

                # Compare to original
                for m in metrics:
                    if orig_metrics[m] < perm_metrics[m]:
                        art_counters[m] += 1

                # Store permutation info in the table
                row_dict = {
                    "Article Index": idx,
                    "Trial ID": trial_id,
                    "Permutation ID": perm_id,
                    "IsOriginal": False,
                    "Text": perm_text[:300] + ("..." if len(perm_text) > 300 else "")
                }
                for m in metrics:
                    row_dict[m] = perm_metrics[m]
                all_rows.append(row_dict)

            # After finishing the permutations in this trial, update counters
            for m in metrics:
                overall_counters[m] += art_counters[m]
            total_permutations += num_permutations

    # 5) Build final DataFrame and save
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(output_path, index=False)
    print(f"[INFO] Detailed results (original + permutations) saved to {output_path}")

    # 6) Compute overall accuracy
    #    - We'll define accuracy as the fraction of permutations that produce metric >= original
    #      across all articles & trials.
    overall_accuracy = {
        m: (overall_counters[m] / total_permutations) * 100
        for m in metrics
    }

    # 7) Print results
    print("\nOverall Accuracy Across All Processed Articles & Trials:")
    for m in metrics:
        print(f"  {m}: {overall_accuracy[m]:.2f}%")

    # 8) Plot final accuracy
    plot_overall_accuracy(overall_accuracy, plot_path)


def plot_overall_accuracy(accuracy_dict, plot_path):
    """
    Plots a simple bar chart of the overall accuracy for each metric.
    """
    plt.figure(figsize=(12, 6))
    metrics = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    plt.bar(metrics, accuracies, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Overall Accuracy: Sentence Ordering Task (Multi-Trial)")
    plt.xlabel("Metrics")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"[INFO] Plot saved to {plot_path}")


# Example usage:
if __name__ == "__main__":
    input_path = "C:/Thesis/MScThesis/data/wsj_cleaned_min_sent_length_6.csv"
    output_path = "C:/Thesis/MScThesis/gc_experiments_results/sentence_ordering/wsj_so_table_trials_final.csv"
    plot_path = "C:/Thesis/MScThesis/gc_experiments_results/plots/wsj_so_metrics_table.png"

    # Process only the first 3 articles for 3 trials, each with 20 permutations
    run_sentence_ordering_experiment_table(
        input_path=input_path,
        output_path=output_path,
        plot_path=plot_path,
        num_permutations=20,
        sample_count=None,     # or None for full dataset
        num_trials=3        # run 3 separate trials
    )
