import os
import glob
import pandas as pd

def load_wsj_articles(wsj_path):
    """
    Load all WSJ article file paths from the provided directory and its subdirectories.
    Returns:
        list: List of file paths for all articles found.
    """
    # Use glob to find all files within the directory and its subdirectories
    all_files = glob.glob(f"{wsj_path}/**/*", recursive=True)
    
    # Return only files (not directories)
    return [file for file in all_files if os.path.isfile(file)]


def load_preprocessed_wsj_data(data_path="src/data/wsj_cleaned_min_sent_length_6.csv"):
    """
    Load the preprocessed WSJ dataset from a CSV file.
    Args:
        data_path (str): Path to the preprocessed dataset CSV file.
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed WSJ articles.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Ensure the file exists.")
    
    #print(f"Loading preprocessed WSJ data from {data_path}...")
    return pd.read_csv(data_path)

# Example usage
if __name__ == "__main__":
    wsj_data = load_preprocessed_wsj_data()
    print(f"Loaded {len(wsj_data)} articles.")
