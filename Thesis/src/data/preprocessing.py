import os
import re
from nltk.tokenize import sent_tokenize
import spacy
import ftfy
import pandas as pd


def preprocess_wsj_articles(file_paths, min_sentences=5):
    """
    Preprocess and filter WSJ articles based on a minimum sentence count.
    Args:
        file_paths (list): List of WSJ article file paths.
        min_sentences (int): Minimum number of sentences required to include an article.
    Returns:
        list: Filtered list of dictionaries containing article IDs, file paths, and sentences.
    """
    processed_wsj_articles = []

    for file_path in file_paths:
        try:
            # Open the file and try reading it with 'utf-8'
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # If 'utf-8' fails, try reading with 'latin-1'
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()

        # Remove ".START" and clean up formatting issues
        content = content.replace(".START", "")
        content = re.sub(r'\.\s*\.\s*\.', '', content)
        content = re.sub(r'(\d)\.\s*,', r'\1,', content)

        # Split content into sentences
        sentences = sent_tokenize(content)

        # Check if the article has at least the minimum number of sentences
        if len(sentences) >= min_sentences:
            # Extract the identifier from the file path
            article_id = os.path.basename(file_path).replace(".txt", "")
            # Store the identifier and content in the filtered list
            processed_wsj_articles.append({
                "id": article_id,
                "file_path": file_path,  # Store file path for later reference
                "sentences": sentences
            })

    return processed_wsj_articles


# Load SpaCy and the tokenizer

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")



def split_into_chunks(text, sentences_per_chunk=5, min_chunks=3):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    total_sentences = len(sentences)

    # --- Case 1: Very few total sentences ---
    if total_sentences < sentences_per_chunk * min_chunks:
        chunk_sizes = [total_sentences // min_chunks] * min_chunks
        for i in range(total_sentences % min_chunks):
            chunk_sizes[i] += 1
    else:
        # --- Case 2: Enough sentences for 'normal' chunks ---
        chunk_sizes = [sentences_per_chunk] * (total_sentences // sentences_per_chunk)
        remainder = total_sentences % sentences_per_chunk
        if remainder:
            chunk_sizes.append(remainder)

    # --- Merge a tiny last chunk if it's too small ---
    if chunk_sizes[-1] <= 3 and len(chunk_sizes) > 1:
        small_chunk = chunk_sizes.pop()
        for i in range(small_chunk):
            chunk_sizes[i % len(chunk_sizes)] += 1

    while len(chunk_sizes) < min_chunks:
        # Take one sentence from the last chunk, form a new chunk of size 1
        chunk_sizes[-1] -= 1
        chunk_sizes.append(1)

    if len(chunk_sizes) == min_chunks:
        sum_sentences = sum(chunk_sizes)  # should match total_sentences
        base = sum_sentences // min_chunks
        remainder = sum_sentences % min_chunks
        # Reset each chunk to the base size
        chunk_sizes = [base] * min_chunks
        # Distribute the remainder across the first few chunks
        for i in range(remainder):
            chunk_sizes[i] += 1

    # --- Finally, build the actual chunks in order ---
    chunks = []
    start_idx = 0
    for size in chunk_sizes:
        chunk_text = " ".join(sentences[start_idx : start_idx + size])
        chunks.append(chunk_text)
        start_idx += size

    return chunks


def preprocess_isot_articles(input_path, output_path, min_sentences=5, fix_encoding=True, remove_duplicates=True):
    """
    Preprocess ISOT articles: clean formatting issues, remove unnecessary text, fix encoding,
    and optionally remove duplicate articles.

    Args:
        input_path (str): Path to the raw dataset (Fake.csv).
        output_path (str): Path to save the cleaned dataset.
        min_sentences (int): Minimum number of sentences required to keep an article.
        fix_encoding (bool): Whether to fix encoding issues with ftfy.
        remove_duplicates (bool): Whether to drop duplicate articles.
    """
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(input_path)

    # Ensure the 'text' column exists
    if 'text' not in data.columns:
        raise ValueError("The input dataset must contain a 'text' column.")

    def clean_article(article, is_fake=True):
        """
        Clean a single article based on predefined rules.
        """
        # Fix encoding issues
        if fix_encoding:
            article = ftfy.fix_text(article)

        # Remove leading location or (Reuters) followed by a hyphen with spaces around it
        match = re.search(r' - ', article)
        if match and match.start() < 150:  # Location often appears at the start
            article = article[match.end():].lstrip()

        # For fake articles: truncate at the last period
        if is_fake:
            last_period_index = article.rfind(".")
            if last_period_index != -1:
                article = article[:last_period_index + 1]

        # Remove editor's notes
        article = re.sub(r"\(Editor’s note:.*?\)", "", article, flags=re.IGNORECASE)
        article = re.sub(r"\(Editor's note:.*?\)", "", article, flags=re.IGNORECASE)

        # Remove symbols, URLs, hashtags, mentions
        article = re.sub(r"[\*\-]", "", article)
        article = re.sub(r'http\S+|www\S+|ftp\S+', '', article)
        article = re.sub(r'@\w+', '', article)
        article = re.sub(r'#\w+', '', article)

        return article.strip()

    # Apply cleaning to all articles
    print("Cleaning articles...")
    data['text'] = data['text'].apply(lambda x: clean_article(str(x)))

    # Remove duplicates based on 'text' column
    if remove_duplicates:
        before = len(data)
        data = data.drop_duplicates(subset='text', keep='first')
        after = len(data)
        print(f"Removed {before - after} duplicate articles.")

    # Filter out articles with fewer than `min_sentences`
    print("Filtering short articles...")
    data['num_sentences'] = data['text'].apply(lambda x: len(re.split(r'\.|\?|!', x)))
    data = data[data['num_sentences'] >= min_sentences]
    data = data.drop(columns=['num_sentences'])  # Remove helper column

    # Save the cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")


# Input and Output Paths
input_file = "C:/Thesis/MScThesis/data/raw/Fake.csv"
output_file = "C:/Thesis/MScThesis/Thesis/src/data/fake_new_processed.csv"

# Run preprocessing
if __name__ == "__main__":
    preprocess_isot_articles(input_path=input_file, output_path=output_file, min_sentences=6)


