import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import re
import spacy

# Load the SpaCy model and add the sentencizer if not already added
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# Function to clean the article text
def clean_text(text):
    """
    Cleans text by:
    - Removing excessive whitespaces, newlines, and tabs
    - Stripping leading and trailing spaces
    - Keeping the text in a consecutive format
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces, tabs, or newlines with a single space
    return text.strip()  # Strip leading and trailing spaces

# Function to count sentences in an article
def count_sentences(text):
    """
    Counts the number of sentences in the text using SpaCy sentencizer.
    """
    doc = nlp(text)
    return len(list(doc.sents))

# Load the full 20 Newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Create a DataFrame with the articles and their respective categories
data = {
    'article': [clean_text(article) for article in newsgroups_data.data],  # Clean the article text
    'category': [newsgroups_data.target_names[label] for label in newsgroups_data.target]  # Category name
}
df = pd.DataFrame(data)

# Remove NaN and empty articles (including whitespace-only)
df = df.dropna(subset=['article'])  # Remove NaN
df = df[df['article'].str.strip() != ""]  # Remove empty or whitespace-only articles

# Filter articles based on minimum sentence length (e.g., 3 sentences or more)
df['num_sentences'] = df['article'].apply(count_sentences)
df = df[df['num_sentences'] >= 3]  # Keep articles with 3 or more sentences

# Save the dataset to the specified folder with the name 'generaldata.csv'
output_path = r"C:\Thesis\MScThesis\data\generaldata.csv"
df.to_csv(output_path, index=False)

print(f"The dataset has been saved to: {output_path}")
print(f"Number of articles after filtering: {len(df)}")


# from sklearn.datasets import fetch_20newsgroups

# # Load the raw 20 Newsgroups dataset
# newsgroups_data = fetch_20newsgroups(subset='all', remove=())  # Do not remove headers, footers, or quotes

# # Inspect the total number of articles
# print(f"Total number of articles: {len(newsgroups_data.data)}")

# # Save all articles to a file for inspection
# output_path = r"C:\Thesis\MScThesis\data\raw_20newsgroups.txt"
# with open(output_path, 'w', encoding='utf-8') as f:
#     for i, article in enumerate(newsgroups_data.data):
#         f.write(f"--- Article {i + 1} ---\n")
#         f.write(article)
#         f.write("\n\n")

# print(f"All articles have been saved to: {output_path}")
