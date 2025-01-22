import os
import json
import pandas as pd

# # Define the paths
# json_folder = "C:/Thesis/MScThesis/data/raw/HealthStory"  # Folder containing the individual news JSON files
# reviews_file = "C:/Thesis/MScThesis/data/HealthStory.json"  # Path to the reviews JSON file
# output_csv = "C:/Thesis/MScThesis/data/HealthStory_dataset.csv"  # Output CSV file path

# # Load the reviews JSON file
# with open(reviews_file, 'r', encoding='utf-8') as f:
#     reviews_data = json.load(f)

# # Extract the news_id and rating from the reviews JSON
# ratings = {review['news_id']: review['rating'] for review in reviews_data}

# # Initialize an empty list to store the data
# data = []

# # Iterate through the individual news JSON files
# for filename in os.listdir(json_folder):
#     if filename.endswith(".json"):
#         filepath = os.path.join(json_folder, filename)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             news_content = json.load(f)
#             news_id = os.path.splitext(filename)[0]  # Extract the ID from the file name
#             text = news_content.get('text', '')  # Get the text field
#             rating = ratings.get(news_id, None)  # Match the rating using news_id
#             if text and rating is not None:  # Ensure both text and rating exist
#                 data.append({"news_id": news_id, "text": text, "rating": rating})

# # Convert the list to a pandas DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to a CSV file
# df.to_csv(output_csv, index=False, encoding='utf-8')

# print(f"Data successfully extracted and saved to {output_csv}")

# import pandas as pd
# import re
# import unicodedata

# def clean_text(text):
#     """
#     Cleans text by:
#     - Fixing encoding issues (including malformed characters like â€)
#     - Normalizing malformed characters
#     - Removing patterns like ()-, ()-- and all text preceding them
#     - Removing specific phrases like 'En EspaÃ±ol' along with trailing whitespace
#     - Retaining other () patterns
#     - Removing excess whitespace
#     - Making sentences consecutive without enters in between
#     """
#     try:
#         # Decode text and handle encoding issues
#         clean_text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
#     except Exception:
#         # If decoding fails, fallback to returning the original text
#         clean_text = text
    
#     # Normalize text to replace malformed characters (e.g., â€™ -> ’)
#     clean_text = unicodedata.normalize("NFKC", clean_text)
    
#     # Replace malformed characters like â€™, â€œ, â€
#     clean_text = re.sub(r'[â€]', "'", clean_text)  # Replace common malformed characters
#     clean_text = re.sub(r'[“”]', '"', clean_text)  # Replace fancy quotes with standard double quotes
#     clean_text = re.sub(r'[‘’]', "'", clean_text)  # Replace fancy apostrophes with standard single quotes
    
#     # Explicitly remove all instances of 'En EspaÃ±ol'
#     clean_text = re.sub(r'\bEn EspaÃ±ol\b', '', clean_text, flags=re.IGNORECASE)
    
#     # Remove patterns like ()-, ()-- including all text preceding them
#     clean_text = re.sub(r'.*?\([^)]+\)\s*[-]{1,2}', '', clean_text)
    
#     # Replace multiple spaces/newlines with a single space
#     clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
#     return clean_text

# def assign_label(rating):
#     """
#     Assigns a label based on the rating:
#     - 'real' if rating is 4 or 5
#     - 'fake' if rating is 1 or 2
#     - 'none' if rating is 3
#     """
#     if rating in [3, 4, 5]:
#         return 'real'
#     elif rating in [0, 1, 2]:
#         return 'fake'
#     else:
#         return None

# # Load the dataset
# file_path = "C:/Thesis/MScThesis/data/healthstory_dataset.csv"  # Update the path to your dataset
# data = pd.read_csv(file_path)

# # Ensure the column names are consistent
# if "text" not in data.columns or "rating" not in data.columns:
#     raise KeyError("The dataset must contain 'text' and 'rating' columns.")

# # Clean the text column for encoding issues and formatting
# data["cleaned_text"] = data["text"].apply(lambda x: clean_text(x) if isinstance(x, str) else x)

# # Add a 'label' column based on the 'rating'
# data["label"] = data["rating"].apply(assign_label)

# # Save the cleaned dataset to a new CSV file
# output_path = "C:/Thesis/MScThesis/data/cleaned_labeled_healthstory_dataset.csv"
# data.to_csv(output_path, index=False)

# print(f"Cleaned and labeled dataset saved to: {output_path}")

# # Define the paths
# json_folder = "C:/Thesis/MScThesis/data/raw/HealthRelease"  # Folder containing the individual news JSON files
# reviews_file = "C:/Thesis/MScThesis/data/HealthRelease.json"  # Path to the reviews JSON file
# output_csv = "C:/Thesis/MScThesis/data/HealthRelease_dataset.csv"  # Output CSV file path

# # Load the reviews JSON file
# with open(reviews_file, 'r', encoding='utf-8') as f:
#     reviews_data = json.load(f)

# # Extract the news_id and rating from the reviews JSON
# ratings = {review['news_id']: review['rating'] for review in reviews_data}

# # Initialize an empty list to store the data
# data = []

# # Iterate through the individual news JSON files
# for filename in os.listdir(json_folder):
#     if filename.endswith(".json"):
#         filepath = os.path.join(json_folder, filename)
#         with open(filepath, 'r', encoding='utf-8') as f:
#             news_content = json.load(f)
#             news_id = os.path.splitext(filename)[0]  # Extract the ID from the file name
#             text = news_content.get('text', '')  # Get the text field
#             rating = ratings.get(news_id, None)  # Match the rating using news_id
#             if text and rating is not None:  # Ensure both text and rating exist
#                 data.append({"news_id": news_id, "text": text, "rating": rating})

# # Convert the list to a pandas DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to a CSV file
# df.to_csv(output_csv, index=False, encoding='utf-8')

# print(f"Data successfully extracted and saved to {output_csv}")


import pandas as pd
import re
import unicodedata
from ftfy import fix_text

def clean_text(text):
    """
    Cleans text by:
    - Fixing encoding issues (including malformed characters like â€)
    - Normalizing malformed characters
    - Replacing instances of 'â€', 'â€“', 'â€”' with proper equivalents or removing them
    - Removing all text preceding '--' (if it occurs within the first 150 characters)
    - Removing patterns like ()-, ()-- and all text preceding them
    - Removing specific phrases like 'En EspaÃ±ol' along with trailing whitespace
    - Retaining other () patterns
    - Removing excess whitespace
    - Making sentences consecutive without enters in between
    """
    try:
        # Fix encoding issues
        clean_text = fix_text(text)
    except Exception:
        # If fixing fails, fallback to returning the original text
        clean_text = text

    # Normalize text to replace malformed characters (e.g., â€™ -> ’)
    clean_text = unicodedata.normalize("NFKC", clean_text)

    # Replace specific malformed characters and sequences
    clean_text = re.sub(r'â€”', '—', clean_text)  # Replace 'â€”' with a long dash
    clean_text = re.sub(r'â€“', '–', clean_text)  # Replace 'â€“' with a standard dash
    clean_text = re.sub(r'â€¦', '...', clean_text)  # Replace 'â€¦' with ellipsis
    clean_text = re.sub(r'â€˜', "'", clean_text)  # Replace opening single quote
    clean_text = re.sub(r'â€™', "'", clean_text)  # Replace closing single quote
    clean_text = re.sub(r'â€œ', '"', clean_text)  # Replace opening double quote
    clean_text = re.sub(r'â€�', '"', clean_text)  # Replace closing double quote
    clean_text = re.sub(r'Â°', '°', clean_text)  # Replace degree symbol
    clean_text = re.sub(r'Â', '', clean_text)  # Remove stray 'Â' characters

    # Remove all text preceding '--' only if it occurs in the first 150 characters
    clean_text = re.sub(r'^(.{0,150})--', '', clean_text, flags=re.DOTALL)

    # Explicitly remove all instances of 'En EspaÃ±ol'
    clean_text = re.sub(r'\bEn EspaÃ±ol\b', '', clean_text, flags=re.IGNORECASE)

    # Remove patterns like ()-, ()-- including all text preceding them
    clean_text = re.sub(r'.*?\([^)]+\)\s*[-]{1,2}', '', clean_text)

    # Replace multiple spaces/newlines with a single space
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text

def assign_label(rating):
    """
    Assigns a label based on the rating:
    - 'real' if rating is 4 or 5
    - 'fake' if rating is 1 or 2
    - 'none' if rating is 3
    """
    if rating in [3, 4, 5]:
        return 'real'
    elif rating in [0, 1, 2]:
        return 'fake'
    else:
        return None

# Load the dataset
file_path = "C:/Thesis/MScThesis/data/healthrelease_dataset.csv"  # Update the path to your dataset
data = pd.read_csv(file_path)

# Ensure the column names are consistent
if "text" not in data.columns or "rating" not in data.columns:
    raise KeyError("The dataset must contain 'text' and 'rating' columns.")

# Clean the text column for encoding issues and formatting
data["cleaned_text"] = data["text"].apply(lambda x: clean_text(x) if isinstance(x, str) else x)

# Add a 'label' column based on the 'rating'
data["label"] = data["rating"].apply(assign_label)

# Save the cleaned dataset to a new CSV file
output_path = "C:/Thesis/MScThesis/data/new_cleaned_labeled_healthrelease_dataset.csv"
data.to_csv(output_path, index=False)

print(f"Cleaned and labeled dataset saved to: {output_path}")




