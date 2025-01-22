from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
import os
import openai

data_path = "src/data/wsj_cleaned_min_sent_length_6.csv"
isot_data_path = "C:/Thesis/MScThesis/Thesis/src/data/new_combined_real_fake.csv"
general_data_path = "C:/Thesis/MScThesis/data/generaldata.csv"
health_data_path = "C:/Thesis/MScThesis/data/FakeHealthdata.csv"
llm_data_path = "C:/Thesis/MScThesis/gc_experiments_results/dataset_comparison/llm_analysis/llm_human_metrics_general_50_topic_model.csv"
df = pd.read_csv(data_path)
isot_df = pd.read_csv(isot_data_path)
general_df = pd.read_csv(general_data_path)
health_df = pd.read_csv(health_data_path)
llm_df = pd.read_csv(llm_data_path)

# print("First Article:")
# print(df.loc[0, 'text'])  # Access the 'text' column of the first row
documents = df['text'].tolist()
isot_documents = isot_df['text'].tolist()
general_documents = general_df['article'].tolist()
health_documents = health_df['cleaned_text'].tolist()
llm_documents = llm_df['text'].tolist()

# Step 1 - Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state = 42)

# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")

# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Step 6 - (Optional) Fine-tune topic representations with 
# a `bertopic.representation` model
#client = openai.OpenAI(api_key="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")
representation_model = KeyBERTInspired()
#representation_model = OpenAI(client, model="gpt-4o")

# All steps together
topic_model = BERTopic(
  embedding_model=embedding_model,          # Step 1 - Extract embeddings
  umap_model=umap_model,                    # Step 2 - Reduce dimensionality
  hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
  vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
  ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
  representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic representations
  calculate_probabilities=True               # Calculate topic probabilities
)

# # topics, probs = topic_model.fit_transform(documents)
# # topic_model.save("fitted_topic_model", save_embedding_model=True)

# output_dir = r"C:/Thesis/MScThesis/Thesis/topic_models"
# os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

# topics, probs = topic_model.fit_transform(general_documents)
# reduced_topic_model = topic_model.reduce_topics(isot_documents, topics, probs, nr_topics=50)
# nr_topics = 50 

# topic_info = reduced_topic_model.get_topic_info()

# # Dynamic output filename based on topic count
# output_file_name = f"isot_{nr_topics}_topics_extracted.csv"
# output_path = os.path.join(output_dir, output_file_name)

# # Save the extracted topics to CSV
# topic_info.to_csv(output_path, index=False)
# print(f"Topics saved to {output_path}")

##2812 FOR HEALTH
output_dir = r"C:/Thesis/MScThesis/Thesis/topic_models"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

topics, probs = topic_model.fit_transform(llm_documents)
nr_topics = 50  # Set the desired number of topics dynamically

# Reduce the topic model
reduced_topic_model = topic_model.reduce_topics(llm_documents, nr_topics=nr_topics)

# Extract topic information
topic_info = reduced_topic_model.get_topic_info()

# Dynamic filenames
# topic_info_file = f"isot_{nr_topics}_topics_extracted.csv"  # CSV file for topic info --> FOR KEYBERTINSPIRED
# model_file = f"isot_{nr_topics}_topic_model"  # File for saving the reduced topic model --> FOR KEYBERTINSPIRED
topic_info_file = f"llm_{nr_topics}_topics_extracted.csv"  # CSV file for topic info
model_file = f"llm_{nr_topics}_topic_model"  # File for saving the reduced topic model

# Save topic info to CSV
topic_info_path = os.path.join(output_dir, topic_info_file)
topic_info.to_csv(topic_info_path, index=False)
print(f"Topic information saved to: {topic_info_path}")

# Save the reduced topic model
model_path = os.path.join(output_dir, model_file)
reduced_topic_model.save(model_path, save_embedding_model=True)
print(f"Reduced topic model saved to: {model_path}")

## 2712 FOR GENERAL
# output_dir = r"C:/Thesis/MScThesis/Thesis/topic_models"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# topics, probs = topic_model.fit_transform(general_documents)
# nr_topics = 50  # Set the desired number of topics dynamically

# # Reduce the topic model
# reduced_topic_model = topic_model.reduce_topics(general_documents, nr_topics=nr_topics)

# # Extract topic information
# topic_info = reduced_topic_model.get_topic_info()

# # Dynamic filenames
# # topic_info_file = f"isot_{nr_topics}_topics_extracted.csv"  # CSV file for topic info --> FOR KEYBERTINSPIRED
# # model_file = f"isot_{nr_topics}_topic_model"  # File for saving the reduced topic model --> FOR KEYBERTINSPIRED
# topic_info_file = f"general_{nr_topics}_topics_extracted.csv"  # CSV file for topic info
# model_file = f"general_{nr_topics}_topic_model"  # File for saving the reduced topic model

# # Save topic info to CSV
# topic_info_path = os.path.join(output_dir, topic_info_file)
# topic_info.to_csv(topic_info_path, index=False)
# print(f"Topic information saved to: {topic_info_path}")

# # Save the reduced topic model
# model_path = os.path.join(output_dir, model_file)
# reduced_topic_model.save(model_path, save_embedding_model=True)
# print(f"Reduced topic model saved to: {model_path}")


# ## 2712 ORIGINAL FOR ISOT
# output_dir = r"C:/Thesis/MScThesis/Thesis/topic_models"
# os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

# topics, probs = topic_model.fit_transform(isot_documents)
# nr_topics = 50  # Set the desired number of topics dynamically

# # Reduce the topic model
# reduced_topic_model = topic_model.reduce_topics(isot_documents, nr_topics=nr_topics)

# # Extract topic information
# topic_info = reduced_topic_model.get_topic_info()

# # Dynamic filenames
# # topic_info_file = f"isot_{nr_topics}_topics_extracted.csv"  # CSV file for topic info --> FOR KEYBERTINSPIRED
# # model_file = f"isot_{nr_topics}_topic_model"  # File for saving the reduced topic model --> FOR KEYBERTINSPIRED
# topic_info_file = f"finetuned_isot_{nr_topics}_topics_extracted.csv"  # CSV file for topic info
# model_file = f"finetuned_isot_{nr_topics}_topic_model"  # File for saving the reduced topic model

# # Save topic info to CSV
# topic_info_path = os.path.join(output_dir, topic_info_file)
# topic_info.to_csv(topic_info_path, index=False)
# print(f"Topic information saved to: {topic_info_path}")

# # Save the reduced topic model
# model_path = os.path.join(output_dir, model_file)
# reduced_topic_model.save(model_path, save_embedding_model=True)
# print(f"Reduced topic model saved to: {model_path}")
# # topic_model.reduce_topics(isot_documents, topics, probs, nr_topics=50)
# # topic_model.save("isot_topic_model", save_embedding_model=True)

