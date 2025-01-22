import pandas as pd
from bertopic import BERTopic

# Paths
model_path = "C:/Thesis/MScThesis/Thesis/topic_models/finetuned_isot_50_topics_model"
data_path = "C:/Thesis/MScThesis/Thesis/src/data/new_combined_real_fake.csv"

# Load the BERTopic model
topic_model = BERTopic.load(model_path)

# Load the dataset
isot_df = pd.read_csv(data_path)

# Extract the text column as documents
isot_documents = isot_df['text'].tolist()

# Transform the documents to get topics and probabilities
topics, probs = topic_model.transform(isot_documents)

# Get document info for the first 10 documents
document_info = topic_model.get_document_info(isot_documents[:10])

# Create a DataFrame to store document information along with topic probabilities
result = pd.DataFrame()

for idx in range(10):  # For the first 10 documents
    doc_info = document_info.iloc[idx]
    topic_probs = probs[idx]  # Probabilities for this document

    # Add the topic probabilities and their sum to the document info
    doc_entry = {
        "Document ID": idx,
        "Document Text": isot_documents[idx],
        "Assigned Topic": doc_info["Topic"],
        "Topic Probabilities": topic_probs.tolist(),  # Convert to a list for saving
        "Sum of Probabilities": sum(topic_probs)  # This should be 1.0
    }
    result = pd.concat([result, pd.DataFrame([doc_entry])], ignore_index=True)

# Save or display the result
result_path = "C:/Thesis/MScThesis/Thesis/topic_models/document_topic_probabilities.csv"
result.to_csv(result_path, index=False)
print(f"Document topic probabilities saved to: {result_path}")

# Print the result for the first 10 documents
print(result.head(10))
