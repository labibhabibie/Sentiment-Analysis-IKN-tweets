import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# Load pre-trained model and tokenizer
model_name = 'indolem/indobert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load training dataset
train_dataset_path = "train.csv"
train_data = pd.read_csv(train_dataset_path)

# Load unlabeled dataset
unlabeled_dataset_path = "IKN_Nusantara.csv"
unlabeled_data = pd.read_csv(unlabeled_dataset_path)

# Function to perform sentiment analysis on a text
def analyze_sentiment(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)

    outputs = model(input_ids)
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.detach().numpy()

# Extract vector representations from each tweet in the training dataset
train_embeddings = []
total_train_tweets = len(train_data['text'])
with tqdm(total=total_train_tweets, desc="Extracting training embeddings") as pbar:
    for tweet in train_data['text']:
        embedding = analyze_sentiment(tweet)
        train_embeddings.append(embedding)
        pbar.update(1)
        pbar.set_postfix({"Progress": f"{pbar.n}/{pbar.total}"})

# Convert tensor embeddings to 2D array
train_embeddings_flat = [e.flatten() for e in train_embeddings]

# Create DataFrame from training embeddings and labels
train_embeddings_df = pd.DataFrame(train_embeddings_flat)
train_labels = train_data['label']
train_dataset = pd.concat([train_embeddings_df, train_labels], axis=1)

# Train a classification model
# Replace the following code with your own classification model training code
# Use train_dataset as the input (embeddings as features and labels as targets)
# Make sure to handle the imbalance in class distribution if necessary

# Extract vector representations from each tweet in the unlabeled dataset
unlabeled_embeddings = []
total_unlabeled_tweets = len(unlabeled_data['title'])
with tqdm(total=total_unlabeled_tweets, desc="Extracting unlabeled embeddings") as pbar:
    for tweet in unlabeled_data['title']:
        embedding = analyze_sentiment(tweet)
        unlabeled_embeddings.append(embedding)
        pbar.update(1)
        pbar.set_postfix({"Progress": f"{pbar.n}/{pbar.total}"})

# Convert tensor embeddings to 2D array
unlabeled_embeddings_flat = [e.flatten() for e in unlabeled_embeddings]

# Create DataFrame from unlabeled embeddings
unlabeled_embeddings_df = pd.DataFrame(unlabeled_embeddings_flat)

# Make predictions on the unlabeled dataset using the trained classification model
# Replace the following code with your own prediction code based on the trained model
# Use unlabeled_embeddings_df as the input (embeddings as features) to obtain predictions

# Save the dataset with sentiment predictions to a CSV file
output_path = "IKN_Nusantara_with_predictions13.csv"
unlabeled_data['sentiment_prediction'] = predictions  # Replace 'predictions' with your own predictions
unlabeled_data.to_csv(output_path, index=False)

print("Sentiment predictions have been successfully performed and saved to", output_path)
