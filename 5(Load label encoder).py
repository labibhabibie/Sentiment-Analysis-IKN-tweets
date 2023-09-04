import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Load dataset
dataset_path = "IKN_Nusantara.csv"
data = pd.read_csv(dataset_path)

# Load model and tokenizer
model_name = 'indolem/indobert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Function to perform sentiment analysis on tweets
def analyze_sentiment(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)

    outputs = model(input_ids)
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.detach().numpy()

# Extract vector representations from each tweet
embeddings = []
total_tweets = len(data['title'])
with tqdm(total=total_tweets, desc="Extracting embeddings") as pbar:
    for tweet in data['title']:
        embedding = analyze_sentiment(tweet)
        embeddings.append(embedding)
        pbar.update(1)
        pbar.set_postfix({"Progress": f"{pbar.n}/{pbar.total}"})

# Convert tensor embeddings to 2D array
embeddings_flat = [e.flatten() for e in embeddings]

# Create DataFrame from embeddings
embeddings_df = pd.DataFrame(embeddings_flat)

# Concatenate vector representations with the dataset
data_with_embeddings = pd.concat([data, embeddings_df], axis=1)

# Load Label Encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = ['negatif', 'netral', 'positif']

# Encode labels to numeric codes
data_with_embeddings['label_encoded'] = label_encoder.transform(data['label'])

# Save the dataset with vector representations to a CSV file
output_path = "IKN_Nusantara_with_embeddings.csv"
data_with_embeddings.to_csv(output_path, index=False)

print("Vector representations have been successfully extracted and saved to", output_path)
