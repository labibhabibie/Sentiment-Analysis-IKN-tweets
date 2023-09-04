import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Load pre-trained model and tokenizer
model_name = 'indolem/indobert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load dataset
dataset_path = "IKN_Nusantara.csv"
data = pd.read_csv(dataset_path)

# Function to perform sentiment analysis on a text
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

# Perform sentiment prediction using SVM
labels = data['label']  # Assuming the dataset contains a 'label' column with sentiment labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

svm_model = SVC(kernel='linear')
svm_model.fit(embeddings_df, encoded_labels)
predictions = svm_model.predict(embeddings_df)
decoded_predictions = label_encoder.inverse_transform(predictions)

# Add predictions to the dataset
data_with_predictions = data.copy()
data_with_predictions['sentiment_prediction'] = decoded_predictions

# Save the dataset with sentiment predictions to a CSV file
output_path = "IKN_Nusantara_with_predictions.csv"
data_with_predictions.to_csv(output_path, index=False)

print("Sentiment predictions have been successfully performed and saved to", output_path)
