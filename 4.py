import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import re

# Memuat dataset
dataset_path = "IKN_Nusantara.csv"
data = pd.read_csv(dataset_path)

# Stoper
def preprocess_text(text):
    # Remove specific symbols using regex
    processed_text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", "", text)
    return processed_text

# Apply preprocessing to the 'description' column
data['title'] = data['title'].apply(preprocess_text)

data.to_csv("sentiment_analysis_results4.csv", index=True)

# Memuat model dan tokenizer
model_name = 'indolem/indobert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Fungsi untuk melakukan analisis sentimen pada tweet
def analyze_sentiment(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(tokens).unsqueeze(0)

    outputs = model(input_ids)
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.detach().numpy()  # Menggunakan detach() sebelum numpy()

# Ekstraksi representasi vektor dari setiap tweet
embeddings = []
total_tweets = len(data['title'])
with tqdm(total=total_tweets, desc="Extracting embeddings") as pbar:
    for tweet in data['title']:
        embedding = analyze_sentiment(tweet)
        embeddings.append(embedding)
        pbar.update(1)
        pbar.set_postfix({"Progress": f"{pbar.n}/{pbar.total}"})

# Mengubah tensor embeddings menjadi array 2D
embeddings_flat = [e.flatten() for e in embeddings]

# Membuat DataFrame dari embeddings
embeddings_df = pd.DataFrame(embeddings_flat)

# Menggabungkan hasil representasi vektor dengan dataset
data_with_embeddings = pd.concat([data, embeddings_df], axis=1)

# Menyimpan dataset dengan representasi vektor ke file CSV
output_path = "IKN_Nusantara_with_embeddings4.csv"
data_with_embeddings.to_csv(output_path, index=False)

print("Representasi vektor telah berhasil diekstraksi dan disimpan di", output_path)
