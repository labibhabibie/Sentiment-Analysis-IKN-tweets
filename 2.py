import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm  # Menambahkan library tqdm untuk presentase loading

# Memuat dataset
dataset_path = "IKN_Nusantara.csv"
data = pd.read_csv(dataset_path)

# Memuat model dan tokenizer
model_name = 'indolem/indobert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Fungsi untuk melakukan analisis sentimen pada tweet
def analyze_sentiment(text):
    # Tokenisasi teks
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Konversi ke tensor
    input_ids = torch.tensor(tokens).unsqueeze(0)  # Menambahkan dimensi batch

    # Peroleh representasi vektor
    outputs = model(input_ids)

    # Peroleh representasi vektor token pertama
    embedding = outputs.last_hidden_state[:, 0, :]

    # Kembalikan representasi vektor
    return embedding

# Ekstraksi representasi vektor dari setiap tweet
embeddings = []
total_tweets = len(data['title'])
with tqdm(total=total_tweets, desc="Extracting embeddings") as pbar:  # Menggunakan tqdm untuk presentase loading
    for tweet in data['title']:
        embedding = analyze_sentiment(tweet)
        embeddings.append(embedding)
        pbar.update(1)  # Memperbarui presentase loading
        pbar.set_postfix({"Progress": f"{pbar.n}/{pbar.total}"})  # Menampilkan kemajuan saat ini

# Menggabungkan hasil representasi vektor dengan dataset
data_with_embeddings = pd.concat([data, pd.DataFrame(embeddings)], axis=1)

# Menyimpan dataset dengan representasi vektor ke file CSV
output_path = "IKN_Nusantara_with_embeddings.csv"
data_with_embeddings.to_csv(output_path, index=False)

print("Representasi vektor telah berhasil diekstraksi dan disimpan di", output_path)
