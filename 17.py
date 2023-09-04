import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import re
import nltk
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F

nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = set(stopwords.words('indonesian'))

# Memuat dataset
dataset_path = "IKN_Nusantara.csv"
data = pd.read_csv(dataset_path)

# Memuat data training
train_dataset_path = "train.csv"
train_data = pd.read_csv(train_dataset_path)

# Preprocessing teks
def preprocess_text(text):
    # Menghapus karakter non-alfabet
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Mengubah teks menjadi lowercase
    text = text.lower()
    # Menghapus stopwords
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text

# Menerapkan preprocessing pada kolom 'text'
data['title'] = data['title'].apply(preprocess_text)

# Split data menjadi data training dan data testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Membangun dataset untuk training
class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['title']
        return text

# Inisialisasi tokenizer dan model BERT
model_name = 'indolem/indobert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Fungsi untuk melakukan tokenisasi teks
def tokenize_text(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens

# Fungsi untuk melakukan padding pada token
def pad_tokens(tokens, max_len):
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return tokens

# Fungsi untuk membangun batch
def collate_fn(batch):
    texts = batch
    tokens = [tokenize_text(text) for text in texts]
    max_len = max([len(token) for token in tokens])
    padded_tokens = [pad_tokens(token, max_len) for token in tokens]
    attention_masks = [[1] * len(token) + [0] * (max_len - len(token)) for token in padded_tokens]
    return torch.tensor(padded_tokens), torch.tensor(attention_masks)

# Membangun data loader untuk testing
test_dataset = SentimentDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# Definisi model RCNN
class RCNN(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(RCNN, self).__init__()
        self.hidden_size = hidden_size
        self.bert = model
        self.conv = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(self.hidden_size + self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.permute(0, 2, 1)
        conv_out = self.conv(hidden_states)
        conv_out = F.relu(conv_out)
        max_pool = F.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2)
        output = self.fc(max_pool)
        return output

# Membangun data loader untuk training
train_dataset = SentimentDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Melakukan training model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RCNN(hidden_size=768, num_classes=3)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
total_steps = len(train_loader) * num_epochs

for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()
    for i, (input_ids, attention_masks) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_masks)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluasi model
model.eval()

# Melakukan prediksi sentimen pada data testing
predictions = []
with torch.no_grad():
    for input_ids, attention_masks in test_loader:
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        outputs = model(input_ids, attention_masks)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

# Menambahkan kolom 'sentiment' pada data testing
test_data['sentiment'] = predictions

# Menyimpan dataset dengan label sentimen ke file CSV
output_path = "IKN_Nusantara_with_sentiment.csv"
test_data.to_csv(output_path, index=False)

print("Dataset dengan label sentimen telah disimpan di", output_path)
