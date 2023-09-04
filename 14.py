import torch
import pandas as pd
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import re
import nltk
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = set(stopwords.words('indonesian'))

# Load pre-trained model and tokenizer
model_name = 'indolem/indobert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Load labeled training dataset
train_dataset_path = "train.csv"
train_data = pd.read_csv(train_dataset_path)

# Preprocess training data
# def preprocess_text(text):
#     # Remove symbols using regex
#     processed_text = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", "", text)

#     # Convert text to lowercase
#     processed_text = processed_text.lower()

#     # Remove stopwords
#     stopwords = set(stopwords.words('indonesian'))
#     processed_text = " ".join(word for word in processed_text.split() if word not in stopwords)

#     return processed_text

def preprocess_text(text):
    # Menghapus karakter non-alfabet
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # Mengubah teks menjadi lowercase
    text = text.lower()
    # Menghapus stopwords
    text = " ".join([word for word in text.split() if word not in stopwords])
    return text

train_data['text'] = train_data['text'].apply(preprocess_text)

# Create custom dataset for RCNN
class RCNNDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

# Create RCNN model
class RCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(RCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bert = model
        self.dropout = nn.Dropout(0.2)
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        embeddings = outputs.last_hidden_state
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        conv_out = self.conv(embeddings)
        conv_out = self.relu(conv_out)
        conv_out = self.maxpool(conv_out)
        conv_out = conv_out.squeeze()
        output = self.fc(conv_out)
        return output

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
embedding_dim = 768
hidden_dim = 128
num_classes = 3
num_epochs = 5
batch_size = 32
learning_rate = 1e-3

# Create RCNN model instance
model = RCNN(embedding_dim, hidden_dim, num_classes).to(device)

# Create optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Split training data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Create RCNN datasets and dataloaders
train_dataset = RCNNDataset(train_data, tokenizer)
val_dataset = RCNNDataset(val_data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_dataloader, desc="Epoch {}".format(epoch + 1)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total
    avg_train_loss = train_loss / len(train_dataloader)

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_dataloader)

    print("Epoch {} - Train Loss: {:.4f}, Train Accuracy: {:.4f}, Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(
        epoch + 1, avg_train_loss, train_accuracy, avg_val_loss, val_accuracy))

# Load unlabeled dataset
unlabeled_dataset_path = "IKN_Nusantara.csv"
unlabeled_data = pd.read_csv(unlabeled_dataset_path)

# Change columns name
unlabeled_data.rename(columns={"title": "text"}, inplace=True)

# Preprocess unlabeled data
unlabeled_data['text'] = unlabeled_data['text'].apply(preprocess_text)

# Create unlabeled dataset for RCNN
unlabeled_dataset = RCNNDataset(unlabeled_data, tokenizer)

unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)

# Predict sentiment using RCNN
predictions = []

model.eval()

with torch.no_grad():
    for batch in tqdm(unlabeled_dataloader, desc="Predicting Sentiment"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, dim=1)

        predictions.extend(predicted.tolist())

# Add predictions to the unlabeled dataset
unlabeled_data['sentiment_prediction'] = predictions

# Save the dataset with sentiment predictions to a CSV file
output_path = "unlabeled_data_with_predictions.csv"
unlabeled_data.to_csv(output_path, index=False)

print("Sentiment predictions have been successfully performed and saved to", output_path)
