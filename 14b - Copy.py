import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
print(os.getcwd())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print('Device using: ',device)

# Hyperparameters
max_length = 256
batch_size = 16 # 8
learning_rate = 3e-5 # 2e-5
num_epochs = 5 # 2

# Custom dataset class
class RCNNDataset(Dataset):
    def __init__(self, data, tokenizer, label_dict):
        self.data = data
        self.tokenizer = tokenizer
        self.label_dict = label_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text_clean']
        label = self.data.iloc[index]['polarity']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        label_id = self.label_dict[label]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_id
        }

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("indolem/indobertweet-base-uncased", use_fast=True)

# Load labeled training data
# train_data = pd.read_csv('c:/Users/labib\Documents/Kuliah/Materi/Sem 6/MPI/Artikel ilmiah/clean.csv')
train_data = pd.read_csv('clean.csv')

actual_label = train_data['polarity']

# Define label dictionary
label_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

# Split labeled data into train and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
print(f'Total Train: {len(train_data)}')
print(train_data[5:],'\n\n')
print(f'Total Val  : {len(val_data)}')
print(val_data[5:])

# Create RCNN datasets
train_dataset = RCNNDataset(train_data, tokenizer, label_dict)
val_dataset = RCNNDataset(val_data, tokenizer, label_dict)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained('indolem/indobertweet-base-uncased', num_labels=3)
model.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)[0]
        _, predicted = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)[0]
            _, predicted = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    val_loss /= len(val_loader)

    # Print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    print()

    
# print(classification_report(prediction, actual_label, target_names=["Negatif", "Netral", "Positif"]))

# Save the trained model
output_path = 'model/rcnn_sentiment_model(Epoch=5)'
print("Output Path: ", output_path)
model.save_pretrained(output_path)
