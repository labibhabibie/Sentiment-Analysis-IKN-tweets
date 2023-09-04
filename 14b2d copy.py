import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from nltk.corpus import stopwords


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device using: ',device)

# Hyperparameters
max_length = 256
batch_size = 16 # 8
learning_rate = 3e-5 # 2e-5
num_epochs = 3 # 2

# Load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("indolem/indobertweet-base-uncased", use_fast=True)

# Load pre-trained model
model = BertForSequenceClassification.from_pretrained('model/rcnn_sentiment_model(Epoch=5)')
model.to(device)
model.eval()

# Custom dataset class for prediction
class PredictionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['title']

        # Preprocess text
        text = preprocess_text(text)

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

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

# Function for text preprocessing
def preprocess_text(text):
    # Remove symbols and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load unlabeled dataset
unlabeled_data = pd.read_csv('IKN_Nusantara.csv')

# Create prediction dataset
prediction_dataset = PredictionDataset(unlabeled_data, tokenizer)

# Create data loader
prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False)

# Perform prediction
predictions = []
with torch.no_grad():
    for batch in tqdm(prediction_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)[0]
        _, predicted = torch.max(outputs, dim=1)

        predictions.extend(predicted.cpu().numpy())

# Perform predictions label
labels = []
for prediction in tqdm(predictions, desc="Adding sentiment label"):
    if prediction == 0:
        labels.append('Negative')
    elif prediction == 1:
        labels.append('Neutral')
    elif prediction == 2:
        labels.append('Positive')

# Add labels to the unlabeled dataset
unlabeled_data['sentiment_label'] = labels

# Output percentage of each label
label_counts = unlabeled_data['sentiment_label'].value_counts(normalize=True)
for label, percentage in label_counts.items():
    print(f"{label}: {percentage * 100:.2f}%")

# Save the updated dataset with sentiment labels
unlabeled_data.to_csv('IKN_Nusantara_with_29Juli.csv', index=False)
