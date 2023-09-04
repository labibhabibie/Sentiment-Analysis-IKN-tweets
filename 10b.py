import pandas as pd

# Load dataset with sentiment predictions
dataset_path = "IKN_Nusantara_with_predictions.csv"
data_with_predictions = pd.read_csv(dataset_path)

# Function to map sentiment predictions to categories
def map_sentiment_category(sentiment):
    if sentiment >= 0.4:  # Adjust the threshold as needed
        return "positive"
    elif sentiment <= -0.4:  # Adjust the threshold as needed
        return "negative"
    else:
        return "neutral"

# Apply mapping function to create sentiment category column
data_with_predictions['sentiment_category'] = data_with_predictions['sentiment_prediction'].apply(map_sentiment_category)

# Group data by sentiment category and count occurrences
sentiment_counts = data_with_predictions['sentiment_category'].value_counts()

# Print sentiment category counts
print("Sentiment Category Counts:")
print(sentiment_counts)
