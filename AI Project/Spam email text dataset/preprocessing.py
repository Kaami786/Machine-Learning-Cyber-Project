import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from collections import defaultdict

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Dictionary for expanding abbreviations
abbreviations = {
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "i've": "i have",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "there's": "there is",
    "that's": "that is",
    # Add more abbreviations as needed
}

# Function to expand abbreviations
def expand_abbreviations(text):
    words = text.split()
    expanded_words = [abbreviations[word] if word in abbreviations else word for word in words]
    return ' '.join(expanded_words)

# Function to preprocess text
def preprocess_text(text):
    # Ensure text is a string
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Expand abbreviations
    text = expand_abbreviations(text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load the CSV file
input_file = 'spam_Emails_data.csv'
output_file = 'preprocesed.csv'
df = pd.read_csv(input_file)

# Assuming text is in a column named 'text'
if 'text' not in df.columns:
    raise ValueError("The input CSV file must have a 'text' column")

# Process each text entry in the dataframe
stop_words = set(stopwords.words('english'))
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove stop words
def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Apply stop word removal
df['processed_text'] = df['processed_text'].apply(remove_stop_words)

# Remove extra stop words
extra_stop_words = set(["additional", "specific", "words", "to", "remove"])
df['processed_text'] = df['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in extra_stop_words]))

# Save the processed dataframe to a new CSV file
df.to_csv(output_file, index=False)

print(f"Processed CSV saved as {output_file}")
