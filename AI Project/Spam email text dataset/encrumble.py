import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the CSV data
data = pd.read_csv('preprocesed.csv')

# Preprocessing
data['text'] = data['text'].astype(str)  # Ensure all text data is treated as strings
data['processed_text'] = data['processed_text'].astype(str).fillna('')  # Fill NaNs with an empty string
X = data['processed_text']
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Tokenize text for CNN
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
max_sequence_length = 100
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Split data into train and test sets
X_train_tfidf, X_test_tfidf, X_train_padded, X_test_padded, y_train, y_test = train_test_split(
    X_tfidf, X_padded, y, test_size=0.2, random_state=42)

# Load models
knn = joblib.load('knn_model.joblib')
decision_tree = joblib.load('decision_tree_model.joblib')
cnn_model = load_model('cnn-text.h5')

# Get predictions from each model
knn_pred = knn.predict(X_test_tfidf)
decision_tree_pred = decision_tree.predict(X_test_tfidf)
cnn_pred_prob = cnn_model.predict(X_test_padded)
cnn_pred = (cnn_pred_prob > 0.5).astype("int32").flatten()

# Create new feature set for stacking
stacked_features = np.column_stack((knn_pred, decision_tree_pred, cnn_pred))

# Train meta-model (Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stacked_features, y_test)

# Make predictions with meta-model
meta_pred = meta_model.predict(stacked_features)

# Evaluate stacked model
conf_matrix = confusion_matrix(y_test, meta_pred)
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(y_test, meta_pred))

accuracy = accuracy_score(y_test, meta_pred)
precision = precision_score(y_test, meta_pred)
recall = recall_score(y_test, meta_pred)
f1 = f1_score(y_test, meta_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the meta-model
joblib_file = "meta_model.joblib"
joblib.dump(meta_model, joblib_file)

print(f"Meta-model saved as {joblib_file}")
