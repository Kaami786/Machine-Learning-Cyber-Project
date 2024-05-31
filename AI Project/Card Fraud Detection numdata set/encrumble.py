import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import os

# Load the dataset
data = pd.read_csv('fraud test.csv')

# Assuming 'category' column contains text data for classification
X = data['category']
y = data['is_fraud']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a new vectorizer and save it
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save the vectorizer

# Transform the test data
X_test_vectorized = vectorizer.transform(X_test)

# Load the saved models
dt_loaded = joblib.load('decision_tree_model.pkl')
knn_loaded = joblib.load('knn_model.pkl')

# Check if CNN model file exists
cnn_model_path = 'cnn_model.h5'
if not os.path.isfile(cnn_model_path):
    print(f"File {cnn_model_path} does not exist. Please check the path.")
else:
    cnn_loaded = load_model(cnn_model_path)

    # Preprocessing for CNN model
    max_num_words = 10000  # This should be the same as used during training
    max_sequence_length = 100  # This should be the same as used during training

    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(X_train)  # Fit on training data to create the word index
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

    # Make predictions
    dt_pred = dt_loaded.predict(X_test_vectorized)
    knn_pred = knn_loaded.predict(X_test_vectorized)
    cnn_pred_probabilities = cnn_loaded.predict(X_test_pad)
    cnn_pred = np.where(cnn_pred_probabilities > 0.5, 1, 0).flatten()

    # Calculate accuracy and precision for Decision Tree and KNN
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_precision = precision_score(y_test, dt_pred, zero_division=1)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred, zero_division=1)

    # Calculate accuracy and precision for CNN
    cnn_accuracy = accuracy_score(y_test, cnn_pred)
    cnn_precision = precision_score(y_test, cnn_pred, zero_division=1)

    print("Decision Tree Accuracy:", dt_accuracy)
    print("Decision Tree Precision:", dt_precision)
    print("KNN Accuracy:", knn_accuracy)
    print("KNN Precision:", knn_precision)
    print("CNN Accuracy:", cnn_accuracy)
    print("CNN Precision:", cnn_precision)

    # Calculate confusion matrix for Decision Tree and KNN
    dt_conf_matrix = confusion_matrix(y_test, dt_pred)
    knn_conf_matrix = confusion_matrix(y_test, knn_pred)
    cnn_conf_matrix = confusion_matrix(y_test, cnn_pred)
    print("Decision Tree Confusion Matrix:")
    print(dt_conf_matrix)
    print("KNN Confusion Matrix:")
    print(knn_conf_matrix)
    print("CNN Confusion Matrix:")
    print(cnn_conf_matrix)
