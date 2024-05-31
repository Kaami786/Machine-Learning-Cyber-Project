import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the CSV data
data = pd.read_csv('preprocesed.csv')

# Preprocessing
data['text'] = data['text'].astype(str)  # Ensure all text data is treated as strings
data = data.dropna(subset=['processed_text'])  # Drop rows with NaN in 'processed_text'
X = data['processed_text']
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Train model
decision_tree.fit(X_train, y_train)

# Evaluate model
y_pred = decision_tree.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Additional metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save the model
joblib_file = "decision_tree_model.joblib"
joblib.dump(decision_tree, joblib_file)

print(f"Model saved as {joblib_file}")
