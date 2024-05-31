import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import joblib  # for loading the models

# Define paths
train_dir = 'Bottle Images'
test_dir = 'Testing'

# Image dimensions
img_width, img_height = 224, 224

# Prepare data generators
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Extract features
def extract_features(generator, num_images):
    features = []
    labels = []
    
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(inputs_batch)
        features.append(features_batch.reshape((features_batch.shape[0], 7 * 7 * 512)))
        labels.append(labels_batch)
        
        if len(features) * generator.batch_size >= num_images:
            break

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

num_test_images = len(test_generator.filenames)
test_features, test_labels = extract_features(test_generator, num_test_images)

# Flatten the labels
test_labels = np.argmax(test_labels, axis=1)

# Load the CNN model
cnn_model = load_model('cnn_best_model.keras')

# Function to get features for KNN and Decision Tree
def get_model_features(generator, model):
    features = []
    for inputs_batch, _ in generator:
        features_batch = model.predict(inputs_batch)
        features.append(features_batch)
        
        if len(features) * generator.batch_size >= len(generator.filenames):
            break

    features = np.concatenate(features)
    return features

# Get CNN features
cnn_features = get_model_features(test_generator, cnn_model)

# Load the KNN model
knn_classifier = joblib.load('knn_classifier_model.pkl')

# Load the Decision Tree model
dt_classifier = joblib.load('dt_classifier_model.pkl')

# Create Stacking Classifier
estimators = [
    ('knn', knn_classifier),
    ('dt', dt_classifier)
]

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    passthrough=True
)

# Train the stacking classifier on the extracted features
stacking_classifier.fit(np.hstack((cnn_features, test_features)), test_labels)

# Save the stacking model
joblib.dump(stacking_classifier, 'stacking_classifier_model.pkl')

# Load the stacking model (if needed)
stacking_classifier = joblib.load('stacking_classifier_model.pkl')

# Make predictions with stacking model
stacking_predictions = stacking_classifier.predict(np.hstack((cnn_features, test_features)))

# Calculate metrics for Stacking Classifier
stacking_conf_matrix = confusion_matrix(test_labels, stacking_predictions)
stacking_accuracy = accuracy_score(test_labels, stacking_predictions)
stacking_precision = precision_score(test_labels, stacking_predictions, average='weighted')

# Print the results
print("Stacking Classifier Confusion Matrix:")
print(stacking_conf_matrix)
print(f"Stacking Classifier Accuracy: {stacking_accuracy}")
print(f"Stacking Classifier Precision: {stacking_precision}")
