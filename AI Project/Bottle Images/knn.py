import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import joblib  # for saving the model

# Define paths
train_dir = 'Bottle Images'
test_dir = 'Testing'

# Image dimensions
img_width, img_height = 224, 224

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Extract features
def extract_features(generator, num_images):
    # Initialize empty lists to store features and labels
    features = []
    labels = []
    
    for inputs_batch, labels_batch in generator:
        # Predict features using VGG16
        features_batch = base_model.predict(inputs_batch)
        # Reshape features to the required shape
        features.append(features_batch.reshape((features_batch.shape[0], 7 * 7 * 512)))
        labels.append(labels_batch)
        
        # Break the loop if we have processed enough images
        if len(features) * generator.batch_size >= num_images:
            break

    # Concatenate lists to form arrays
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

num_train_images = len(train_generator.filenames)
num_test_images = len(test_generator.filenames)

train_features, train_labels = extract_features(train_generator, num_train_images)
test_features, test_labels = extract_features(test_generator, num_test_images)

# Flatten the labels
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Train the KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_classifier.fit(train_features, train_labels)

# Save the model
joblib.dump(knn_classifier, 'knn_classifier_model.pkl')

# Load the model (if needed)
knn_classifier = joblib.load('knn_classifier_model.pkl')

# Make predictions
test_predictions = knn_classifier.predict(test_features)

# Calculate metrics
conf_matrix = confusion_matrix(test_labels, test_predictions)
accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average='weighted')

# Print the results
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
