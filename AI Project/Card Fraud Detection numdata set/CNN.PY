from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import pandas as pd
import numpy as np


# Load the dataset
data = pd.read_csv('fraud test.csv')

# Encode categorical variables
label_encoders = {}
for column in ['merchant', 'category', 'gender', 'state', 'job']:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split features and target variable
X = data.drop(columns=['is_fraud'])
y = data['is_fraud']


# Encode categorical variables if needed
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input data for CNN
X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Predict classes for test set
y_pred_probabilities = model.predict(X_test)
y_pred = np.argmax(y_pred_probabilities, axis=-1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
num_parameters = model.count_params()
num_layers = len(model.layers)

# Print evaluation metrics
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Confusion Matrix:')
print(conf_matrix)
print('Number of parameters:', num_parameters)
print('Number of layers:', num_layers)

# Save the model
model.save('fraud_detection_cnn_model.h5')
# Save the model using the native Keras format
model.save('fraud_detection_model.keras')

