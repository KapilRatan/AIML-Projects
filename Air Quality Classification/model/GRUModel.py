import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the dataset
data = pd.read_csv("Air_Quality_Classified_Updated.csv")  # Replace with your dataset file path

# Map Air Quality Class to numeric labels
label_mapping = {
    "Class 1 (Good Air Quality)": 0,
    "Class 2 (Moderate Air Quality)": 1,
    "Class 3 (Unhealthy Air Quality)": 2,
    "Class 4 (Very Unhealthy Air Quality)": 3,
    "Class 5 (Hazardous Air Quality)": 4,
}
data['Air Quality Class'] = data['Air Quality Class'].map(label_mapping)

# Select relevant features and target
features = ['Data Value', 'Time Period']  # Add other relevant features if applicable
target = 'Air Quality Class'

# Encode categorical features
label_encoder = LabelEncoder()
data['Time Period'] = label_encoder.fit_transform(data['Time Period'])

# Normalize numerical features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Split data into train and test sets
X = data[features].values
y = data[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input for GRU: [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the GRU model
model = Sequential([
    GRU(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),
    Dense(32, activation='tanh'),
    Dense(len(label_mapping), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model and the label encoder
model.save('air_quality_gru_model.h5')  # Save the trained model
import joblib
joblib.dump(label_encoder, 'label_encoder.pkl')  # Save the label encoder
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

# Optionally, plot results
# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Compute confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.keys(), yticklabels=label_mapping.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
