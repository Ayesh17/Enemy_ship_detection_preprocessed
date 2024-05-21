import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix, accuracy_score

# Folder structure
train_data_dir = 'train_data'
test_data_dir = 'test_data'

# Iterate over all CSV files in the train_data directory
csv_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
print("csv", csv_files)

# Read the CSV files and concatenate them into a single DataFrame
data = pd.concat([pd.read_csv(os.path.join(train_data_dir, f)) for f in csv_files])

# Extract input features and target variable
print("X_shape", data.shape)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print("y_shape", y.shape)
# Reshape the input features
# X shape: (num_samples, timesteps, features)
print("X_shape", X.shape)
num_samples, num_features = X.shape
timesteps = 1  # Adjust the number of time steps as needed
print("Total number of elements:", num_samples * timesteps * num_features)

X = X.reshape(num_samples, 1, num_features)

# One-hot encode the target variable
y_one_hot = keras.utils.to_categorical(y, num_classes=3)

# Define the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, num_features)))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y_one_hot, epochs=10, batch_size=32)

print()
print("*" * 210)
print("Training finished")
print()


# Make predictions on all test files
csv_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
for csv_file in csv_files:
    print("csv", csv_file)
    data = pd.read_csv(os.path.join(test_data_dir, csv_file))
    X = data.iloc[:, :-1].values
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape based on the number of samples and features
    print("X", len(X))
    predictions = model.predict(X)
    print(f"Prediction for {csv_file}: {predictions}")

    # Calculate accuracy and confusion matrix
    true_labels = data.iloc[:, -1].values
    y_pred = np.argmax(predictions, axis=1)
    # confusion_matrix = confusion_matrix(true_labels, y_pred)
    # print("y_pred",y_pred)
    # print("true_labels",true_labels)
    print("Accuracy:", accuracy_score(true_labels, y_pred))
    # print("Confusion matrix:")
    # print(confusion_matrix)
