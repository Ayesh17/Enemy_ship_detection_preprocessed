import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

# List of behavior classes
behavior_classes = ['benign', 'herd', 'ram']

# Folder structure
train_data_dir = 'train_data'

# Iterate over behavior classes
for behavior_class in behavior_classes:

    # Get list of CSV files for the behavior class
    csv_files = [f for f in os.listdir(os.path.join(train_data_dir, behavior_class)) if f.endswith('.csv')]

    # Read the CSV files and concatenate them into a single DataFrame
    data = pd.concat([pd.read_csv(os.path.join(train_data_dir, behavior_class, f)) for f in csv_files])

    # Extract input features and target variable
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Reshape the input features
    # X shape: (num_samples, timesteps, features)
    print("X_shape", X.shape)
    num_samples, num_features = X.shape
    print("num_samples",num_samples)
    print("num_features",num_features)
    timesteps = 1  # Adjust the number of time steps as needed
    print("Total number of elements:", num_samples * timesteps * num_features)

    X = X.reshape(num_samples, 1, num_features)

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(32, input_shape=(timesteps, num_features)))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)

print()
print("*" * 210)
print("Training finished")
print()

# Make predictions on new data
new_data = pd.read_csv('test.csv').values
new_data = data.iloc[:, :-1].values
new_data = new_data[1]
print("new_X_shape", new_data.shape)

new_data = new_data.reshape(1, 1, 38)
predictions = model.predict(new_data)
print(predictions)