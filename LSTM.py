import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Read the CSV file
data = pd.read_csv('input.csv')

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

X = X.reshape(344, 1, 38)

# Define the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, num_features)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions on new data
new_data = pd.read_csv('test.csv').values
new_data = data.iloc[:, :-1].values
new_data = new_data[1]
print("new_X_shape", new_data.shape)
# num_samples, num_features = new_data.shape
# print("new_samples",num_samples)
# print("new_features",num_features)

new_data = new_data.reshape(1, 1, 38)
predictions = model.predict(new_data)
print(predictions)
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Read the CSV file
data = pd.read_csv('input.csv')

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

X = X.reshape(344, 1, 38)

# Define the LSTM model
model = Sequential()
model.add(LSTM(32, input_shape=(timesteps, num_features)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions on new data
new_data = pd.read_csv('test.csv').values
new_data = data.iloc[:, :-1].values
new_data = new_data[1]
print("new_X_shape", new_data.shape)

new_data = new_data.reshape(1, 1, 38)
predictions = model.predict(new_data)
print(predictions)