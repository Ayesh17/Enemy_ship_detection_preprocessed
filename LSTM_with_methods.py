import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
import sklearn.model_selection
from sklearn.metrics import confusion_matrix, accuracy_score

# Folder structure
train_data_dir = 'train_data'
test_data_dir = 'test_data'

def load_dataset():
    behavior_classes = ['BENIGN', 'RAM', 'BLOCK', 'HERD', 'CROSS', 'HEADON', 'OVERTAKE', 'STATIONARY']

    # Read all CSV files in the train_data directory
    csv_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
    data = pd.DataFrame()  # Initialize an empty DataFrame

    # Concatenate all CSV files into a single DataFrame
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(train_data_dir, csv_file))
        data = pd.concat([data, df])

    return data

def train_test_split(data, test_size=0.2):
    """Splits the dataset into train and test sets."""
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

def design_model(num_features):
    timesteps = 1
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(timesteps, num_features)))
    model.add(Dense(8, activation='softmax'))
    return model

def make_pdf_of_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    labels = ['BLOCK', 'BENIGN', 'STATIONARY', 'HEADON', 'OVERTAKE', 'CROSS', 'RAM', 'HERD']
    sns.heatmap(cm, annot=True, cmap="Reds", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf")

def test(X_test, y_test):
    predictions = model.predict(X_test)

    # Calculate accuracy and confusion matrix
    true_labels = y_test
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(true_labels, y_pred)
    print("Confusion", cm)
    make_pdf_of_confusion_matrix(cm)
    acc = accuracy_score(true_labels, y_pred)
    return acc

# Load the dataset
data = load_dataset()

# Split the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(data)

# Convert the DataFrames to NumPy arrays.
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Reshape the arrays.
num_samples, num_features = X_train.shape
X_train = X_train.reshape(num_samples, 1, num_features)
y_train = y_train.reshape(num_samples, 1)

num_samples, num_features = X_test.shape
X_test = X_test.reshape(num_samples, 1, num_features)
y_test = y_test.reshape(num_samples, 1)

# One-hot encode the target variable
y_one_hot = keras.utils.to_categorical(y_train, num_classes=8)

model = design_model(num_features)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_one_hot, epochs=10, batch_size=16)

acc = test(X_test, y_test)
print("Overall Accuracy : ", acc)
