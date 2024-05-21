import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


def load_dataset(data_path='data'):
    with open(os.path.join(data_path, 'train_dataset.pkl'), 'rb') as f:
        save = pickle.load(f)
    dataset = save['dataset']
    labels = save['labels']
    print('Dataset', dataset.shape, labels.shape)

    # Reshape the labels to match the model's output shape
    labels = labels[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, X_train, y_train, epochs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print('Confusion matrix:')
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)
    make_pdf_of_confusion_matrix(cm)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy:', scores[1])

def make_pdf_of_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    labels = ['BENIGN', 'RAM', 'BLOCK',]
    sns.heatmap(cm, annot=True, cmap="Reds", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf")

def main():
    # Load the dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Convert labels to categorical format
    num_classes = 8  # Number of classes
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_model(input_shape, num_classes)

    # Train the model
    train_model(model, X_train, y_train, epochs=100)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
