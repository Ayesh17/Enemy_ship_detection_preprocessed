import os
from random import random

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import random
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from Feature_Selection_Methods.univariate_feature_selection import univariate_feature_selection
from Feature_Selection_Methods.decision_trees_based_feature_selection import decision_trees_based_feature_selection
from Feature_Selection_Methods.random_forest_based_feature_selection import random_forest_based_feature_selection
from Feature_Selection_Methods.recursive_feature_selection import recursive_feature_selection
from DL_Models.RNN_model import RNN_model
from DL_Models.GRU_model import GRU_model
from DL_Models.LSTM_model import LSTM_model
from DL_Models.Bidirectional_RNN_model import Bidirectional_RNN_model
from DL_Models.Bidirectional_GRU_model import Bidirectional_GRU_model
from DL_Models.Bidirectional_LSTM_model import Bidirectional_LSTM_model



# Set the environment variable to ensure reproducibility
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# ...

# Folder structure
train_data_dir = '../Train_data'
test_data_dir = '../Test_data'

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(data_dir, data_path='data'):
    if data_dir == train_data_dir:
        with open(os.path.join(data_path, 'train_dataset.pkl'), 'rb') as f:
            save = pickle.load(f)
    elif data_dir == test_data_dir:
        with open(os.path.join(data_path, 'test_dataset.pkl'), 'rb') as f:
            save = pickle.load(f)
    else:
        with open(os.path.join(data_path, 'other_dataset.pkl'), 'rb') as f:
            save = pickle.load(f)
    dataset = save['dataset']
    labels = save['labels']
    print('Dataset', dataset.shape, labels.shape)

    # Reshape the labels to match the model's output shape
    labels = labels[:, -1]

    return dataset, labels


def train_model(model, X_train, y_train, X_val, y_val,  epochs):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    highest_accuracy = 0.0
    best_weights = None
    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, verbose=0)

        # Calculate training accuracy and loss
        train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1]
        train_loss = model.evaluate(X_train, y_train, verbose=0)[0]

        # Calculate validation accuracy and loss
        val_accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
        val_loss = model.evaluate(X_val, y_val, verbose=0)[0]

        print(f'Epoch {epoch + 1}/{epochs} - Training loss: {train_loss:.4f} - Training accuracy: {train_accuracy:.4f} - Validation loss: {val_loss:.4f} - Validation accuracy: {val_accuracy:.4f}')

        if val_accuracy > highest_accuracy:
            highest_accuracy = val_accuracy
            best_weights = model.get_weights()
    model.set_weights(best_weights)


def evaluate_model(model, X_test, y_test):

    # Confusion matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    print('Confusion matrix:')
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)
    make_pdf_of_confusion_matrix(cm)

    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    print('Accuracy:', accuracy)

    # Each class evaluation
    precision = precision_score(y_true_classes, y_pred_classes, average='macro')
    recall = recall_score(y_true_classes, y_pred_classes, average='macro')
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')

    # specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Negative Rate

    print()
    print("Macro Evaluation Results")
    print('Precision:', precision)
    print('Recall:', recall)
    # print('Specificity:', specificity)
    print('F1-score:', f1)


    # Overall Evaluation using average = None
    precision = precision_score(y_true_classes, y_pred_classes, average=None)
    recall = recall_score(y_true_classes, y_pred_classes, average=None)
    f1 = f1_score(y_true_classes, y_pred_classes, average=None)

    print()
    print("Average = None Evaluation Results")
    print('Precision:', precision)
    print('Recall:', recall)
    # print('Specificity:', specificity)
    print('F1-score:', f1)


def make_pdf_of_confusion_matrix(cm):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 10))
    labels = ['BENIGN', 'RAM', 'BLOCK',]
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.pdf")


def main():
    # Load the dataset
    dataset, labels = load_dataset(train_data_dir)
    dataset_2, labels_2 = load_dataset(test_data_dir)

    # Concatenate dataset_2 and labels_2 with dataset and labels
    dataset = np.concatenate((dataset, dataset_2))
    labels = np.concatenate((labels, labels_2))

    print("pre_feature_selection")
    # dataset_selected, labels_selected = feature_selection(dataset, labels)
    percentile = 70

    # dataset_selected = univariate_feature_selection(dataset, labels, percentile)
    # dataset_selected = decision_trees_based_feature_selection(dataset, labels, percentile)
    dataset_selected = random_forest_based_feature_selection(dataset, labels, percentile)
    # dataset_selected = recursive_feature_selection(dataset, labels, percentile)

    print("post_feature_selection")

    X_train, X_val, y_train, y_val = train_test_split(dataset_selected, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=42)
    print("X_train", len(X_train))
    print("X_val", len(X_val))
    print("X_test", len(X_test))


    # Convert labels to categorical format
    num_classes = 3  # Number of classes
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    # model = RNN_model(input_shape, num_classes)
    # model = GRU_model(input_shape, num_classes)
    # model = LSTM_model(input_shape, num_classes)
    # model = Bidirectional_RNN_model(input_shape, num_classes)
    model = Bidirectional_GRU_model(input_shape, num_classes)
    # model = Bidirectional_LSTM_model(input_shape, num_classes)

    # Train the model
    train_model(model, X_train, y_train, X_val, y_val, epochs=100)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()

