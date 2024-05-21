import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Folder structure
train_data_dir = 'train_data'
test_data_dir = 'test_data'

# Set random seed for reproducibility
np.random.seed(42)

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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    cm = confusion_matrix(y_test, y_pred)
    make_pdf_of_confusion_matrix(cm)

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
    dataset, labels = load_dataset(train_data_dir)
    dataset_2, labels_2 = load_dataset(test_data_dir)

    # Concatenate dataset_2 and labels_2 with dataset and labels
    dataset = np.concatenate((dataset, dataset_2))
    labels = np.concatenate((labels, labels_2))

    # Flatten the frames dimension
    num_samples, num_frames, num_features = dataset.shape
    dataset = dataset.reshape(num_samples * num_frames, num_features)
    labels = np.repeat(labels, num_frames, axis=0)  # Repeat labels for each frame

    print("dataset", len(dataset))
    print("labels", len(labels))

    print("dataset", dataset)
    print("labels", labels)

    X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.3, random_state=42)
    print("X_train", len(X_train))
    print("X_val", len(X_val))

    # Flatten the time steps dimension
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    print("X_train_flat", len(X_train_flat[0]))
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # Create the Gradient Boosting model
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train_flat, y_train)

    # Evaluate the model
    evaluate_model(model, X_val_flat, y_val)


if __name__ == '__main__':
    main()
