import os
import numpy as np
import pandas as pd
import pickle

# Folder structure
train_data_dir = 'train_data'
test_data_dir = 'test_data'

#Convert
def convert_dataset(data_dir):
    # Read all CSV files in the data directory
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print("csv", csv_files)
    dataset = []
    labels = []
    max_sequence_length = 200  # Maximum sequence length

    # Iterate over CSV files
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, csv_file))

        # Ignore CSV files with less than 200 rows
        if df.shape[0] < max_sequence_length:
            continue

        # Consider only the first 200 rows as sequence_length
        sequence_data = df.iloc[:max_sequence_length, :-1].values  # First 38 features as data
        sequence_labels = df.iloc[:max_sequence_length, -1].values.reshape(-1, 1)  # Last feature as label

        dataset.append(sequence_data)
        labels.append(sequence_labels)

    # Convert the lists to numpy arrays
    dataset = np.array(dataset)
    labels = np.array(labels)

    # Save the data as a pickle file
    data_path = 'data'  # Set the path to save the pickle file
    if data_dir == train_data_dir:
        pickle_file = os.path.join(data_path, 'train_dataset.pkl')
    elif data_dir == test_data_dir:
        pickle_file = os.path.join(data_path, 'test_dataset.pkl')
    else:
        pickle_file = os.path.join(data_path, 'other_dataset.pkl')

    with open(pickle_file, 'wb') as f:
        pickle.dump({'dataset': dataset, 'labels': labels}, f)

# Read
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
    return dataset, labels


convert_dataset(train_data_dir)
convert_dataset(test_data_dir)

loaded_dataset, loaded_labels = load_dataset(train_data_dir)
loaded_dataset, loaded_labels = load_dataset(test_data_dir)
