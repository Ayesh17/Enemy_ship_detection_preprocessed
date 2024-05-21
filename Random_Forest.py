import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Folder structure
train_data_dir = 'train_data'
test_data_dir = 'test_data'

# Read all CSV files in the train_data directory
csv_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]

# Combine all CSV files into a single DataFrame
all_data = pd.DataFrame()
for csv_file in csv_files:
    data = pd.read_csv(os.path.join(train_data_dir, csv_file))
    all_data = pd.concat([all_data, data])

# Extract input features and target variable
X = all_data.iloc[:, :-1].values
y = all_data.iloc[:, -1].values

# Create a random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=15)

# Train the model
rf.fit(X, y)

# behavior_classes = ['BENIGN', 'RAM', 'BLOCK']
behavior_classes = ['BENIGN', 'RAM', 'BLOCK', 'HERD', 'CROSS', 'HEADON', 'OVERTAKE', 'STATIONARY']
acc_list = []

# Make predictions on all test files
for behavior_class in behavior_classes:
    test_dir = os.path.join(test_data_dir, behavior_class)
    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    overall_accuracy = 0
    for csv_file in csv_files:
        # Read the test CSV file
        data = pd.read_csv(os.path.join(test_dir, csv_file))
        X = data.iloc[:, :-1].values
        predictions = rf.predict(X)

        # Calculate accuracy and confusion matrix
        true_labels = data.iloc[:, -1].values
        accuracy = accuracy_score(true_labels, predictions)
        overall_accuracy += accuracy

    print("behavior", behavior_class)
    overall_acc = overall_accuracy / len(csv_files)
    print("Overall accuracy:", overall_acc)
    acc_list.append(overall_acc)

print("acc_list", acc_list)

