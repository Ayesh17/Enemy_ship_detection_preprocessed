import os
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Folder structure
train_data_dir = 'train_data'
test_data_dir = 'test_data'

def load_dataset(dir, type):
    behavior_classes = ['BLOCK', 'BENIGN', 'STATIONARY', 'HEADON', 'OVERTAKE', 'CROSS', 'RAM', 'HERD']

    if type =="train":
        # Make predictions on all test files
        csv_files = [f for f in os.listdir(dir) if f.endswith('.csv')]
        csv_files = csv_files[:1000]
        for csv_file in csv_files:
            # Read the CSV files and concatenate them into a single DataFrame
            print("csv", csv_file)
            data = pd.concat([pd.read_csv(os.path.join(dir, f)) for f in csv_files])

    elif type == "test":
        data =[]
        # Make predictions on all test files
        for behavior_class in behavior_classes:
            test_dir = os.path.join(dir, behavior_class)
            csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
            csv_files = csv_files[:50]
            for csv_file in csv_files:
                # Read the CSV files and concatenate them into a single DataFrame
                csv_data = pd.concat([pd.read_csv(os.path.join(test_dir, f)) for f in csv_files])
            for row in csv_data.values:
                data.append(row)
        data = pd.DataFrame(data)
    else:
        print(" Wrong input for type")

    return data


# def load_dataset():
#     behavior_classes = ['BENIGN', 'RAM', 'BLOCK', 'HERD', 'CROSS', 'HEADON', 'OVERTAKE', 'STATIONARY']
#
#     # Make predictions on all test files
#     csv_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
#     csv_files = csv_files[:10]
#     for csv_file in csv_files:
#         # Read the CSV files and concatenate them into a single DataFrame
#         print("csv", csv_file)
#         data = pd.concat([pd.read_csv(os.path.join(train_data_dir, f)) for f in csv_files])
#     print("data", data)
#     return data

def train_test_split(data, test_size=0.2):
  """Splits the dataset into train and test sets."""
  X = data.iloc[:, :-1]
  y = data.iloc[:, -1]
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size)
  return X_train, X_test, y_train, y_test

def design_model(num_features):
    timesteps = 1
    model = Sequential()
    model.add(LSTM(32, input_shape=(timesteps, num_features)))
    model.add(Dense(8, activation='softmax'))
    return model

def design_model():
    # timesteps = 1
    # model = Sequential()
    # model.add(LSTM(32, input_shape=(timesteps, num_features)))
    # model.add(Dense(8, activation='softmax'))

    # Create a random forest classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=10)

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
    print(predictions)

    # Calculate accuracy and confusion matrix
    true_labels = y_test
    # y_pred = np.argmax(predictions, axis=1)
    y_pred = predictions

    cm = confusion_matrix(true_labels, y_pred)
    print("Confusion Matrix")
    print(cm)
    make_pdf_of_confusion_matrix(cm)
    acc = accuracy_score(true_labels, y_pred)
    return acc


#Load the dataset
data = load_dataset(train_data_dir, "train")

# Split the dataset into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(data)

# Convert the DataFrames to NumPy arrays.
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

model = design_model()

# Train the model
model.fit(X_train, y_train)

print("X_test", len(X_test))
print("y_test", len(y_test))

acc = test(X_test, y_test)
print("Overall Accuracy : ", acc)


# behavior_classes = ['BENIGN', 'RAM', 'BLOCK']
behavior_classes = ['BENIGN', 'RAM', 'BLOCK', 'HERD', 'CROSS', 'HEADON', 'OVERTAKE', 'STATIONARY']
acc_list = []

# X_test = []
# y_test = []



#Separate testing statics

#Load the dataset
data = load_dataset(test_data_dir, "test")
# data = data.to_numpy()

print("data", len(data))
# print("data_0", data)
X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values
print("X_test", len(X_test))
print("y_test", len(y_test))

acc = test(X_test, y_test)
print("Overall Accuracy : ", acc)



# Make predictions on all test files
for behavior_class in behavior_classes:
    test_dir = os.path.join(test_data_dir, behavior_class)
    csv_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    overall_accuracy = 0

    for csv_file in csv_files:
        # Read the test CSV file
        data = pd.concat([pd.read_csv(os.path.join(test_dir, f)) for f in csv_files])

        # data = pd.read_csv(os.path.join(test_dir, csv_file))
        X = data.iloc[:, :-1].values
        # X_test.append(X)
        # print("X", len(X))
        # print("X_test", len(X_test))
        predictions = model.predict(X)
        # print("pred", len(predictions))

        # Calculate accuracy and confusion matrix
        true_labels = data.iloc[:, -1].values
        # y_test.append(true_labels)
        accuracy = accuracy_score(true_labels, predictions)
        overall_accuracy += accuracy

    print("behavior", behavior_class)
    overall_acc = overall_accuracy / len(csv_files)
    print("Overall accuracy:", overall_acc)
    acc_list.append(overall_acc)

print("Overall Accuracy : ", acc_list)

