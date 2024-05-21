import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def decision_trees_based_feature_selection(X, y, percentile):
    # Assuming X is your 3D dataset with shape (2814, 200, 38)
    # Print the shape of X before reshaping
    print("pre_feature_selection", X.shape)

    # Reshape it to (2814*200, 38) for feature selection
    X_reshaped = X.reshape((2814 * 200, 38))

    y_repeated = np.repeat(y, 200)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_repeated, test_size=0.2, random_state=42)

    # Use Decision Tree as the estimator for feature importance
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Get feature importances
    feature_importances = dt.feature_importances_

    # Get indices of the top features based on importance
    top_feature_indices = np.argsort(feature_importances)[::-1][:int((percentile / 100) * X_reshaped.shape[1])]

    # Apply the same feature selection to the test set
    X_selected = X[:, :, top_feature_indices]

    # Display the selected feature indices
    print("Selected feature indices:", top_feature_indices)
    print("Number of selected features:", len(top_feature_indices))
    print("X_selected", X_selected.shape)
    print("X_selected", X_selected[0])

    return X_selected
