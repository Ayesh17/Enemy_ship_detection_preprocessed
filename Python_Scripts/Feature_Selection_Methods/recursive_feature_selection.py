import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
def recursive_feature_selection(X, y, percentile, n_estimators=20):
    # Assuming X is your 3D dataset with shape (2814, 200, 38)
    # Print the shape of X before reshaping
    print("pre_feature_selection", X.shape)

    # Reshape it to (2814*200, 38) for feature selection
    X_reshaped = X.reshape((2814 * 200, 38))

    y_repeated = np.repeat(y, 200)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_repeated, test_size=0.2, random_state=42)

    # Use Random Forest as the estimator for RFE
    estimator = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

    # Initialize RFE with the chosen estimator and fit to the training data
    rfe = RFE(estimator, n_features_to_select=int((percentile / 100) * X_reshaped.shape[1]))
    X_rfe = rfe.fit_transform(X_train, y_train)

    # Apply the same feature selection to the test set
    X_selected = X[:, :, rfe.support_]

    # Display the selected feature indices
    print("Selected feature indices:", np.where(rfe.support_)[0])
    print("Number of selected features:", rfe.n_features_)
    print("X_selected", X_selected.shape)
    print("X_selected", X_selected[0])

    return X_selected
