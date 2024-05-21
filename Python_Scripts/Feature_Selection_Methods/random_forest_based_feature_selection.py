import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
def random_forest_based_feature_selection(X, y, percentile):
    # Assuming X is your 3D dataset with shape (2814, 200, 38)
    # Print the shape of X before reshaping
    print("pre_feature_selection", X.shape)

    # Reshape it to (2814*200, 38) for feature selection
    X_reshaped = X.reshape((2814 * 200, 38))

    y_repeated = np.repeat(y, 200)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_repeated, test_size=0.2, random_state=42)

    # Use Random Forest for feature selection
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Get feature importances
    feature_importances = clf.feature_importances_

    # Select top 85% features based on importance
    selected_feature_indices = np.argsort(feature_importances)[-int((percentile / 100) * len(feature_importances)):]

    # Apply the same feature selection to the test set
    X_selected = X[:, :, selected_feature_indices]

    # Display the selected feature indices
    print("Selected feature indices:", selected_feature_indices)
    print("Number of selected features:", len(selected_feature_indices))
    print("X_selected", X_selected.shape)
    print("X_selected", X_selected[0])

    # Create a directory to save individual CSV files
    output_directory = 'selected_features_csv'
    os.makedirs(output_directory, exist_ok=True)

    # Save each instance as a separate CSV file
    for i in range(X_selected.shape[0]):
        instance_data = X_selected[i, :, :]
        instance_df = pd.DataFrame(instance_data.reshape((200, -1)))
        instance_df.to_csv(os.path.join(output_directory, f'selected_features_instance_{i + 1}.csv'), index=False)

    return X_selected
