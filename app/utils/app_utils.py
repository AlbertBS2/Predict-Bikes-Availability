import json
import numpy as np
import pandas as pd
import joblib


def preprocess(path_in, path_out, name_out):
    """
    Preprocess the dataset and saves it as a csv file.

    Args:
        path_in (str): path to the dataset
        path_out (str): path to save the preprocessed dataset
        name_out (str): name for the preprocessed dataset
    """

    # Load the dataset
    data = pd.read_csv(path_in)

    # Define the numerical features
    num_features = ['temp', 'dew', 'humidity', 'windspeed', 'cloudcover', 'visibility']

    # Drop holiday and snow columns
    data = data.drop(columns=['holiday', 'snow'])

    # Add a binary feature called "day" where 1 means "hour_of_day" is between 7 and 20, and 0 otherwise
    data['day'] = ((data['hour_of_day'] >= 7) & (data['hour_of_day'] <= 20)).astype(int)

    # Encode "snowdepth" as a binary feature where 1 means if there is snow and 0 otherwise
    data['snowdepth'] = (data['snowdepth'] > 0).astype(int)

    # Add a binary feature called "rain" where 1 means if "precip" is greater than 0, and 0 otherwise
    data['rain'] = (data['precip'] > 0).astype(int)

    # Drop "precip" column
    data = data.drop(columns=['precip'])

    # Compute scaler stats
    scaler_stats = {
        feature: {
            'mean': data[feature].mean(),
            'std': data[feature].std()
        }
        for feature in num_features
    }

    # Save scaler stats to JSON or joblib
    with open(path_out + 'scaler_stats.json', 'w') as f:
        json.dump(scaler_stats, f)

    # Apply normalization
    for feature in num_features:
        mean = scaler_stats[feature]['mean']
        std = scaler_stats[feature]['std']
        data[feature] = (data[feature] - mean) / std

    # Save the preprocessed dataset as csv
    data.to_csv(path_out + name_out, index=False)


def load_model_from_json(model, json_file):
    """
    Load a model from a json file
    
    Parameters:
        model: The model class to load
        json_file (str): The path to the json file

    Returns:
        model: The model loaded from the json file
    """

    with open(json_file, 'r') as f:
        params = json.load(f)

    model = model(**params)

    return model


def fit_and_save_model(model, training_data, scaler_stats, target_column, class_zero, out_path):
    """
    Fits a model on the given data and saves the predictions on the evaluation data.
    
    Parameters:
        model: The model
        training_data (csv): Path to the training data
        scaler_stats (json): Path to the scaler stats
        target_column (str): The name of the target column to be predicted
        class_zero (str): The name of the class to be used as the reference class (0)
        out_path (str): The path to save the trained model
    """

    # Load scaler stats
    with open(scaler_stats, 'r') as f:
        scaler_stats = json.load(f)

    # Load training data
    training_data = pd.read_csv(training_data)

    # Assign 0 to the class_zero and 1 to the other class in the target column
    training_data[target_column] = np.where(training_data[target_column] == class_zero, 0, 1)

    # Split training data into features and target
    X_train = training_data.copy()
    y_train = X_train.pop(target_column)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(
        {
            'model': model,
            'scaler_stats': scaler_stats
        },
        out_path
    )