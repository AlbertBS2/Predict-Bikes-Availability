# Preprocess the dataset

import pandas as pd

################################################### VARIABLES ###################################################

# Path to the dataset
path_train = "data/training_data_fall2024.csv"
path_test = "data/test_data_fall2024.csv"

# Path to save the preprocessed dataset
path_out = "data/"
name_out_train = "training_data_preprocessed.csv"
name_out_test = "test_data_preprocessed.csv"


################################################### FUNCTIONS ###################################################

def preprocess(path_in, path_out, name_out):
    """
    Preprocess the dataset and saves it as a .csv file.

    Args:
        path_in (str): path to the dataset
        path_out (str): path to save the preprocessed dataset
        name_out (str): name for the preprocessed dataset

    Returns:
        df (pd.DataFrame): preprocessed dataset
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

    # Normalize the numerical features
    for feature in num_features:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()

    # Save the preprocessed dataset as csv
    data.to_csv(path_out + name_out, index=False)

    return data


################################################### MAIN ###################################################

# Execute only if the script is run directly
if __name__ == "__main__":
    # Select which dataset to preprocess
    while True:
        try:
            selector = input("Which dataset do you want to preprocess? (train/test/both): ")
            assert selector in ["train", "test", "both"]
            break
        except AssertionError:
            print("Invalid input. Please enter 'train', 'test' or 'both'.")

    # Preprocess the training dataset
    if selector == "train":
        preprocess(path_train, path_out, name_out_train)

    # Preprocess the test dataset
    elif selector == "test":
        preprocess(path_test, path_out, name_out_test)

    elif selector == "both":
        preprocess(path_train, path_out, name_out_train)
        preprocess(path_test, path_out, name_out_test)
