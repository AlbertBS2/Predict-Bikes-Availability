import pandas as pd
import json


def preprocess_input_streamlit(data: dict, scaler_stats: dict) -> pd.DataFrame:
    """
    Takes the input dataset and returns a preprocessed DataFrame.

    Args:
        data (dict): Data to preprocess
        scaler_stats (dict): Dictionary containing mean and std for normalization

    Returns:
        DataFrame: Preprocessed DataFrame
    """

    # Convert input data dict to df
    df = pd.DataFrame(data)

    # Define the numerical features
    num_features = ['temp', 'dew', 'humidity', 'windspeed', 'cloudcover', 'visibility']

    # Add a binary feature called "day" where 1 means "hour_of_day" is between 7 and 20, and 0 otherwise
    df['day'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 20)).astype(int)

    # Add a binary feature called "rain" where 1 means if "precip" is greater than 0, and 0 otherwise
    df['rain'] = (df['precip'] > 0).astype(int)

    # Drop "precip" column
    df = df.drop(columns=['precip'])

    # Normalize the numerical features
    for feature in num_features:
        stats = scaler_stats[feature]
        df[feature] = (df[feature] - stats['mean']) / stats['std']

    return df