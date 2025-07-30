import pandas as pd


def preprocess_input_streamlit(data: dict) -> pd.DataFrame:
    """
    Takes the input dataset and returns a preprocessed DataFrame.

    Args:
        data (dict): Data to preprocess

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
        df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

    return df