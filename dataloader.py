import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, c, target_column, class_zero, convert_cat_target=False, test_size=0.2, random_state=0, delimiter=','):
    """
    Loads data from a CSV file, processes it, and returns train-test splits.

    Parameters:
        filepath (str): Path to the CSV file.
        target_column (str): The name of the target column to be predicted.
        class_zero (str): The name of the class to be used as the reference class (0).
        convert_cat_target (bool): Whether to convert the target column to binary.
        test_size (float): Fraction of the data to use as test set.
        random_state (int): Random seed for reproducibility.
        delimiter (str): The delimiter used in the CSV file.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """

    df = pd.read_csv(filepath, delimiter=delimiter)

    if convert_cat_target:
        # Assign 0 to the class_zero and 1 to the other class in the target column
        df[target_column] = np.where(df[target_column] == class_zero, 0, 1)
    
    # Split into features and target
    X = df.copy()
    y = X.pop(target_column)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
