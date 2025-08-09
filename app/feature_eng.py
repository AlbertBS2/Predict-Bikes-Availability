# Preprocess the dataset
from utils.app_utils import preprocess


# Path to the dataset
path_train = "app/data/training_data_fall2024.csv"
path_test = "app/data/test_data_fall2024.csv"

# Path to save the preprocessed dataset
path_out = "app/data/"
name_out_train = "training_data_preprocessed.csv"
name_out_test = "test_data_preprocessed.csv"

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
