from sklearn.ensemble import RandomForestClassifier
from utils.app_utils import preprocess, load_model_from_json, fit_and_save_model


model = load_model_from_json(RandomForestClassifier, "app/model/best_params/rf_best_params.json")

# Preprocess the training dataset
preprocess(
    path_in="app/data/training_data_fall2024.csv",
    path_out="app/data/",
    name_out="training_data_preprocessed.csv",
)

fit_and_save_model(
    model=model,
    training_data="app/data/training_data_preprocessed.csv",
    scaler_stats="app/data/scaler_stats.json",
    target_column="increase_stock",
    class_zero='low_bike_demand',
    out_path="app/model/rf_model.joblib"
)
