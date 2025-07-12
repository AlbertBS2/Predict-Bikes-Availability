# Frontend code for the Streamlit app
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import calendar
from sklearn.ensemble import RandomForestClassifier
from libs.utils import load_model_from_json, fit_and_evaluate
from libs.dataloader import load_and_split_data


st.title("Do we need more bikes in Washington DC?")

# Load random forest model from json file with optimal hyperparameters
rf_model = load_model_from_json(RandomForestClassifier, 'output/best_params/rf_best_params.json')

# Load data
X_train, X_test, y_train, y_test = load_and_split_data("data/training_data_preprocessed.csv", 
                                                       target_column='increase_stock', 
                                                       class_zero='low_bike_demand', 
                                                       test_size=0.2, 
                                                       random_state=0)


def obtain_predictions(model, training_data, X_eval):
    """
    Obtains the predictions on the evaluation data.
    
    Parameters:
        model: The model
        training_data (csv): Path to the training data
        X_eval (csv): Path to the evaluation data
        target_column (str): The name of the target column to be predicted
        class_zero (str): The name of the class to be used as the reference class (0)
    """

    # Load training data
    training_data = pd.read_csv(training_data)

    # Assign 0 to the class_zero and 1 to the other class in the target column
    training_data['increase_stock'] = np.where(training_data['increase_stock'] == 'low_bike_demand', 0, 1)

    # Split training data into features and target
    X_train = training_data.copy()
    y_train = X_train.pop('increase_stock')

    # Load evaluation data
    X_eval = pd.read_csv(X_eval)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Compute the predictions on the evaluation data
    y_pred = model.predict(X_eval)

    # Convert predictions to a DataFrame
    pred_df = X_eval.copy()
    pred_df = pred_df.drop(columns=['weekday', 'summertime', 'day'])
    pred_df["Bike demand"] = np.where(y_pred == 0, "Low", 'High')
    
    return pred_df

# with st.expander("Month"):
#     this_month = datetime.date.today().month
#     month_abbr = calendar.month_abbr[1:]
#     month_str = st.radio("", month_abbr, index=this_month - 1, horizontal=True)
#     month = month_abbr.index(month_str) + 1


# Obtain predictions using the trained model
predictions = obtain_predictions(
    model=rf_model,
    training_data="data/training_data_preprocessed.csv",
    X_eval="data/test_data_preprocessed.csv"
)

st.dataframe(predictions, hide_index=True)
