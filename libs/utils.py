
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




def find_optimal_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, save_dir="", save_file='knn_best_params.json', extra_args={}, verbose_training=False):

    """
    Find the optimal hyperparameters for a model using GridSearchCV
    
    Parameters:
        model: The model class
        param_grid (dict): The hyperparameters to search over
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        cv (int): The number of cross-validation folds
        scoring (str): The scoring metric
        n_jobs (int): The number of jobs to run in parallel
        save_dir (str): The directory to save the best parameters
        save_file (str): The file to save the best parameters
        extra_args (dict): Extra arguments to pass to the model

    Returns:
        dict: The best hyperparameters found
    """
    
    model = model(**extra_args)
    gs_cv = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)

    if 'CatBoostClassifier' in str(model):
        gs_cv.fit(X_train, y_train, verbose=verbose_training)
    else:
        gs_cv.fit(X_train, y_train)

    best_params = gs_cv.best_params_
    print("Best parameters found: ", best_params)

    if extra_args:
        best_params.update(extra_args)
    
    if save_dir:
        print("Saving best parameters to '{}'".format(os.path.join(save_dir, save_file).replace('\\', '/').strip()))
        with open(os.path.join(save_dir, save_file), 'w') as f:
            json.dump(best_params, f)
    
    return best_params


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

def plot_roc_curves(results):
    """
    Plot the ROC curves for the models
    
    Parameters:
        results (dict): The results from fit_and_evaluate_multiple
    
    Preconditions:
        - results contains the fpr and tpr for each model, as well as the roc_auc
    """
    plt.figure(figsize=(10, 7))
    for model_name, metrics in results.items():
        plt.plot(metrics["fpr"], metrics["tpr"], label=f"{model_name} (AUC = {metrics['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def fit_and_evaluate_multiple(models, X_train, y_train, X_test, y_test, verbose=False, verbose_training=False, float_precision=4):
    """
    Fits multiple models on the given data and evaluates them on the testing data.
    
    Parameters:
        models (list): The models
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        X_test (pd.DataFrame): The testing data
        y_test (pd.Series): The testing labels
        verbose (bool): Whether to print the results
        float_precision (int): The number of decimal places to print

    Returns:
        dict: The accuracy, precision, recall, F1, ROC AUC, and confusion matrix for each model
    """
    
    results = {}
    for model in models:
        results[model.__class__.__name__] = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose, verbose_training, float_precision)
    return results

def fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False, verbose_training=False, float_precision=4):
    """
    Fits a model on the given data and evaluates it on the testing data.
    
    Parameters:
        model: The model
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        X_test (pd.DataFrame): The testing data
        y_test (pd.Series): The testing labels
        verbose (bool): Whether to print the results
        float_precision (int): The number of decimal places to print

    Returns:
        dict: The accuracy, precision, recall, F1, ROC AUC, confusion matrix, fpr, tpr
    """
    
    if 'CatBoostClassifier' in str(model):
        model.fit(X_train, y_train, verbose=verbose_training)
    else:
        model.fit(X_train, y_train)
    return evaluate(model, X_test, y_test, verbose, float_precision)

def evaluate(model, X_test, y_test, verbose=False, float_precision=4):
    """
    Evaluates a model on the given data and returns the accuracy, precision, recall, F1, ROC AUC, and confusion matrix, fpr, tpr as a dictionary.
    
    Parameters:
        model: The model
        X_test (pd.DataFrame): The testing data
        y_test (pd.Series): The testing labels
        verbose (bool): Whether to print the results
        float_precision (int): The number of decimal places to print
    Returns:
        dict: The accuracy, precision, recall, F1, ROC AUC, confusion matrix, fpr, tpr
    """
    
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    if verbose:
        print(f"Evaluating {model.__class__.__name__}")
        print(f"Accuracy: {acc:.{float_precision}f}")
        print(f"Precision: {precision:.{float_precision}f}")
        print(f"Recall: {recall:.{float_precision}f}")
        print(f"F1: {f1:.{float_precision}f}")
        print(f"ROC AUC: {roc_auc:.{float_precision}f}")
        print(f"Confusion Matrix: \n{cm}")
        print()
    
    results_dict = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }

    return results_dict


def fit_and_save_predictions(model, training_data, X_eval, target_column, class_zero):
    """
    Fits a model on the given data and saves the predictions on the evaluation data.
    
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
    training_data[target_column] = np.where(training_data[target_column] == class_zero, 0, 1)

    # Split training data into features and target
    X_train = training_data.copy()
    y_train = X_train.pop(target_column)

    # Load evaluation data
    X_eval = pd.read_csv(X_eval)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Compute the predictions on the evaluation data
    y_pred = model.predict(X_eval)

    # Reshape the predictions to a single row
    y_pred_row = np.reshape(y_pred, (1, -1))
    
    # Save the predictions to a CSV file
    y_pred_df = pd.DataFrame(y_pred_row)
    y_pred_df.to_csv("data/final_predictions.csv", header=False, index=False)
