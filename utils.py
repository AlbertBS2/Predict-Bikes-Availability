
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def find_optimal_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, save_dir="", save_file='knn_best_params.npy'):

    gs_cv = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    gs_cv.fit(X_train, y_train)

    print("Best parameters found: ", gs_cv.best_params_)

    if save_dir:
        print("Saving best parameters to '{}'".format(os.path.join(save_dir, save_file).replace('\\', '/').strip()))
        with open(os.path.join(save_dir, save_file), 'w') as f:
            json.dump(gs_cv.best_params_, f)
    
    return gs_cv.best_params_


def load_model_from_json(model, json_file, extra_parms={}):
    """
    Load a model from a json file
    
    Parameters:
    model: The model class to load
    json_file (str): The path to the json file
    extra_parms (dict): Extra parameters to pass to the model

    Returns:
    model: The model loaded from the json file
    """
    with open(json_file, 'r') as f:
        params = json.load(f)

    model = model(**params, **extra_parms)

    return model

def plot_roc_curves(results):
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

def fit_and_evaluate_multiple(models, X_train, y_train, X_test, y_test, verbose=False, float_precision=4):
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
        results[model.__class__.__name__] = fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose, float_precision)
    return results

def fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False, float_precision=4):
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
        tuple: The accuracy, precision, recall, F1, ROC AUC, and confusion matrix
    """
    
    model.fit(X_train, y_train)
    return evaluate(model, X_test, y_test, verbose, float_precision)

def evaluate(model, X_test, y_test, verbose=False, float_precision=4):
    """
    Evaluates a model on the given data and returns the accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
    
    Parameters:
        model: The model
        X_test (pd.DataFrame): The testing data
        y_test (pd.Series): The testing labels
        verbose (bool): Whether to print the results
        float_precision (int): The number of decimal places to print
    Returns:
        tuple: The accuracy, precision, recall, F1, ROC AUC, and confusion matrix
    """
    
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    if verbose:
        print(f"Accuracy: {acc:.{float_precision}f}")
        print(f"Precision: {precision:.{float_precision}f}")
        print(f"Recall: {recall:.{float_precision}f}")
        print(f"F1: {f1:.{float_precision}f}")
        print(f"ROC AUC: {roc_auc:.{float_precision}f}")
        print(f"Confusion Matrix: \n{cm}")
    
    return acc, precision, recall, f1, roc_auc, cm