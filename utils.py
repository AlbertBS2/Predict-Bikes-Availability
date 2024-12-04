
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import os


def find_optimal_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, save_dir="", save_file='knn_best_params.npy'):

    knn_gscv = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
    knn_gscv.fit(X_train, y_train)

    print("Best parameters found: ", knn_gscv.best_params_)

    if save_dir:
        print("Saving best parameters to '{}'".format(os.path.join(save_dir, save_file).replace('\\', '/').strip()))
        np.save(os.path.join(save_dir, save_file), knn_gscv.best_params_)
    return knn_gscv.best_params_


def load_model_from_numpy(numpy_file):
    """
    Load a model from a numpy file
    
    Parameters:
    numpy_file (str): The path to the numpy file containing the model parameters

    Returns:
    model: The model loaded from the numpy file
    """
    params = np.load(numpy_file, allow_pickle=True).item()
    model = model(**params)
    return model

def fit_and_evaluate(model, X_train, y_train, X_test, y_test, verbose=False, float_precision=3):
    """
    Runs a KNN model on the given data and returns the accuracy, precision, recall, F1, ROC AUC, and confusion matrix.
    
    Parameters:
        X_train (pd.DataFrame): The training data
        y_train (pd.Series): The training labels
        X_test (pd.DataFrame): The testing data
        y_test (pd.Series): The testing labels
        n_neighbors (int): The number of neighbors to use
        verbose (bool): Whether to print the results
        float_precision (int): The number of decimal places to print
    Returns:
        tuple: The accuracy, precision, recall, F1, ROC AUC, and confusion matrix
    """
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    if verbose:
        print(f"Accuracy: {acc:.{float_precision}f}")
        print(f"Precision: {precision:.{float_precision}f}")
        print(f"Recall: {recall:.{float_precision}f}")
        print(f"F1: {f1:.{float_precision}f}")
        print(f"ROC AUC: {roc_auc:.{float_precision}f}")
        print(f"Confusion Matrix: \n{cm}")
    
    return acc, precision, recall, f1, roc_auc, cm