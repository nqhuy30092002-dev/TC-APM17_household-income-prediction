"""
evaluation.py - Model evaluation utilities for household income prediction.

Provides functions to compute regression metrics, cross-validation scores,
and feature importance extraction. Works on raw (untransformed) income values.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute regression evaluation metrics on raw income values.

    Parameters
    ----------
    y_true : np.ndarray
        Actual income values.
    y_pred : np.ndarray
        Predicted income values.

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero in MAPE
    nonzero_mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)

    metrics = {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
        'MAPE': mape,
    }
    return metrics


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'r2',
) -> dict:
    """
    Perform cross-validation and return summary statistics.

    Parameters
    ----------
    model : sklearn estimator
        The model to evaluate.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector (raw income).
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric.

    Returns
    -------
    dict
        {'CV_R2_Mean': float, 'CV_R2_Std': float, 'cv_scores': np.ndarray}
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        'CV_R2_Mean': float(cv_scores.mean()),
        'CV_R2_Std': float(cv_scores.std()),
        'cv_scores': cv_scores,
    }


def get_feature_importance(model, feature_names: list[str]) -> pd.Series:
    """
    Extract feature importance from a trained model.

    For tree-based models, uses `feature_importances_`.
    For linear models, uses absolute coefficient values (normalized).

    Parameters
    ----------
    model : sklearn estimator
        A trained model.
    feature_names : list[str]
        Ordered feature names.

    Returns
    -------
    pd.Series
        Feature importance sorted descending.
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, 'coef_'):
        raw = pd.Series(np.abs(model.coef_), index=feature_names)
        importance = raw / raw.sum()
    else:
        return pd.Series(dtype=float)

    return importance.sort_values(ascending=False)


def format_results_table(results: dict) -> pd.DataFrame:
    """
    Format model results into a comparison DataFrame sorted by validation R².

    Parameters
    ----------
    results : dict
        {model_name: {metric: value, ...}, ...}

    Returns
    -------
    pd.DataFrame
        Sorted by Val_R2 descending (or R2 if Val_R2 not present).
    """
    df = pd.DataFrame(results).T
    sort_col = 'Val_R2' if 'Val_R2' in df.columns else 'R2'
    return df.sort_values(sort_col, ascending=False)
