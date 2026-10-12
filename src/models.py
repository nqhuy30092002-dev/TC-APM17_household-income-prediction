"""
models.py - Model definitions for household income prediction.

Provides a registry of scikit-learn regression models with pre-configured
hyperparameters for predicting log-transformed household income.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def get_model_registry() -> dict:
    """
    Return a dictionary of named models with tuned hyperparameters.

    Returns
    -------
    dict
        {model_name: sklearn estimator instance}
    """
    return {
        'Linear (Log)': LinearRegression(),
        'Ridge (Log)': Ridge(alpha=10.0),
        'Lasso (Log)': Lasso(alpha=0.01),
        'Random Forest (Log)': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        'Gradient Boosting (Log)': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
    }


def is_tree_based(model_name: str) -> bool:
    """
    Check whether a model is tree-based (does not require feature scaling).

    Parameters
    ----------
    model_name : str
        Name of the model.

    Returns
    -------
    bool
    """
    return 'Forest' in model_name or 'Boosting' in model_name
