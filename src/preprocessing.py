"""
preprocessing.py - Data preprocessing pipeline for household income prediction.

Handles:
- Label encoding of categorical features
- Feature engineering
- Feature scaling
- Train/Validation/Test split (70/15/15)

Note: Log transform has been removed. Models are trained directly on raw income.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def encode_categorical(
    df: pd.DataFrame,
    cat_cols: list[str],
    encoders: dict | None = None
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Label-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cat_cols : list[str]
        List of categorical column names to encode.
    encoders : dict or None
        Pre-fitted encoders (for transform-only mode). If None, fit new ones.

    Returns
    -------
    tuple
        (encoded_df, label_encoders, encoding_maps)
    """
    df = df.copy()
    label_encoders = encoders or {}
    encoding_maps = {}

    for col in cat_cols:
        if col not in label_encoders:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[f'{col}_encoded'] = le.transform(df[col])

        encoding_maps[col] = dict(
            zip(le.classes_.tolist(), le.transform(le.classes_).tolist())
        )

    return df, label_encoders, encoding_maps


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from existing columns.

    New features:
    - Age_WorkExp_Ratio: Age / (Work_Experience + 1)
    - Dependents_per_HHSize: Number_of_Dependents / Household_Size

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with new engineered feature columns.
    """
    df = df.copy()
    df['Age_WorkExp_Ratio'] = df['Age'] / (df['Work_Experience'] + 1)
    df['Dependents_per_HHSize'] = df['Number_of_Dependents'] / df['Household_Size']
    return df


def build_feature_columns(num_cols: list[str], cat_cols: list[str]) -> list[str]:
    """
    Build the ordered list of feature column names for model training.

    Parameters
    ----------
    num_cols : list[str]
        Numerical column names (excluding target).
    cat_cols : list[str]
        Categorical column names.

    Returns
    -------
    list[str]
        Ordered feature column names.
    """
    feature_cols = [c for c in num_cols if c != 'Income']
    for col in cat_cols:
        feature_cols.append(f'{col}_encoded')
    feature_cols.extend(['Age_WorkExp_Ratio', 'Dependents_per_HHSize'])
    return feature_cols


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    X_test: np.ndarray = None,
    scaler: StandardScaler = None
) -> tuple:
    """
    Fit and/or transform features using StandardScaler.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_val : np.ndarray, optional
        Validation features to transform.
    X_test : np.ndarray, optional
        Test features to transform.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, fit a new one on X_train.

    Returns
    -------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)

    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def split_data(X: pd.DataFrame, y: pd.Series, train_size=0.70, val_size=0.15, random_state=42):
    """
    Split data into train / validation / test sets (70/15/15 by default).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    train_size : float
        Fraction for training (default 0.70).
    val_size : float
        Fraction for validation (default 0.15).
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    remaining = 1.0 - train_size
    val_of_remaining = val_size / remaining

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=remaining, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_of_remaining), random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_pipeline(df: pd.DataFrame) -> dict:
    """
    Run the full preprocessing pipeline (without log transform).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    dict
        Dictionary containing processed data and fitted transformers.
    """
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # 1. Encode categoricals
    df_processed, label_encoders, encoding_maps = encode_categorical(df, cat_cols)

    # 2. Feature engineering
    df_processed = engineer_features(df_processed)

    # 3. Build feature list
    feature_cols = build_feature_columns(num_cols, cat_cols)

    return {
        'df': df_processed,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols,
        'label_encoders': label_encoders,
        'encoding_maps': encoding_maps,
    }
