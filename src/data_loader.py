"""
data_loader.py - Load and validate the household income dataset.
"""

import pandas as pd
import os


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the household income dataset from CSV.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'data/data.csv'.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    if filepath is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv'),
            os.path.join(os.getcwd(), 'data', 'data.csv'),
            os.path.join(os.getcwd(), 'data.csv'),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                filepath = candidate
                break

    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found. Looked in: {candidates if filepath is None else filepath}"
        )

    df = pd.read_csv(filepath)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a summary of the dataset's structure and quality.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.

    Returns
    -------
    dict
        Summary statistics including shape, dtypes, missing values, and duplicates.
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'numerical_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
    }
    return summary


if __name__ == '__main__':
    df = load_data()
    summary = get_data_summary(df)
    print(f"Dataset loaded: {summary['shape']}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['total_missing']}")
    print(f"Duplicates: {summary['duplicates']}")
