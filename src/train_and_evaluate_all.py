"""
train_and_evaluate_all.py - End-to-end training and evaluation pipeline.

Usage:
    python -m src.train_and_evaluate_all
    # or
    python src/train_and_evaluate_all.py
"""

import sys
import os
import json
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from src.data_loader import load_data, get_data_summary
from src.preprocessing import preprocess_pipeline, scale_features
from src.models import get_model_registry, is_tree_based
from src.evaluation import evaluate_model, cross_validate_model, get_feature_importance, format_results_table


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def train_and_evaluate(data_path=None, output_dir=None):
    """
    Run the full training and evaluation pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to data CSV. Defaults to data/data.csv.
    output_dir : str, optional
        Directory for saving models and artifacts.
    """
    # Fix Windows console encoding
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    project_root = os.path.join(os.path.dirname(__file__), '..')
    if output_dir is None:
        output_dir = project_root

    # ──────────────────────────────────────────
    # 1. Load Data
    # ──────────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: DATA LOADING")
    print("=" * 70)

    df = load_data(data_path)
    summary = get_data_summary(df)
    print(f"  Shape: {summary['shape']}")
    print(f"  Missing: {summary['total_missing']}  |  Duplicates: {summary['duplicates']}")

    # ──────────────────────────────────────────
    # 2. Preprocessing
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: PREPROCESSING")
    print("=" * 70)

    pipeline = preprocess_pipeline(df)
    df_processed = pipeline['df']
    feature_cols = pipeline['feature_cols']
    encoding_maps = pipeline['encoding_maps']
    cat_cols = pipeline['cat_cols']
    num_cols = pipeline['num_cols']

    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Log-Income skewness: {df_processed['Income_log'].skew():.4f}")

    # ──────────────────────────────────────────
    # 3. Train/Test Split & Scaling
    # ──────────────────────────────────────────
    X = df_processed[feature_cols]
    y_log = df_processed['Income_log']

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_raw = np.expm1(y_test_log)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train.values, X_test.values)

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ──────────────────────────────────────────
    # 4. Train & Evaluate All Models
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    models = get_model_registry()
    results = {}
    trained_models = {}

    for name, model in models.items():
        tree = is_tree_based(name)

        if tree:
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            cv_result = cross_validate_model(model, X, y_log)
        else:
            model.fit(X_train_scaled, y_train_log)
            y_pred_log = model.predict(X_test_scaled)
            cv_result = cross_validate_model(model, scaler.transform(X), y_log)

        y_pred_raw = np.expm1(y_pred_log)
        metrics = evaluate_model(y_test_raw, y_pred_raw, y_test_log, y_pred_log)
        metrics.update(cv_result)

        results[name] = metrics
        trained_models[name] = model

        print(f"\n  {name}:")
        print(f"    R2(log)={metrics.get('R2_log', 0):.4f}  "
              f"CV={metrics['CV_R2_Mean']:.4f}+-{metrics['CV_R2_Std']:.4f}  "
              f"MAE=${metrics['MAE']:,.0f}")

    # ──────────────────────────────────────────
    # 5. Select Best Model
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: MODEL SELECTION")
    print("=" * 70)

    table = format_results_table(
        {k: {m: v for m, v in v.items() if m != 'cv_scores'} for k, v in results.items()}
    )
    print(f"\n{table.to_string()}")

    best_name = table.index[0]
    best_model = trained_models[best_name]
    print(f"\n  Best: {best_name}  (CV R2 = {results[best_name]['CV_R2_Mean']:.4f})")

    # Feature importance
    fi = get_feature_importance(best_model, feature_cols)
    print("\n  Feature Importance:")
    for feat, imp in fi.head(10).items():
        print(f"    {feat:35s} {imp:.4f}")

    # ──────────────────────────────────────────
    # 6. Save Artifacts
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: SAVING ARTIFACTS")
    print("=" * 70)

    models_dir = os.path.join(output_dir, 'models')
    static_dir = os.path.join(output_dir, 'static')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    print(f"  Saved: models/model.joblib, models/scaler.joblib")

    # Web model (Ridge for JS prediction)
    ridge_web = Ridge(alpha=10.0)
    ridge_web.fit(X_train_scaled, y_train_log)

    web_params = {
        'type': 'ridge_log',
        'description': 'Ridge Regression on log-transformed income',
        'use_log_transform': True,
        'coefficients': ridge_web.coef_.tolist(),
        'intercept': float(ridge_web.intercept_),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'feature_columns': feature_cols,
        'encoding_maps': encoding_maps,
        'category_options': {col: sorted(df[col].unique().tolist()) for col in cat_cols},
        'numerical_ranges': {
            col: {'min': int(df[col].min()), 'max': int(df[col].max())}
            for col in num_cols if col != 'Income'
        },
    }

    metadata = {
        'best_model_name': best_name,
        'web_model_name': 'Ridge (Log-Transformed)',
        'feature_columns': feature_cols,
        'categorical_columns': cat_cols,
        'numerical_columns': [c for c in num_cols if c != 'Income'],
        'metrics': {k: round(v, 4) for k, v in results[best_name].items() if k != 'cv_scores'},
        'feature_importance': {k: round(v, 6) for k, v in fi.to_dict().items()},
        'income_stats': {
            'mean': float(df['Income'].mean()),
            'median': float(df['Income'].median()),
            'min': float(df['Income'].min()),
            'max': float(df['Income'].max()),
            'std': float(df['Income'].std()),
            'q25': float(df['Income'].quantile(0.25)),
            'q75': float(df['Income'].quantile(0.75)),
        },
        'all_model_results': {
            k: {m: round(v, 4) for m, v in v.items() if m != 'cv_scores'}
            for k, v in results.items()
        },
        'dataset_info': {
            'total_rows': len(df),
            'features': len(feature_cols),
            'categorical_features': len(cat_cols),
            'numerical_features': len(num_cols) - 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
        },
    }

    for fname, data in [('model_metadata.json', metadata), ('web_model_params.json', web_params)]:
        path = os.path.join(static_dir, fname)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"  Saved: static/{fname}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    train_and_evaluate()
