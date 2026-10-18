"""
train_and_evaluate_all.py - End-to-end training and evaluation pipeline.

Split: 70% Train / 15% Validation / 15% Test
Target: Raw income (no log transform)
Model selection: Based on Validation R²

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

from sklearn.linear_model import Ridge

from src.data_loader import load_data, get_data_summary
from src.preprocessing import preprocess_pipeline, scale_features, split_data
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

    Split: 70/15/15 (Train/Validation/Test)
    No log transform — models predict raw income directly.

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
    print(f"  Income — Mean: ${df['Income'].mean():,.0f}  Median: ${df['Income'].median():,.0f}")
    print(f"  Income — Skewness: {df['Income'].skew():.4f}")

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
    print(f"  No log transform — predicting raw Income directly")

    # ──────────────────────────────────────────
    # 3. Train / Validation / Test Split
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)")
    print("=" * 70)

    X = df_processed[feature_cols]
    y = df_processed['Income']

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"  Train:      {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train.values, X_val.values, X_test.values
    )

    # ──────────────────────────────────────────
    # 4. Train & Evaluate All Models
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: MODEL TRAINING & EVALUATION (on Validation Set)")
    print("=" * 70)

    models = get_model_registry()
    results = {}
    trained_models = {}

    for name, model in models.items():
        tree = is_tree_based(name)

        # Train
        if tree:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            cv_result = cross_validate_model(model, X, y)
        else:
            model.fit(X_train_scaled, y_train)
            y_val_pred = model.predict(X_val_scaled)
            y_test_pred = model.predict(X_test_scaled)
            cv_result = cross_validate_model(model, scaler.transform(X), y)

        val_metrics = evaluate_model(y_val, y_val_pred)
        test_metrics = evaluate_model(y_test, y_test_pred)

        results[name] = {
            'Val_R2':   val_metrics['R2'],
            'Val_MAE':  val_metrics['MAE'],
            'Val_RMSE': val_metrics['RMSE'],
            'Val_MAPE': val_metrics['MAPE'],
            'Test_R2':   test_metrics['R2'],
            'Test_MAE':  test_metrics['MAE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Test_MAPE': test_metrics['MAPE'],
            'CV_R2_Mean': cv_result['CV_R2_Mean'],
            'CV_R2_Std': cv_result['CV_R2_Std'],
        }
        trained_models[name] = model

        print(f"\n  {name}:")
        print(f"    Val  R2={val_metrics['R2']:.4f}  MAE=${val_metrics['MAE']:,.0f}  RMSE=${val_metrics['RMSE']:,.0f}")
        print(f"    Test R2={test_metrics['R2']:.4f}  MAE=${test_metrics['MAE']:,.0f}  RMSE=${test_metrics['RMSE']:,.0f}")
        print(f"    CV   R2={cv_result['CV_R2_Mean']:.4f} ± {cv_result['CV_R2_Std']:.4f}")

    # ──────────────────────────────────────────
    # 5. Select Best Model (by Validation R²)
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 5: MODEL COMPARISON & SELECTION")
    print("=" * 70)

    display_cols = ['Val_R2', 'Val_MAE', 'Val_RMSE', 'Test_R2', 'Test_MAE', 'CV_R2_Mean']
    table = format_results_table(
        {k: {m: v for m, v in v.items()} for k, v in results.items()}
    )[display_cols]
    print(f"\n{table.to_string()}")

    best_name = max(results, key=lambda n: results[n]['Val_R2'])
    best_model = trained_models[best_name]
    print(f"\n  ★ Best Model (by Val R²): {best_name}")
    print(f"    Val  R2 = {results[best_name]['Val_R2']:.4f}  MAE=${results[best_name]['Val_MAE']:,.0f}")
    print(f"    Test R2 = {results[best_name]['Test_R2']:.4f}  MAE=${results[best_name]['Test_MAE']:,.0f}")
    print(f"    CV   R2 = {results[best_name]['CV_R2_Mean']:.4f} ± {results[best_name]['CV_R2_Std']:.4f}")

    # Feature importance
    fi = get_feature_importance(best_model, feature_cols)
    print("\n  Feature Importance (Top 10):")
    for feat, imp in fi.head(10).items():
        print(f"    {feat:35s} {imp:.4f}")

    # ──────────────────────────────────────────
    # 6. Save Artifacts
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 6: SAVING ARTIFACTS")
    print("=" * 70)

    models_dir = os.path.join(output_dir, 'models')
    static_dir = os.path.join(output_dir, 'static')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(models_dir, 'model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    print(f"  Saved: models/model.joblib, models/scaler.joblib")

    # Web model: Ridge (linear, easy to run in JS — no log transform)
    ridge_web = Ridge(alpha=10.0)
    ridge_web.fit(X_train_scaled, y_train)
    y_web_val = ridge_web.predict(X_val_scaled)
    web_val_metrics = evaluate_model(y_val, y_web_val)

    web_params = {
        'type': 'ridge_raw',
        'description': 'Ridge Regression on raw income (no log transform)',
        'use_log_transform': False,
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
        'web_model_name': 'Ridge Regression (Raw Income)',
        'feature_columns': feature_cols,
        'categorical_columns': cat_cols,
        'numerical_columns': [c for c in num_cols if c != 'Income'],
        'split_info': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'train_pct': 70,
            'val_pct': 15,
            'test_pct': 15,
        },
        'metrics': {k: round(v, 4) for k, v in results[best_name].items()},
        'web_model_metrics': {
            'Val_R2': round(web_val_metrics['R2'], 4),
            'Val_MAE': round(web_val_metrics['MAE'], 2),
            'Val_RMSE': round(web_val_metrics['RMSE'], 2),
        },
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
            k: {m: round(v, 4) for m, v in v.items()}
            for k, v in results.items()
        },
        'dataset_info': {
            'total_rows': len(df),
            'features': len(feature_cols),
            'categorical_features': len(cat_cols),
            'numerical_features': len(num_cols) - 1,
            'train_size': len(X_train),
            'val_size': len(X_val),
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
