# src/__init__.py
"""
Household Income Prediction — Source Package
=============================================
Modular codebase for data loading, preprocessing, model training, and evaluation.

Modules:
    data_loader      — Load and validate the CSV dataset
    preprocessing    — Log-transform, label encoding, feature engineering, scaling
    models           — Model registry (Linear, Ridge, Lasso, Random Forest, GB)
    evaluation       — Metrics computation, cross-validation, feature importance
    train_and_evaluate_all — End-to-end training pipeline

Best Model: Random Forest Regressor (n_estimators=300, max_depth=20)
"""
