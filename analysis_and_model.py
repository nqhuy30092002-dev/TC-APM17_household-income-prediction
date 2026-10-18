"""
Household Income Prediction - Data Analysis & Model Building
=============================================================
Performs EDA, trains 5 ML models (Linear, Ridge, Lasso, Random Forest,
Gradient Boosting) directly on raw income (no log transform), uses a
70/15/15 Train/Validation/Test split, selects the best model by
Validation R², and exports artifacts for the web application.

Outputs:
    models/model.joblib         - Best model
    models/scaler.joblib        - Fitted StandardScaler
    static/model_metadata.json  - Metrics, feature importance, income stats
    static/web_model_params.json - Ridge coefficients for JS prediction
"""


import pandas as pd
import numpy as np
import json
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. LOAD & EXPLORE DATA
# ============================================================
print("=" * 70)
print("PHASE 1: DATA LOADING & EXPLORATION")
print("=" * 70)

df = pd.read_csv('data/data.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# ============================================================
# 2. EDA
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['number']).columns.tolist()
print(f"\nCategorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")

print(f"\n--- Income Distribution ---")
print(f"Mean: ${df['Income'].mean():,.2f}")
print(f"Median: ${df['Income'].median():,.2f}")
print(f"Std: ${df['Income'].std():,.2f}")
print(f"Skewness: {df['Income'].skew():.4f}")
print(f"Min: ${df['Income'].min():,.2f}  Max: ${df['Income'].max():,.2f}")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: DATA PREPROCESSING (No log transform)")
print("=" * 70)

df_clean = df.copy()

# Encode categorical variables
label_encoders = {}
encoding_maps = {}

for col in cat_cols:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    encoding_maps[col] = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
    print(f"  Encoded {col}: {encoding_maps[col]}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: FEATURE ENGINEERING")
print("=" * 70)

# Create additional features
df_clean['Age_WorkExp_Ratio'] = df_clean['Age'] / (df_clean['Work_Experience'] + 1)
df_clean['Dependents_per_HHSize'] = df_clean['Number_of_Dependents'] / df_clean['Household_Size']

feature_cols = [c for c in num_cols if c != 'Income']
for col in cat_cols:
    feature_cols.append(col + '_encoded')
# Add engineered features
feature_cols.extend(['Age_WorkExp_Ratio', 'Dependents_per_HHSize'])

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

X = df_clean[feature_cols]
y = df_clean['Income']  # Raw income — no log transform

# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)")
print("=" * 70)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
# Second split: 50% of temp = 15% val, 50% of temp = 15% test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

print(f"Train set:      {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 6. MODEL BUILDING — Predict raw income directly
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: MODEL BUILDING & EVALUATION (Raw Income, No Log Transform)")
print("=" * 70)

models_config = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
}

results = {}
trained_models = {}


def calc_metrics(y_true, y_pred, prefix=''):
    """Compute MAE, RMSE, R2, MAPE for a given split."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    return {
        f'{prefix}MAE': mae, f'{prefix}RMSE': rmse,
        f'{prefix}R2': r2, f'{prefix}MAPE': mape,
    }


for name, model in models_config.items():
    is_tree = 'Forest' in name or 'Boosting' in name

    if is_tree:
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    else:
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        y_test_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='r2')

    val_m = calc_metrics(y_val.values, y_val_pred, 'Val_')
    test_m = calc_metrics(y_test.values, y_test_pred, 'Test_')

    results[name] = {
        **val_m, **test_m,
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std(),
    }
    trained_models[name] = model

    print(f"\n  {name}:")
    print(f"    Val  R2={val_m['Val_R2']:.4f}  MAE=${val_m['Val_MAE']:,.0f}  RMSE=${val_m['Val_RMSE']:,.0f}  MAPE={val_m['Val_MAPE']:.1f}%")
    print(f"    Test R2={test_m['Test_R2']:.4f}  MAE=${test_m['Test_MAE']:,.0f}  RMSE=${test_m['Test_RMSE']:,.0f}  MAPE={test_m['Test_MAPE']:.1f}%")
    print(f"    CV   R2={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 7. SELECT BEST MODEL (by Validation R²)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: MODEL COMPARISON & SELECTION")
print("=" * 70)

results_df = pd.DataFrame(results).T.sort_values('Val_R2', ascending=False)
print("\n" + results_df[['Val_R2', 'Val_MAE', 'Val_RMSE', 'Test_R2', 'Test_MAE', 'CV_R2_Mean']].to_string())

best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]
is_best_tree = 'Forest' in best_model_name or 'Boosting' in best_model_name

print(f"\n★ Best Model: {best_model_name}")
print(f"  Val  R2: {results[best_model_name]['Val_R2']:.4f}")
print(f"  Test R2: {results[best_model_name]['Test_R2']:.4f}")
print(f"  Val  MAE: ${results[best_model_name]['Val_MAE']:,.2f}")
print(f"  Test MAE: ${results[best_model_name]['Test_MAE']:,.2f}")
print(f"  CV R2: {results[best_model_name]['CV_R2_Mean']:.4f} ± {results[best_model_name]['CV_R2_Std']:.4f}")

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("PHASE 8: FEATURE IMPORTANCE")
print("=" * 70)

feat_importance_dict = {}
for name, model in trained_models.items():
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        fi = fi.sort_values(ascending=False)
        if name == best_model_name:
            feat_importance_dict = fi.to_dict()
            print(f"\nFeature Importance ({name}):")
            for feat, imp in fi.items():
                bar = "#" * int(imp * 100)
                print(f"  {feat:40s} {imp:.4f} {bar}")

# Fallback: use Ridge coefficients
if not feat_importance_dict:
    ridge = trained_models.get('Ridge Regression')
    if ridge:
        fi = pd.Series(np.abs(ridge.coef_), index=feature_cols)
        fi = fi / fi.sum()
        fi = fi.sort_values(ascending=False)
        feat_importance_dict = fi.to_dict()
        print("\nFeature Importance (Ridge — normalized |coef|):")
        for feat, imp in fi.items():
            print(f"  {feat:40s} {imp:.4f}")

# ============================================================
# 9. SAVE MODEL & ARTIFACTS FOR WEB APP
# ============================================================
print("\n" + "=" * 70)
print("PHASE 9: SAVING MODEL & ARTIFACTS")
print("=" * 70)

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)
joblib.dump(best_model, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print(f"  Saved model: models/model.joblib")
print(f"  Saved scaler: models/scaler.joblib")

# Web model: Ridge on raw income (can run in JS — no log/exp needed)
ridge_web = Ridge(alpha=10.0)
ridge_web.fit(X_train_scaled, y_train)
y_web_val = ridge_web.predict(X_val_scaled)
web_r2 = r2_score(y_val, y_web_val)
web_mae = mean_absolute_error(y_val, y_web_val)
print(f"\n  Web model (Ridge, raw income): Val R2={web_r2:.4f}, Val MAE=${web_mae:,.2f}")

web_model_params = {
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

# Model metadata for display
model_metadata = {
    'best_model_name': best_model_name,
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
    'metrics': {
        'Val_R2':   round(results[best_model_name]['Val_R2'], 4),
        'Val_MAE':  round(results[best_model_name]['Val_MAE'], 2),
        'Val_RMSE': round(results[best_model_name]['Val_RMSE'], 2),
        'Val_MAPE': round(results[best_model_name]['Val_MAPE'], 2),
        'Test_R2':   round(results[best_model_name]['Test_R2'], 4),
        'Test_MAE':  round(results[best_model_name]['Test_MAE'], 2),
        'Test_RMSE': round(results[best_model_name]['Test_RMSE'], 2),
        'Test_MAPE': round(results[best_model_name]['Test_MAPE'], 2),
        'CV_R2_Mean': round(results[best_model_name]['CV_R2_Mean'], 4),
        'CV_R2_Std':  round(results[best_model_name]['CV_R2_Std'], 4),
    },
    'web_model_metrics': {
        'Val_R2': round(web_r2, 4),
        'Val_MAE': round(web_mae, 2),
    },
    'feature_importance': {k: round(v, 6) for k, v in feat_importance_dict.items()},
    'income_stats': {
        'mean': float(round(df['Income'].mean(), 2)),
        'median': float(round(df['Income'].median(), 2)),
        'min': float(round(df['Income'].min(), 2)),
        'max': float(round(df['Income'].max(), 2)),
        'std': float(round(df['Income'].std(), 2)),
        'q25': float(round(df['Income'].quantile(0.25), 2)),
        'q75': float(round(df['Income'].quantile(0.75), 2)),
    },
    'all_model_results': {
        name: {k: round(v, 4) for k, v in metrics.items()}
        for name, metrics in results.items()
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

with open('static/model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

with open('static/web_model_params.json', 'w', encoding='utf-8') as f:
    json.dump(web_model_params, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

print(f"  Saved metadata: model_metadata.json")
print(f"  Saved web params: web_model_params.json")

# ============================================================
# 10. GENERATE SAMPLE PREDICTIONS FOR VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 10: VALIDATION SAMPLES")
print("=" * 70)

sample_data = [
    {'Age': 35, 'Number_of_Dependents': 2, 'Work_Experience': 10, 'Household_Size': 4,
     'Education_Level': "Bachelor's", 'Occupation': 'Technology', 'Location': 'Urban',
     'Marital_Status': 'Married', 'Employment_Status': 'Full-time',
     'Homeownership_Status': 'Own', 'Type_of_Housing': 'Single-family home',
     'Gender': 'Male', 'Primary_Mode_of_Transportation': 'Car'},
    {'Age': 25, 'Number_of_Dependents': 0, 'Work_Experience': 3, 'Household_Size': 1,
     'Education_Level': 'High School', 'Occupation': 'Others', 'Location': 'Rural',
     'Marital_Status': 'Single', 'Employment_Status': 'Part-time',
     'Homeownership_Status': 'Rent', 'Type_of_Housing': 'Apartment',
     'Gender': 'Female', 'Primary_Mode_of_Transportation': 'Public transit'},
    {'Age': 55, 'Number_of_Dependents': 3, 'Work_Experience': 30, 'Household_Size': 5,
     'Education_Level': 'Doctorate', 'Occupation': 'Finance', 'Location': 'Urban',
     'Marital_Status': 'Married', 'Employment_Status': 'Self-employed',
     'Homeownership_Status': 'Own', 'Type_of_Housing': 'Single-family home',
     'Gender': 'Male', 'Primary_Mode_of_Transportation': 'Car'},
]

print("\nSample Predictions (Web Ridge model — raw income):")
for i, sample in enumerate(sample_data):
    features = []
    for col in feature_cols:
        if col.endswith('_encoded'):
            original_col = col.replace('_encoded', '')
            features.append(encoding_maps[original_col][sample[original_col]])
        elif col == 'Age_WorkExp_Ratio':
            features.append(sample['Age'] / (sample['Work_Experience'] + 1))
        elif col == 'Dependents_per_HHSize':
            features.append(sample['Number_of_Dependents'] / sample['Household_Size'])
        else:
            features.append(sample[col])

    features_scaled = scaler.transform([features])
    pred_income = ridge_web.predict(features_scaled)[0]
    print(f"  Sample {i+1}: ${pred_income:,.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nBest model: {best_model_name}")
print(f"Val R²: {results[best_model_name]['Val_R2']:.4f}")
print(f"Test R²: {results[best_model_name]['Test_R2']:.4f}")
print(f"Ready to build web application!")
