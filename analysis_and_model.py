"""
Household Income Prediction - Data Analysis & Model Building
=============================================================
Performs EDA, trains 5 ML models (Linear, Ridge, Lasso, Random Forest,
Gradient Boosting) on log-transformed income, selects the best model
(Random Forest), and exports artifacts for the web application.

Outputs:
    models/model.joblib         - Best model (Random Forest)
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
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

# ============================================================
# 3. DATA PREPROCESSING - Use log transform for the highly skewed income
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: DATA PREPROCESSING")
print("=" * 70)

# Since Income is highly right-skewed (skewness ~3.0), use log transform
# This is a much better approach for this type of data
df_clean = df.copy()

# Log-transform income 
df_clean['Income_log'] = np.log1p(df_clean['Income'])
print(f"Log-transformed Income - Skewness: {df_clean['Income_log'].skew():.4f}")

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
y_log = df_clean['Income_log']  # Use log-transformed target
y_raw = df_clean['Income']

# ============================================================
# 5. MODEL BUILDING - Two approaches:
#    A) Predict log(Income) then exp() back
#    B) Direct prediction on raw Income
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: MODEL BUILDING & EVALUATION")
print("=" * 70)

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
y_train_raw = np.expm1(y_train_log)
y_test_raw = np.expm1(y_test_log)

print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Approach A: Log-transformed target
print("\n--- APPROACH A: Log-Transformed Income ---")
models_log = {
    'Linear (Log)': LinearRegression(),
    'Ridge (Log)': Ridge(alpha=10.0),
    'Lasso (Log)': Lasso(alpha=0.01),
    'Random Forest (Log)': RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1),
    'Gradient Boosting (Log)': GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42),
}

results = {}
trained_models = {}

for name, model in models_log.items():
    is_tree = 'Forest' in name or 'Boosting' in name
    
    if is_tree:
        model.fit(X_train, y_train_log)
        y_pred_log = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_test_scaled)
    
    # Convert back to original scale for evaluation
    y_pred_raw = np.expm1(y_pred_log)
    
    mae = mean_absolute_error(y_test_raw, y_pred_raw)
    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    r2 = r2_score(y_test_raw, y_pred_raw)
    # R2 on log scale
    r2_log = r2_score(y_test_log, y_pred_log)
    mape = np.mean(np.abs((y_test_raw - y_pred_raw) / y_test_raw)) * 100
    
    if is_tree:
        cv_scores = cross_val_score(model, X, y_log, cv=5, scoring='r2', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, scaler.transform(X), y_log, cv=5, scoring='r2')
    
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'R2_log': r2_log,
        'MAPE': mape,
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std(),
    }
    trained_models[name] = model
    
    print(f"\n  {name}:")
    print(f"    MAE:  ${mae:,.2f}")
    print(f"    RMSE: ${rmse:,.2f}")
    print(f"    R2 (raw):  {r2:.4f}")
    print(f"    R2 (log):  {r2_log:.4f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    CV R2 (log, 5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ============================================================
# 6. SELECT BEST MODEL
# ============================================================
print("\n" + "=" * 70)
print("PHASE 6: MODEL COMPARISON & SELECTION")
print("=" * 70)

results_df = pd.DataFrame(results).T.sort_values('R2_log', ascending=False)
print("\n" + results_df.to_string())

best_model_name = results_df.index[0]
best_model = trained_models[best_model_name]
is_best_tree = 'Forest' in best_model_name or 'Boosting' in best_model_name

print(f"\nBest Model: {best_model_name}")
print(f"  R2 (log scale): {results[best_model_name]['R2_log']:.4f}")
print(f"  R2 (raw scale): {results[best_model_name]['R2']:.4f}")
print(f"  MAE: ${results[best_model_name]['MAE']:,.2f}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("PHASE 7: FEATURE IMPORTANCE")
print("=" * 70)

# Use the best tree model for feature importance
feat_importance_dict = {}
for name, model in trained_models.items():
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=feature_cols)
        fi = fi.sort_values(ascending=False)
        if 'Random Forest' in name:
            feat_importance_dict = fi.to_dict()
            print(f"\nFeature Importance ({name}):")
            for feat, imp in fi.items():
                bar = "#" * int(imp * 100)
                print(f"  {feat:40s} {imp:.4f} {bar}")

# If no tree model, use coefficient magnitudes
if not feat_importance_dict:
    ridge = trained_models.get('Ridge (Log)')
    if ridge:
        fi = pd.Series(np.abs(ridge.coef_), index=feature_cols)
        fi = fi / fi.sum()
        fi = fi.sort_values(ascending=False)
        feat_importance_dict = fi.to_dict()

# ============================================================
# 8. SAVE MODEL & ARTIFACTS FOR WEB APP
# ============================================================
print("\n" + "=" * 70)
print("PHASE 8: SAVING MODEL & ARTIFACTS")
print("=" * 70)

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)
joblib.dump(best_model, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
print(f"  Saved model: models/model.joblib")
print(f"  Saved scaler: models/scaler.joblib")

# For web app: export Ridge/Linear model (can run in JS) 
# Also train a Ridge on log-scaled target for the web app
ridge_web = Ridge(alpha=10.0)
ridge_web.fit(X_train_scaled, y_train_log)

y_pred_web = np.expm1(ridge_web.predict(X_test_scaled))
web_r2 = r2_score(y_test_raw, y_pred_web)
web_mae = mean_absolute_error(y_test_raw, y_pred_web)
print(f"\n  Web model (Ridge on log-scale): R2={web_r2:.4f}, MAE=${web_mae:,.2f}")

web_model_params = {
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

# Model metadata for display
model_metadata = {
    'best_model_name': best_model_name,
    'web_model_name': 'Ridge (Log-Transformed)',
    'feature_columns': feature_cols,
    'categorical_columns': cat_cols,
    'numerical_columns': [c for c in num_cols if c != 'Income'],
    'metrics': {
        'R2': round(results[best_model_name]['R2'], 4),
        'R2_log': round(results[best_model_name]['R2_log'], 4),
        'MAE': round(results[best_model_name]['MAE'], 2),
        'RMSE': round(results[best_model_name]['RMSE'], 2),
        'MAPE': round(results[best_model_name]['MAPE'], 2),
        'CV_R2_Mean': round(results[best_model_name]['CV_R2_Mean'], 4),
    },
    'web_model_metrics': {
        'R2': round(web_r2, 4),
        'MAE': round(web_mae, 2),
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
# 9. GENERATE SAMPLE PREDICTIONS FOR VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("PHASE 9: VALIDATION SAMPLES")
print("=" * 70)

# Test web model on some samples
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

print("\nSample Predictions:")
for i, sample in enumerate(sample_data):
    # Build feature vector
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
    pred_log = ridge_web.predict(features_scaled)[0]
    pred_income = np.expm1(pred_log)
    print(f"  Sample {i+1}: ${pred_income:,.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nBest model: {best_model_name}")
print(f"Ready to build web application!")
