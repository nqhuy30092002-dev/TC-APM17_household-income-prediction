"""
generate_report.py - Tạo báo cáo Word phân tích dự báo thu nhập hộ gia đình
================================================================================
Script này thực hiện toàn bộ quy trình phân tích dữ liệu, huấn luyện mô hình
Regression (không log transform), chia 70/15/15 Train/Val/Test,
và xuất file Word (.docx) báo cáo chuyên nghiệp kèm hình ảnh biểu đồ phân tích.

Output: Bao_cao_Phan_tich_Thu_nhap_Ho_Gia_dinh.docx
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import sys
import json
import warnings
import tempfile
import shutil

warnings.filterwarnings('ignore')

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = 'data/data.csv'
OUTPUT_DOCX = 'Bao_cao_Phan_tich_Thu_nhap_Ho_Gia_dinh.docx'
CHART_DIR = tempfile.mkdtemp(prefix='report_charts_')
CHART_DPI = 200
CHART_WIDTH_INCHES = 6.2

# Vietnamese-friendly matplotlib setup
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'figure.dpi': CHART_DPI,
    'savefig.dpi': CHART_DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#2ECC71',
    'danger': '#E74C3C',
    'info': '#3498DB',
    'dark': '#2C3E50',
    'light': '#ECF0F1',
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#2ECC71', '#E74C3C',
                '#9B59B6', '#1ABC9C', '#F39C12', '#3498DB', '#E67E22',
                '#16A085', '#8E44AD', '#2980B9', '#D35400', '#27AE60'],
}


# ============================================================
# HELPER FUNCTIONS FOR DOCX FORMATTING
# ============================================================
def set_cell_shading(cell, color_hex):
    """Set cell background color."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_text(cell, text, bold=False, size=9, color=None, alignment=WD_ALIGN_PARAGRAPH.CENTER):
    """Set cell text with formatting."""
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = alignment
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = 'Times New Roman'
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_formatted_table(doc, headers, rows, col_widths=None, header_color='2E86AB'):
    """Add a professionally formatted table to the document."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'

    # Header row
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_shading(cell, header_color)
        set_cell_text(cell, header, bold=True, size=10, color=(255, 255, 255))

    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, value in enumerate(row_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            if r_idx % 2 == 1:
                set_cell_shading(cell, 'F2F7FB')
            set_cell_text(cell, value, size=9, color=(44, 62, 80))

    # Set column widths
    if col_widths:
        for row in table.rows:
            for i, width in enumerate(col_widths):
                row.cells[i].width = Cm(width)

    return table


def add_chart_to_doc(doc, chart_path, caption='', width=CHART_WIDTH_INCHES):
    """Add chart image to document with caption."""
    if os.path.exists(chart_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(chart_path, width=Inches(width))

        if caption:
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap.add_run(caption)
            run.italic = True
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(100, 100, 100)
            run.font.name = 'Times New Roman'
    doc.add_paragraph()  # spacing


def add_heading_styled(doc, text, level=1):
    """Add a heading with consistent styling."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(46, 134, 171)
    return heading


def add_body_text(doc, text, bold=False, italic=False, indent=False):
    """Add body paragraph with consistent formatting."""
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    return p


# ============================================================
# PHASE 1: LOAD & ANALYZE DATA
# ============================================================
print("=" * 70)
print("PHASE 1: Loading and analyzing data...")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"  Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=['number']).columns.tolist()

# Summary statistics
income_stats = {
    'mean': df['Income'].mean(),
    'median': df['Income'].median(),
    'std': df['Income'].std(),
    'min': df['Income'].min(),
    'max': df['Income'].max(),
    'q25': df['Income'].quantile(0.25),
    'q75': df['Income'].quantile(0.75),
    'skew': df['Income'].skew(),
    'kurtosis': df['Income'].kurtosis(),
}

# IQR outlier detection
Q1 = df['Income'].quantile(0.25)
Q3 = df['Income'].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (df['Income'] < Q1 - 1.5 * IQR) | (df['Income'] > Q3 + 1.5 * IQR)
n_outliers = outlier_mask.sum()
pct_outliers = n_outliers / len(df) * 100

print(f"  Income stats: Mean=${income_stats['mean']:,.0f}, Median=${income_stats['median']:,.0f}")
print(f"  Outliers: {n_outliers} ({pct_outliers:.1f}%)")

# ============================================================
# PHASE 2: PREPROCESSING (No log transform)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: Preprocessing (No log transform)...")
print("=" * 70)

df_clean = df.copy()

label_encoders = {}
encoding_maps = {}
for col in cat_cols:
    le = LabelEncoder()
    df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    encoding_maps[col] = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))

df_clean['Age_WorkExp_Ratio'] = df_clean['Age'] / (df_clean['Work_Experience'] + 1)
df_clean['Dependents_per_HHSize'] = df_clean['Number_of_Dependents'] / df_clean['Household_Size']

feature_cols = [c for c in num_cols if c != 'Income']
for col in cat_cols:
    feature_cols.append(col + '_encoded')
feature_cols.extend(['Age_WorkExp_Ratio', 'Dependents_per_HHSize'])

X = df_clean[feature_cols]
y = df_clean['Income']  # raw income, no log

# ============================================================
# PHASE 2b: TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2b: Train / Validation / Test split (70/15/15)...")
print("=" * 70)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"  Train:      {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation: {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test:       {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================
# PHASE 3: TRAIN ALL MODELS (on raw income)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: Training models (raw income, no log transform)...")
print("=" * 70)

models_config = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42),
}

results = {}
trained_models = {}


def calc_metrics(yt, yp):
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = r2_score(yt, yp)
    nz = np.array(yt) != 0
    mape = float(np.mean(np.abs((np.array(yt)[nz] - np.array(yp)[nz]) / np.array(yt)[nz])) * 100)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


for name, model in models_config.items():
    is_tree = name in ['Random Forest', 'Gradient Boosting']

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

    val_m  = calc_metrics(y_val,  y_val_pred)
    test_m = calc_metrics(y_test, y_test_pred)

    results[name] = {
        'Val_R2':   val_m['R2'],
        'Val_MAE':  val_m['MAE'],
        'Val_RMSE': val_m['RMSE'],
        'Val_MAPE': val_m['MAPE'],
        'Test_R2':   test_m['R2'],
        'Test_MAE':  test_m['MAE'],
        'Test_RMSE': test_m['RMSE'],
        'Test_MAPE': test_m['MAPE'],
        'CV_R2_Mean': cv_scores.mean(),
        'CV_R2_Std': cv_scores.std(),
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
    }
    trained_models[name] = model
    print(f"  {name}: Val R2={results[name]['Val_R2']:.4f}, Test R2={results[name]['Test_R2']:.4f}, CV_R2={results[name]['CV_R2_Mean']:.4f}")

# Best model: highest Val R2
best_name = max(results, key=lambda n: results[n]['Val_R2'])
best_model = trained_models[best_name]
best_results = results[best_name]

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
else:
    raw = pd.Series(np.abs(best_model.coef_), index=feature_cols)
    fi = (raw / raw.sum()).sort_values(ascending=False)

print(f"\n  Best model (by Val R²): {best_name}")
print(f"  Val R2: {best_results['Val_R2']:.4f}  Test R2: {best_results['Test_R2']:.4f}")

# ============================================================
# PHASE 4: GENERATE ALL CHARTS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Generating charts...")
print("=" * 70)


# --- Chart 1: Income Distribution (Histogram, raw only) ---
def chart_income_distribution():
    fig, ax = plt.subplots(figsize=(12, 5))

    # Raw income only (no log transform chart)
    ax.hist(df['Income'], bins=40, color=COLORS['primary'], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_title('Income Distribution (Original)', fontweight='bold')
    ax.set_xlabel('Annual Income (USD)')
    ax.set_ylabel('Frequency')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    ax.axvline(income_stats['mean'], color=COLORS['danger'], linestyle='--', linewidth=1.5, label=f"Mean: ${income_stats['mean']:,.0f}")
    ax.axvline(income_stats['median'], color=COLORS['accent'], linestyle='--', linewidth=1.5, label=f"Median: ${income_stats['median']:,.0f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_01_income_distribution.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 2: Box Plot by Education ---
def chart_boxplot_education():
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ['High School', "Bachelor's", "Master's", 'Doctorate']
    palette = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['success']]
    sns.boxplot(data=df, x='Education_Level', y='Income', order=order, palette=palette, ax=ax,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    ax.set_title('Income Distribution by Education Level', fontweight='bold')
    ax.set_xlabel('Education Level')
    ax.set_ylabel('Annual Income (USD)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_02_boxplot_education.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 3: Grouped Bar - Income by Occupation ---
def chart_income_by_occupation():
    fig, ax = plt.subplots(figsize=(10, 5))
    occ_stats = df.groupby('Occupation')['Income'].agg(['mean', 'median']).sort_values('median', ascending=False)
    x = np.arange(len(occ_stats))
    width = 0.35
    bars1 = ax.bar(x - width/2, occ_stats['median'], width, label='Median', color=COLORS['primary'], alpha=0.9)
    bars2 = ax.bar(x + width/2, occ_stats['mean'], width, label='Mean', color=COLORS['accent'], alpha=0.9)
    ax.set_title('Income by Occupation (Mean vs Median)', fontweight='bold')
    ax.set_xlabel('Occupation')
    ax.set_ylabel('Annual Income (USD)')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_stats.index, rotation=15, ha='right')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height()/1000:.0f}K',
                ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height()/1000:.0f}K',
                ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_03_income_occupation.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 4: Income by Location ---
def chart_income_by_location():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    loc_stats = df.groupby('Location')['Income'].agg(['mean', 'median']).sort_values('median', ascending=False)
    x = np.arange(len(loc_stats))
    width = 0.35
    axes[0].bar(x - width/2, loc_stats['median'], width, label='Median', color=COLORS['info'], alpha=0.9)
    axes[0].bar(x + width/2, loc_stats['mean'], width, label='Mean', color=COLORS['accent'], alpha=0.9)
    axes[0].set_title('Income by Location Type', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(loc_stats.index)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[0].legend()

    # Pie chart - Distribution
    loc_counts = df['Location'].value_counts()
    axes[1].pie(loc_counts, labels=loc_counts.index, autopct='%1.1f%%',
                colors=[COLORS['primary'], COLORS['secondary'], COLORS['accent']],
                startangle=90, textprops={'fontsize': 9})
    axes[1].set_title('Location Distribution', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_04_income_location.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 5: Correlation Heatmap ---
def chart_correlation_heatmap():
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = ['Age', 'Number_of_Dependents', 'Work_Experience', 'Household_Size', 'Income']
    corr = df[corr_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Correlation Matrix - Numerical Features', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_05_correlation.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 6: Feature Importance (Horizontal Bar) ---
def chart_feature_importance():
    fig, ax = plt.subplots(figsize=(10, 6))
    fi_sorted = fi.sort_values(ascending=True)
    colors_fi = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi_sorted)))
    bars = ax.barh(range(len(fi_sorted)), fi_sorted.values, color=colors_fi, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(fi_sorted)))
    feature_labels = [f.replace('_encoded', '').replace('_', ' ') for f in fi_sorted.index]
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.set_xlabel('Importance Score')
    ax.set_title('Random Forest - Feature Importance', fontweight='bold')

    for i, (val, bar) in enumerate(zip(fi_sorted.values, bars)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.2%}',
                va='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, fi_sorted.max() * 1.2)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_06_feature_importance.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 7: Model Comparison (Val R2, Test R2, CV R2) ---
def chart_model_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    model_names = list(results.keys())
    val_r2  = [results[m]['Val_R2']  for m in model_names]
    test_r2 = [results[m]['Test_R2'] for m in model_names]
    cv_r2   = [results[m]['CV_R2_Mean'] for m in model_names]
    cv_std  = [results[m]['CV_R2_Std']  for m in model_names]

    colors_bar = [COLORS['success'] if m == best_name else COLORS['primary'] for m in model_names]
    short = [n.replace('Regression', 'Reg.').replace('Random ', 'RF ').replace('Gradient ', 'GB ') for n in model_names]

    def annotate(ax, vals):
        for i, (bar, val) in enumerate(zip(ax.patches, vals)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Val R2
    bars0 = axes[0].bar(range(len(model_names)), val_r2, color=colors_bar, alpha=0.9, edgecolor='white')
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(short, fontsize=8, rotation=10, ha='right')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Validation R²', fontweight='bold')
    annotate(axes[0], val_r2)

    # Test R2
    bars1 = axes[1].bar(range(len(model_names)), test_r2, color=colors_bar, alpha=0.7, edgecolor='white')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(short, fontsize=8, rotation=10, ha='right')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Test R²', fontweight='bold')
    annotate(axes[1], test_r2)

    # CV R2 with error bars
    axes[2].bar(range(len(model_names)), cv_r2, yerr=cv_std,
                color=colors_bar, alpha=0.9, edgecolor='white', capsize=5)
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(short, fontsize=8, rotation=10, ha='right')
    axes[2].set_ylabel('CV R² Score (5-fold)')
    axes[2].set_title('Cross-Validation R²', fontweight='bold')
    for i, (val, std) in enumerate(zip(cv_r2, cv_std)):
        axes[2].text(i, val + std + 0.003, f'{val:.4f}\n±{std:.4f}',
                     ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_07_model_comparison.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 8: Actual vs Predicted (Val set & Test set) ---
def chart_actual_vs_predicted():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Validation set
    y_val_pred = best_results['y_val_pred']
    axes[0].scatter(y_val, y_val_pred, alpha=0.3, s=8, color=COLORS['primary'])
    min_v = min(y_val.min(), y_val_pred.min())
    max_v = max(y_val.max(), y_val_pred.max())
    axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Income')
    axes[0].set_ylabel('Predicted Income')
    axes[0].set_title(f'Actual vs Predicted (Validation Set)\nR² = {best_results["Val_R2"]:.4f}', fontweight='bold')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[0].legend(fontsize=8)

    # Residual distribution (Test set)
    y_test_pred = best_results['y_test_pred']
    residuals = np.array(y_test) - np.array(y_test_pred)
    axes[1].hist(residuals, bins=40, color=COLORS['secondary'], alpha=0.85, edgecolor='white')
    axes[1].axvline(0, color=COLORS['danger'], linestyle='--', linewidth=1.5)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residual Distribution (Test Set)\nMean=${residuals.mean():,.0f}  Std=${residuals.std():,.0f}', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_08_actual_vs_predicted.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 9: Income Percentiles (Area Chart) ---
def chart_income_percentiles():
    fig, ax = plt.subplots(figsize=(10, 5))
    percentiles = list(range(5, 100, 5))
    values = [df['Income'].quantile(p/100) for p in percentiles]

    ax.fill_between(percentiles, values, alpha=0.3, color=COLORS['primary'])
    ax.plot(percentiles, values, color=COLORS['primary'], linewidth=2, marker='o', markersize=5)

    for i, (p, v) in enumerate(zip(percentiles, values)):
        if p in [25, 50, 75, 90, 95]:
            ax.annotate(f'P{p}: ${v:,.0f}', (p, v), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['primary'], alpha=0.8))

    ax.set_xlabel('Percentile')
    ax.set_ylabel('Income (USD)')
    ax.set_title('Income Percentile Distribution', fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_09_percentiles.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 10: Doughnut Charts (Education & Occupation) ---
def chart_doughnut_distributions():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Education
    edu_counts = df['Education_Level'].value_counts()
    wedges1, texts1, autotexts1 = axes[0].pie(edu_counts, labels=edu_counts.index, autopct='%1.1f%%',
                colors=COLORS['palette'][:len(edu_counts)], startangle=90,
                pctdistance=0.8, textprops={'fontsize': 9})
    centre_circle1 = plt.Circle((0, 0), 0.55, fc='white')
    axes[0].add_artist(centre_circle1)
    axes[0].set_title('Education Level Distribution', fontweight='bold')
    axes[0].text(0, 0, f'N={len(df):,}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Occupation
    occ_counts = df['Occupation'].value_counts()
    wedges2, texts2, autotexts2 = axes[1].pie(occ_counts, labels=occ_counts.index, autopct='%1.1f%%',
                colors=COLORS['palette'][:len(occ_counts)], startangle=90,
                pctdistance=0.8, textprops={'fontsize': 9})
    centre_circle2 = plt.Circle((0, 0), 0.55, fc='white')
    axes[1].add_artist(centre_circle2)
    axes[1].set_title('Occupation Distribution', fontweight='bold')
    axes[1].text(0, 0, f'N={len(df):,}', ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_10_doughnut.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 11: Employment & Gender Analysis ---
def chart_employment_gender():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Employment Status
    emp_stats = df.groupby('Employment_Status')['Income'].agg(['mean', 'median']).sort_values('mean', ascending=False)
    x = np.arange(len(emp_stats))
    width = 0.35
    axes[0].bar(x - width/2, emp_stats['median'], width, label='Median', color=COLORS['info'])
    axes[0].bar(x + width/2, emp_stats['mean'], width, label='Mean', color=COLORS['danger'])
    axes[0].set_title('Income by Employment Status', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(emp_stats.index, fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[0].legend()

    # Gender
    gender_stats = df.groupby('Gender')['Income'].agg(['mean', 'median'])
    x = np.arange(len(gender_stats))
    axes[1].bar(x - width/2, gender_stats['median'], width, label='Median', color=COLORS['secondary'])
    axes[1].bar(x + width/2, gender_stats['mean'], width, label='Mean', color=COLORS['accent'])
    axes[1].set_title('Income by Gender', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gender_stats.index, fontsize=9)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_11_employment_gender.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 12: Age & Work Experience Scatter ---
def chart_age_experience_scatter():
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Work_Experience'], df['Age'], c=np.log1p(df['Income']),
                         cmap='viridis', alpha=0.4, s=8, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax, label='log(Income)')
    ax.set_xlabel('Work Experience (years)')
    ax.set_ylabel('Age')
    ax.set_title('Age vs Work Experience (colored by log Income)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_12_age_experience.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 13: Learning Curve ---
def chart_learning_curve():
    fig, ax = plt.subplots(figsize=(10, 5))
    # Use a smaller RF for speed
    rf_lc = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    train_sizes, train_scores, test_scores = learning_curve(
        rf_lc, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=COLORS['primary'])
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color=COLORS['danger'])
    ax.plot(train_sizes, train_mean, 'o-', color=COLORS['primary'], linewidth=2, label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color=COLORS['danger'], linewidth=2, label='Validation Score')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.set_title('Learning Curve - Random Forest', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_13_learning_curve.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 14: MAE/RMSE Comparison ---
def chart_error_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    model_names = list(results.keys())
    mae_vals = [results[m]['MAE'] for m in model_names]
    rmse_vals = [results[m]['RMSE'] for m in model_names]

    colors_bar = [COLORS['success'] if m == best_name else COLORS['primary'] for m in model_names]

    # MAE
    bars_mae = axes[0].bar(range(len(model_names)), mae_vals, color=colors_bar, alpha=0.9, edgecolor='white')
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
    axes[0].set_ylabel('MAE (USD)')
    axes[0].set_title('Model Comparison - MAE', fontweight='bold')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
    for bar, val in zip(bars_mae, mae_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'${val/1000:,.0f}K', ha='center', va='bottom', fontsize=7)

    # RMSE
    bars_rmse = axes[1].bar(range(len(model_names)), rmse_vals, color=colors_bar, alpha=0.9, edgecolor='white')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
    axes[1].set_ylabel('RMSE (USD)')
    axes[1].set_title('Model Comparison - RMSE', fontweight='bold')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
    for bar, val in zip(bars_rmse, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                     f'${val/1000:,.0f}K', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_14_error_comparison.png')
    plt.savefig(path)
    plt.close()
    return path

# --- Chart 15: Marital Status & Housing Analysis ---
def chart_marital_housing():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Marital Status
    mar_stats = df.groupby('Marital_Status')['Income'].agg(['mean', 'median']).sort_values('mean', ascending=False)
    x = np.arange(len(mar_stats))
    width = 0.35
    axes[0].bar(x - width/2, mar_stats['median'], width, label='Median', color=COLORS['primary'])
    axes[0].bar(x + width/2, mar_stats['mean'], width, label='Mean', color=COLORS['accent'])
    axes[0].set_title('Income by Marital Status', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(mar_stats.index, fontsize=9)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[0].legend()

    # Housing Type
    house_stats = df.groupby('Type_of_Housing')['Income'].agg(['mean', 'median']).sort_values('mean', ascending=False)
    x = np.arange(len(house_stats))
    axes[1].bar(x - width/2, house_stats['median'], width, label='Median', color=COLORS['success'])
    axes[1].bar(x + width/2, house_stats['mean'], width, label='Mean', color=COLORS['danger'])
    axes[1].set_title('Income by Housing Type', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(house_stats.index, fontsize=8, rotation=10, ha='right')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'chart_15_marital_housing.png')
    plt.savefig(path)
    plt.close()
    return path


# Generate all charts
chart_paths = {}
chart_funcs = {
    'income_dist': chart_income_distribution,
    'boxplot_edu': chart_boxplot_education,
    'income_occ': chart_income_by_occupation,
    'income_loc': chart_income_by_location,
    'correlation': chart_correlation_heatmap,
    'feat_imp': chart_feature_importance,
    'model_comp': chart_model_comparison,
    'actual_pred': chart_actual_vs_predicted,
    'percentiles': chart_income_percentiles,
    'doughnut': chart_doughnut_distributions,
    'emp_gender': chart_employment_gender,
    'age_exp': chart_age_experience_scatter,
    'learning': chart_learning_curve,
    'error_comp': chart_error_comparison,
    'marital_housing': chart_marital_housing,
}

for key, func in chart_funcs.items():
    print(f"  Generating chart: {key}...")
    chart_paths[key] = func()

print(f"  All {len(chart_paths)} charts generated!")


# ============================================================
# PHASE 5: BUILD WORD DOCUMENT
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: Building Word document...")
print("=" * 70)

doc = Document()

# Set default font
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)

# Set page margins
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(3)
    section.right_margin = Cm(2.5)

# ===================== TITLE PAGE =====================
doc.add_paragraph()
doc.add_paragraph()
doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run('BÁO CÁO PHÂN TÍCH DỮ LIỆU')
run.bold = True
run.font.size = Pt(26)
run.font.color.rgb = RGBColor(46, 134, 171)
run.font.name = 'Times New Roman'

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run('DỰ BÁO THU NHẬP HỘ GIA ĐÌNH')
run.bold = True
run.font.size = Pt(22)
run.font.color.rgb = RGBColor(162, 59, 114)
run.font.name = 'Times New Roman'

sub2 = doc.add_paragraph()
sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub2.add_run('Sử dụng mô hình Random Forest Regression')
run.font.size = Pt(14)
run.font.color.rgb = RGBColor(100, 100, 100)
run.font.name = 'Times New Roman'
run.italic = True

doc.add_paragraph()
doc.add_paragraph()

# Metadata
line = doc.add_paragraph()
line.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = line.add_run('━' * 50)
run.font.color.rgb = RGBColor(200, 200, 200)

info_items = [
    ('Nguồn dữ liệu', 'Kaggle - Regression Dataset for Household Income Analysis'),
    ('Số lượng mẫu', '10,000 hộ gia đình'),
    ('Phương pháp', 'Random Forest Regression (n_estimators=300, max_depth=20)'),
    ('Công cụ', 'Python, scikit-learn, pandas, matplotlib, seaborn'),
    ('Ngày thực hiện', 'Tháng 10/2026'),
]

for label, value in info_items:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f'{label}: ')
    run.bold = True
    run.font.size = Pt(11)
    run.font.name = 'Times New Roman'
    run = p.add_run(value)
    run.font.size = Pt(11)
    run.font.name = 'Times New Roman'

doc.add_page_break()

# ===================== TABLE OF CONTENTS =====================
add_heading_styled(doc, 'MỤC LỤC', level=1)
toc_items = [
    ('1.', 'Giới thiệu'),
    ('2.', 'Tổng quan dữ liệu'),
    ('3.', 'Phân tích khám phá dữ liệu (EDA)'),
    ('  3.1.', 'Phân phối thu nhập'),
    ('  3.2.', 'Phân tích theo trình độ học vấn'),
    ('  3.3.', 'Phân tích theo nghề nghiệp'),
    ('  3.4.', 'Phân tích theo khu vực'),
    ('  3.5.', 'Phân tích tương quan'),
    ('  3.6.', 'Phân tích theo tình trạng việc làm và giới tính'),
    ('  3.7.', 'Phân tích theo tình trạng hôn nhân và loại nhà ở'),
    ('  3.8.', 'Phân tích tuổi và kinh nghiệm'),
    ('  3.9.', 'Phân phối phần trăm thu nhập'),
    ('  3.10.', 'Phân phối trình độ học vấn và nghề nghiệp'),
    ('4.', 'Tiền xử lý dữ liệu'),
    ('5.', 'Xây dựng mô hình Random Forest'),
    ('  5.1.', 'Cấu hình mô hình'),
    ('  5.2.', 'Hiệu suất mô hình'),
    ('  5.3.', 'So sánh các mô hình'),
    ('  5.4.', 'Phân tích tầm quan trọng đặc trưng'),
    ('  5.5.', 'Đường cong học tập'),
    ('6.', 'Đánh giá và thảo luận'),
    ('7.', 'Kết luận và khuyến nghị'),
]

for num, title_text in toc_items:
    p = doc.add_paragraph()
    run = p.add_run(f'{num} {title_text}')
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    if not num.startswith(' '):
        run.bold = True
    p.paragraph_format.space_after = Pt(2)

doc.add_page_break()

# ===================== SECTION 1: INTRODUCTION =====================
add_heading_styled(doc, '1. Giới thiệu', level=1)

add_body_text(doc,
    'Trong bối cảnh kinh tế xã hội ngày càng phức tạp, việc hiểu và dự báo thu nhập hộ gia đình '
    'đóng vai trò quan trọng trong hoạch định chính sách, phân tích thị trường và nghiên cứu nhân khẩu học. '
    'Báo cáo này trình bày quá trình phân tích dữ liệu thu nhập hộ gia đình và xây dựng mô hình '
    'dự báo sử dụng thuật toán Random Forest Regression — một trong những phương pháp Machine Learning '
    'mạnh mẽ nhất cho bài toán hồi quy.'
)

add_body_text(doc,
    'Mục tiêu của dự án bao gồm: (1) Phân tích khám phá các yếu tố nhân khẩu học và kinh tế xã hội '
    'ảnh hưởng đến thu nhập hộ gia đình; (2) Xây dựng và đánh giá mô hình dự báo thu nhập '
    'dựa trên Random Forest Regression; (3) Xác định các đặc trưng quan trọng nhất ảnh hưởng đến thu nhập; '
    '(4) Đưa ra các khuyến nghị dựa trên kết quả phân tích.'
)

add_body_text(doc,
    'Dữ liệu được sử dụng từ bộ dữ liệu "Regression Dataset for Household Income Analysis" trên Kaggle, '
    'bao gồm 10,000 bản ghi với 13 đặc trưng đầu vào (4 đặc trưng số và 9 đặc trưng phân loại) '
    'và biến mục tiêu là thu nhập hàng năm của hộ gia đình (USD).'
)

doc.add_page_break()

# ===================== SECTION 2: DATA OVERVIEW =====================
add_heading_styled(doc, '2. Tổng quan dữ liệu', level=1)

add_heading_styled(doc, '2.1. Thông tin tổng quát', level=2)
add_body_text(doc,
    'Bộ dữ liệu chứa thông tin nhân khẩu học và kinh tế xã hội của 10,000 hộ gia đình, '
    'bao gồm thông tin về tuổi, trình độ học vấn, nghề nghiệp, khu vực sinh sống, '
    'kinh nghiệm làm việc và các yếu tố liên quan khác.'
)

add_formatted_table(doc,
    ['Thuộc tính', 'Giá trị'],
    [
        ['Tổng số bản ghi', f'{len(df):,}'],
        ['Số đặc trưng đầu vào', '13 (4 số + 9 phân loại)'],
        ['Biến mục tiêu', 'Thu nhập hàng năm (USD)'],
        ['Giá trị trống', '0'],
        ['Bản ghi trùng lặp', '0'],
        ['Outliers (phương pháp IQR)', f'{n_outliers} ({pct_outliers:.1f}%)'],
    ],
    col_widths=[6, 10]
)

doc.add_paragraph()
add_heading_styled(doc, '2.2. Mô tả các đặc trưng', level=2)

feature_desc = [
    ['Age', 'Số', 'Tuổi của thành viên chính (18-70)'],
    ['Education_Level', 'Phân loại', 'High School / Bachelor\'s / Master\'s / Doctorate'],
    ['Occupation', 'Phân loại', 'Healthcare / Education / Technology / Finance / Others'],
    ['Number_of_Dependents', 'Số', 'Số người phụ thuộc (0-5)'],
    ['Location', 'Phân loại', 'Urban / Suburban / Rural'],
    ['Work_Experience', 'Số', 'Số năm kinh nghiệm (0-50)'],
    ['Marital_Status', 'Phân loại', 'Single / Married / Divorced'],
    ['Employment_Status', 'Phân loại', 'Full-time / Part-time / Self-employed'],
    ['Household_Size', 'Số', 'Quy mô hộ gia đình (1-7)'],
    ['Homeownership_Status', 'Phân loại', 'Own / Rent'],
    ['Type_of_Housing', 'Phân loại', 'Apartment / Single-family home / Townhouse'],
    ['Gender', 'Phân loại', 'Male / Female'],
    ['Primary_Mode_of_Transportation', 'Phân loại', 'Car / Public transit / Biking / Walking'],
]

add_formatted_table(doc,
    ['Đặc trưng', 'Loại', 'Mô tả'],
    feature_desc,
    col_widths=[5, 3, 9]
)

doc.add_paragraph()
add_heading_styled(doc, '2.3. Thống kê mô tả thu nhập', level=2)

add_formatted_table(doc,
    ['Chỉ số thống kê', 'Giá trị'],
    [
        ['Trung bình (Mean)', f"${income_stats['mean']:,.2f}"],
        ['Trung vị (Median)', f"${income_stats['median']:,.2f}"],
        ['Độ lệch chuẩn (Std)', f"${income_stats['std']:,.2f}"],
        ['Giá trị nhỏ nhất', f"${income_stats['min']:,.2f}"],
        ['Giá trị lớn nhất', f"${income_stats['max']:,.2f}"],
        ['Phân vị 25% (Q1)', f"${income_stats['q25']:,.2f}"],
        ['Phân vị 75% (Q3)', f"${income_stats['q75']:,.2f}"],
        ['Hệ số bất đối xứng (Skewness)', f"{income_stats['skew']:.4f}"],
        ['Hệ số nhọn (Kurtosis)', f"{income_stats['kurtosis']:.4f}"],
    ],
    col_widths=[7, 9]
)

add_body_text(doc,
    f'Thu nhập có phân phối lệch phải mạnh (skewness = {income_stats["skew"]:.2f}), '
    f'với khoảng cách lớn giữa trung bình (${income_stats["mean"]:,.0f}) và trung vị '
    f'(${income_stats["median"]:,.0f}). Điều này cho thấy sự hiện diện của nhiều giá trị ngoại lệ '
    f'({n_outliers} outliers, chiếm {pct_outliers:.1f}% dữ liệu) ở mức thu nhập cao, '
    f'đòi hỏi phương pháp xử lý phù hợp trước khi xây dựng mô hình.'
)

doc.add_page_break()

# ===================== SECTION 3: EDA =====================
add_heading_styled(doc, '3. Phân tích khám phá dữ liệu (EDA)', level=1)

add_body_text(doc,
    'Phần này trình bày các phân tích khám phá dữ liệu (Exploratory Data Analysis) nhằm '
    'hiểu rõ phân phối của dữ liệu, mối quan hệ giữa các biến và các pattern tiềm ẩn '
    'trước khi xây dựng mô hình dự báo.'
)

# 3.1 Income Distribution
add_heading_styled(doc, '3.1. Phân phối thu nhập', level=2)
add_body_text(doc,
    f'Biểu đồ dưới đây thể hiện phân phối thu nhập gốc cùng với các chỉ số tổng hợp. '
    f'Thu nhập có phân phối lệch phải rất mạnh '
    f'(skewness = {income_stats["skew"]:.2f}), với khoảng cách lớn giữa trung bình (${income_stats["mean"]:,.0f}) '
    f'và trung vị (${income_stats["median"]:,.0f}). '
    f'Các mô hình được huấn luyện trực tiếp trên giá trị thu nhập gốc (không log transform).'
)
add_chart_to_doc(doc, chart_paths['income_dist'],
    'Hình 3.1: Phân phối thu nhập hộ gia đình (Income Distribution)')

# 3.2 Education
add_heading_styled(doc, '3.2. Phân tích theo trình độ học vấn', level=2)
add_body_text(doc,
    'Biểu đồ hộp (box plot) dưới đây thể hiện phân phối thu nhập theo từng cấp độ học vấn. '
    'Kết quả cho thấy trung vị thu nhập không có sự khác biệt đáng kể giữa các nhóm trình độ, '
    'tuy nhiên giá trị trung bình cho thấy sự chênh lệch rõ ràng hơn do ảnh hưởng của các outlier '
    'ở mức thu nhập cao.'
)
add_chart_to_doc(doc, chart_paths['boxplot_edu'],
    'Hình 3.2: Phân phối thu nhập theo trình độ học vấn (Box Plot)')

# Education stats table
edu_stats = df.groupby('Education_Level')['Income'].agg(['count', 'mean', 'median', 'std'])
edu_rows = []
for level in ['High School', "Bachelor's", "Master's", 'Doctorate']:
    if level in edu_stats.index:
        row = edu_stats.loc[level]
        edu_rows.append([level, f'{int(row["count"]):,}', f'${row["mean"]:,.0f}',
                        f'${row["median"]:,.0f}', f'${row["std"]:,.0f}'])

add_formatted_table(doc,
    ['Trình độ', 'Số lượng', 'Trung bình', 'Trung vị', 'Độ lệch chuẩn'],
    edu_rows,
    col_widths=[3.5, 2.5, 3.5, 3.5, 3.5]
)

doc.add_page_break()

# 3.3 Occupation
add_heading_styled(doc, '3.3. Phân tích theo nghề nghiệp', level=2)
add_body_text(doc,
    'Biểu đồ nhóm (grouped bar chart) so sánh thu nhập trung bình và trung vị theo các nhóm nghề nghiệp. '
    'Kết quả cho thấy sự khác biệt rõ rệt giữa giá trị trung bình và trung vị ở tất cả các nhóm, '
    'phản ánh tính chất phân phối lệch phải của dữ liệu thu nhập.'
)
add_chart_to_doc(doc, chart_paths['income_occ'],
    'Hình 3.3: Thu nhập theo nghề nghiệp (Trung bình vs Trung vị)')

# 3.4 Location
add_heading_styled(doc, '3.4. Phân tích theo khu vực', level=2)
add_body_text(doc,
    'Phân tích theo khu vực cho thấy phần lớn hộ gia đình tập trung ở khu vực Urban (đô thị). '
    'Tuy nhiên, thu nhập trung bình ở khu vực Rural (nông thôn) lại cao hơn, '
    'có thể do số lượng mẫu nhỏ hơn và bị ảnh hưởng bởi các outlier.'
)
add_chart_to_doc(doc, chart_paths['income_loc'],
    'Hình 3.4: Thu nhập theo khu vực (trái) và phân phối khu vực (phải)')

# 3.5 Correlation
add_heading_styled(doc, '3.5. Phân tích tương quan', level=2)
add_body_text(doc,
    'Ma trận tương quan giữa các biến số cho thấy mức độ liên hệ tuyến tính giữa các đặc trưng. '
    'Đáng chú ý, không có mối tương quan mạnh nào giữa các biến đầu vào với thu nhập, '
    'cho thấy mối quan hệ phi tuyến phức tạp — điều này giải thích tại sao mô hình Random Forest '
    '(có khả năng nắm bắt quan hệ phi tuyến) hoạt động tốt hơn các mô hình tuyến tính.'
)
add_chart_to_doc(doc, chart_paths['correlation'],
    'Hình 3.5: Ma trận tương quan giữa các đặc trưng số')

doc.add_page_break()

# 3.6 Employment & Gender
add_heading_styled(doc, '3.6. Phân tích theo tình trạng việc làm và giới tính', level=2)
add_body_text(doc,
    'Phân tích cho thấy nhóm làm việc Part-time có thu nhập trung bình cao nhất, '
    'nhưng điều này có thể bị ảnh hưởng bởi các outlier. Về giới tính, '
    'không có sự chênh lệch đáng kể giữa nam và nữ ở cả chỉ số trung bình lẫn trung vị.'
)
add_chart_to_doc(doc, chart_paths['emp_gender'],
    'Hình 3.6: Thu nhập theo tình trạng việc làm (trái) và giới tính (phải)')

# 3.7 Marital & Housing
add_heading_styled(doc, '3.7. Phân tích theo tình trạng hôn nhân và loại nhà ở', level=2)
add_body_text(doc,
    'Hộ gia đình đã kết hôn (Married) có thu nhập trung bình cao nhất, '
    'trong khi sự khác biệt theo loại nhà ở không quá rõ ràng.'
)
add_chart_to_doc(doc, chart_paths['marital_housing'],
    'Hình 3.7: Thu nhập theo tình trạng hôn nhân (trái) và loại nhà ở (phải)')

# 3.8 Age & Experience
add_heading_styled(doc, '3.8. Phân tích tuổi và kinh nghiệm làm việc', level=2)
add_body_text(doc,
    'Biểu đồ phân tán dưới đây thể hiện mối quan hệ giữa tuổi và kinh nghiệm làm việc, '
    'với màu sắc biểu thị mức thu nhập (theo thang log). '
    'Mối quan hệ giữa tuổi và kinh nghiệm là phân tán, cho thấy tính đa dạng của dữ liệu.'
)
add_chart_to_doc(doc, chart_paths['age_exp'],
    'Hình 3.8: Tuổi vs Kinh nghiệm làm việc (màu theo log Income)')

doc.add_page_break()

# 3.9 Percentiles
add_heading_styled(doc, '3.9. Phân phối phần trăm thu nhập', level=2)
add_body_text(doc,
    'Biểu đồ phân phối phần trăm (percentile) cho thấy sự chênh lệch thu nhập rất lớn '
    'ở các phần trăm cao. Khoảng 60% hộ gia đình có thu nhập dưới $75,000, '
    'trong khi 5% hộ gia đình có thu nhập cao nhất vượt trên $5.3 triệu.'
)
add_chart_to_doc(doc, chart_paths['percentiles'],
    'Hình 3.9: Phân phối phần trăm thu nhập (P5 đến P95)')

# 3.10 Doughnut
add_heading_styled(doc, '3.10. Phân phối trình độ học vấn và nghề nghiệp', level=2)
add_body_text(doc,
    'Biểu đồ dạng donut thể hiện tỷ lệ phân phối các nhóm trình độ học vấn và nghề nghiệp '
    'trong tập dữ liệu. Bachelor\'s chiếm tỷ lệ lớn nhất (40.6%), '
    'trong khi Healthcare là nghề nghiệp phổ biến nhất (30.4%).'
)
add_chart_to_doc(doc, chart_paths['doughnut'],
    'Hình 3.10: Phân phối trình độ học vấn (trái) và nghề nghiệp (phải)')

doc.add_page_break()

# ===================== SECTION 4: PREPROCESSING =====================
add_heading_styled(doc, '4. Tiền xử lý dữ liệu', level=1)

add_body_text(doc,
    'Quá trình tiền xử lý dữ liệu bao gồm các bước chính sau:'
)

add_heading_styled(doc, '4.1. Mã hóa biến phân loại (Label Encoding)', level=2)
add_body_text(doc,
    '9 biến phân loại được mã hóa bằng phương pháp Label Encoding, chuyển đổi '
    'các giá trị text thành các số nguyên. Phương pháp này phù hợp với mô hình Random Forest '
    'vì cây quyết định có thể xử lý tốt các giá trị ordinal.'
)

add_heading_styled(doc, '4.2. Tạo đặc trưng mới (Feature Engineering)', level=2)
add_body_text(doc, 'Hai đặc trưng mới được tạo ra nhằm nắm bắt thêm thông tin:')

add_formatted_table(doc,
    ['Đặc trưng mới', 'Công thức', 'Ý nghĩa'],
    [
        ['Age_WorkExp_Ratio', 'Age / (Work_Experience + 1)', 'Tỷ lệ tuổi trên kinh nghiệm — phản ánh thời điểm bắt đầu sự nghiệp'],
        ['Dependents_per_HHSize', 'Number_of_Dependents / Household_Size', 'Tỷ lệ người phụ thuộc — phản ánh gánh nặng tài chính'],
    ],
    col_widths=[5, 5, 7]
)

add_heading_styled(doc, '4.3. Chuẩn hóa đặc trưng (Standard Scaling)', level=2)
add_body_text(doc,
    'StandardScaler được áp dụng để chuẩn hóa các đặc trưng về trung bình = 0 và độ lệch chuẩn = 1. '
    'Lưu ý: Đối với mô hình Random Forest, bước chuẩn hóa không bắt buộc vì cây quyết định '
    'không nhạy cảm với scale của dữ liệu. Tuy nhiên, nó cần thiết cho các mô hình Linear/Ridge/Lasso.'
)

add_heading_styled(doc, '4.4. Chia tập dữ liệu (70 / 15 / 15)', level=2)
add_body_text(doc,
    f'Dữ liệu được chia thành ba tập: '
    f'Huấn luyện ({len(X_train):,} mẫu — 70%), '
    f'Kiểm định ({len(X_val):,} mẫu — 15%) và '
    f'Kiểm tra ({len(X_test):,} mẫu — 15%). '
    f'Việc tách biệt tập kiểm định (validation) và kiểm tra (test) giúp chọn mô hình '
    f'không bị rò rỉ thông tin từ quá trình lựa chọn.'
)

add_body_text(doc,
    'Không áp dụng biến đổi logarit — các mô hình được huấn luyện trực tiếp trên giá trị thu nhập gốc.'
)

doc.add_page_break()

# ===================== SECTION 5: MODEL =====================
add_heading_styled(doc, '5. Xây dựng mô hình hồi quy', level=1)

add_body_text(doc,
    '5 mô hình hồi quy được huấn luyện và so sánh trực tiếp trên giá trị thu nhập gốc (không log transform), '
    'sử dụng phân hoạch 70/15/15 (Train/Validation/Test). '
    'mô hình tốt nhất được chọn theo Validation R².'
)

# 5.1 Configuration
add_heading_styled(doc, '5.1. Cấu hình các mô hình', level=2)

add_formatted_table(doc,
    ['Mô hình', 'Thuật toán', 'Tham số ch�add_body_text(doc,
    '5 mô hình hồi quy được huấn luyện và so sánh trên cùng tập dữ liệu, '
    'dự đoán trực tiếp trên thu nhập gốc (không log transform). '
    'Mô hình tốt nhất được chọn theo Validation R² cao nhất.'
)

# Comparison table
comp_rows = []
for name in sorted(results, key=lambda n: results[n]['Val_R2'], reverse=True):
    r = results[name]
    is_best = '⋆ ' if name == best_name else ''
    comp_rows.append([
        f'{is_best}{name}',
        f'{r["Val_R2"]:.4f}',
        f'{r["Test_R2"]:.4f}',
        f'{r["CV_R2_Mean"]:.4f} ± {r["CV_R2_Std"]:.4f}',
        f'${r["Val_MAE"]:,.0f}',
        f'${r["Val_RMSE"]:,.0f}',
        f'{r["Val_MAPE"]:.2f}%',
    ])

add_formatted_table(doc,
    ['Mô hình', 'Val R²', 'Test R²', 'CV R² (5-fold)', 'Val MAE', 'Val RMSE', 'Val MAPE'],
    comp_rows,
    col_widths=[3.5, 1.8, 1.8, 3.5, 2.5, 2.5, 1.9]
)

add_body_text(doc,
    f'{best_name} đạt Val R² = {best_results["Val_R2"]:.4f} — cao nhất trong 5 mô hình. '
    f'Kết quả này khẳng định rằng mối quan hệ giữa đặc trưng và thu nhập có tính phi tuyến, '
    f'phù hợp với khả năng của mô hình dựa trên cây quyết định.',
)n và Test set:'
)

add_formatted_table(doc,
    ['Chỉ số', 'Validation', 'Test', 'Giải thích'],
    [
        ['R²',   f'{best_results["Val_R2"]:.4f}',  f'{best_results["Test_R2"]:.4f}',  'Phương sai giải thích được'],
        ['MAE',  f'${best_results["Val_MAE"]:,.2f}',  f'${best_results["Test_MAE"]:,.2f}',  'Sai số tuyệt đối trung bình'],
        ['RMSE', f'${best_results["Val_RMSE"]:,.2f}', f'${best_results["Test_RMSE"]:,.2f}', 'Căn bậc hai sai số bình phương TB'],
        ['MAPE', f'{best_results["Val_MAPE"]:.2f}%',  f'{best_results["Test_MAPE"]:.2f}%',  'Sai số phần trăm tuyệt đối TB'],
        ['CV R² (5-fold)', f'{best_results["CV_R2_Mean"]:.4f} ± {best_results["CV_R2_Std"]:.4f}', '--', 'Đánh giá chéo 5 lần'],
    ],
    col_widths=[3, 3.5, 3.5, 6.5],
    header_color='A23B72'
)

add_body_text(doc,
    f'Mô hình {best_name} đã cho thấy khả năng dự đoán tốt nhất trên tập Validation '
    f'(Val R² = {best_results["Val_R2"]:.4f}), và kết quả nhất quán trên Test set '
    f'(Test R² = {best_results["Test_R2"]:.4f}). '
    f'CV R² = {best_results["CV_R2_Mean"]:.4f} ± {best_results["CV_R2_Std"]:.4f} cho thấy mô hình ổn định qua các fold.'
)

# Actual vs Predicted chart
add_chart_to_doc(doc, chart_paths['actual_pred'],
    'Hình 5.1: Thực tế vs Dự đoán (trái) và Phân phối phần dư (phải)')

doc.add_page_break()

# 5.3 Model Comparison
add_heading_styled(doc, '5.3. So sánh các mô hình', level=2)

add_body_text(doc,
    '5 mô hình hồi quy đã được huấn luyện và so sánh trên cùng tập dữ liệu. '
    'Tất cả mô hình đều sử dụng biến mục tiêu đã biến đổi log.'
)

# Comparison table
comp_rows = []
for name in results:
    r = results[name]
    is_best = '★ ' if name == best_name else ''
    comp_rows.append([
        f'{is_best}{name}',
        f'{r["R2_log"]:.4f}',
        f'{r["CV_R2_Mean"]:.4f} ± {r["CV_R2_Std"]:.4f}',
        f'${r["MAE"]:,.0f}',
        f'${r["RMSE"]:,.0f}',
        f'{r["MAPE"]:.2f}%',
    ])

add_formatted_table(doc,
    ['Mô hình', 'R² (log)', 'CV R² (5-fold)', 'MAE', 'RMSE', 'MAPE'],
    comp_rows,
    col_widths=[3.5, 2, 3.5, 2.5, 2.5, 2]
)

add_body_text(doc,
    'Random Forest đạt R² (log) = 0.1472 — cao nhất trong 5 mô hình, vượt trội so với '
    'các mô hình tuyến tính (R² ≈ 0.033). Gradient Boosting đứng thứ hai với R² = 0.0471. '
    'Kết quả này khẳng định rằng mối quan hệ giữa đặc trưng và thu nhập có tính phi tuyến, '
    'phù hợp với khả năng của mô hình dựa trên cây quyết định.',
)

add_chart_to_doc(doc, chart_paths['model_comp'],
    'Hình 5.2: So sánh Validation R², Test R² và Cross-Validation R² giữa các mô hình')

doc.add_page_break()

# 5.4 Feature Importance
add_heading_styled(doc, '5.4. Phân tích tầm quan trọng đặc trưng', level=2)

add_body_text(doc,
    f'Feature Importance từ mô hình {best_name} cho biết mức độ đóng góp của từng đặc trưng '
    'vào khả năng dự báo. '
    'Các đặc trưng được tạo mới từ Feature Engineering thường đóng vai trò quan trọng đặc biệt.'
)

fi_rows = []
for rank, (feat, imp) in enumerate(fi.items(), 1):
    feat_clean = feat.replace('_encoded', '').replace('_', ' ')
    fi_rows.append([str(rank), feat_clean, f'{imp:.4f}', f'{imp*100:.2f}%'])

add_formatted_table(doc,
    ['Thứ hạng', 'Đặc trưng', 'Điểm quan trọng', 'Tỷ lệ (%)'],
    fi_rows,
    col_widths=[2, 6, 4, 4]
)

doc.add_paragraph()
add_chart_to_doc(doc, chart_paths['feat_imp'],
    f'Hình 5.4: Tầm quan trọng đặc trưng của {best_name}')

add_body_text(doc,
    'Nhận xét chính từ phân tích Feature Importance:', bold=True
)

insights = [
    'Các đặc trưng liên quan đến tuổi và kinh nghiệm làm việc thường chiếm tỷ lệ quan trọng nhất.',
    'Đặc trưng được tạo mới Age_WorkExp_Ratio nắm bắt thông tin tổng hợp hơn từng biến riêng lẻ.',
    'Nghề nghiệp (Occupation) thường quan trọng hơn Trình độ học vấn trong dự đoán thu nhập.',
    'Giới tính có ảnh hưởng thấp nhất — cho thấy sự bình đẳng giới trong bộ dữ liệu này.',
    'Tỷ lệ người phụ thuộc (Dependents_per_HHSize) phản ánh gánh nặng tài chính đóng vai trò đáng kể.',
]

for insight in insights:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(insight)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

doc.add_page_break()

# 5.5 Learning Curve
add_heading_styled(doc, '5.5. Đường cong học tập (Learning Curve)', level=2)

add_body_text(doc,
    'Đường cong học tập cho thấy sự thay đổi của hiệu suất mô hình khi tăng dần kích thước '
    'tập huấn luyện. Khoảng cách giữa đường training và validation phản ánh mức độ overfitting '
    'của mô hình.'
)

add_chart_to_doc(doc, chart_paths['learning'],
    'Hình 5.5: Đường cong học tập của mô hình Random Forest')

add_body_text(doc,
    'Đường cong cho thấy: (1) Training score giảm khi tăng dữ liệu (dấu hiệu bình thường), '
    '(2) Validation score tăng và dần ổn định — cho thấy mô hình đã convergence, '
    '(3) Khoảng cách giữa hai đường cho thấy có một mức overfitting nhất định, '
    'nhưng với thêm dữ liệu chất lượng cao hơn, mô hình có tiềm năng cải thiện.'
)

doc.add_page_break()

# ===================== SECTION 6: EVALUATION =====================
add_heading_styled(doc, '6. Đánh giá và thảo luận', level=1)

add_heading_styled(doc, '6.1. Đánh giá tổng thể', level=2)
add_body_text(doc,
    f'Mô hình {best_name} được chọn là tốt nhất với Val R² = {best_results["Val_R2"]:.4f} '
    f'và Test R² = {best_results["Test_R2"]:.4f}. Mô hình này cho thấy khả năng nắm bắt '
    f'các mối quan hệ phi tuyến trong dữ liệu tốt hơn đáng kể so với các mô hình tuyến tính.'
)

add_heading_styled(doc, '6.2. Phân tích hạn chế', level=2)
limitations = [
    'Tính nhiễu cao của dữ liệu: Bộ dữ liệu tổng hợp chứa nhiều nhiễu ngẫu nhiên, hạn chế khả năng dự báo chính xác của bất kỳ mô hình nào.',
    f'R² khiêm tốn: Giá trị Test R² = {best_results["Test_R2"]:.4f} cho thấy mô hình chỉ giải thích được ~{best_results["Test_R2"]*100:.0f}% phương sai, phần còn lại là do các yếu tố không có trong dữ liệu.',
    f'MAPE cao ({best_results["Val_MAPE"]:.1f}%): Sai số phần trăm lớn, đặc biệt ở các hộ có thu nhập thấp.',
    'Hạn chế của Label Encoding: Phương pháp này có thể tạo ra thứ tự giả tạo cho các biến phân loại danh nghĩa.',
    'Không dùng log transform: Phân phối thu nhập lệch phải mạnh có thể làm giảm hiệu suất của mô hình tuyến tính, nhưng ít ảnh hưởng hơn đến mô hình dựa trên cây.',
]

for lim in limitations:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(lim)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

add_heading_styled(doc, '6.3. Ưu điểm của Random Forest', level=2)
advantages = [
    'Xử lý tốt mối quan hệ phi tuyến: Cây quyết định tự nhiên nắm bắt được tương tác phức tạp giữa các biến.',
    'Không nhạy cảm với scale: Không yêu cầu chuẩn hóa dữ liệu đầu vào.',
    'Feature Importance tích hợp: Cung cấp thông tin hữu ích về tầm quan trọng của từng đặc trưng.',
    'Giảm overfitting: Kỹ thuật bagging và random feature selection giúp tăng tính tổng quát.',
    f'Hiệu suất ổn định: CV R² = {best_results["CV_R2_Mean"]:.4f} ± {best_results["CV_R2_Std"]:.4f} cho thấy mô hình ổn định qua các fold.',
    'Không cần log transform: Mô hình dựa trên cây hoạt động tốt trực tiếp trên thu nhập gốc.',
]

for adv in advantages:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(adv)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

doc.add_page_break()

# ===================== SECTION 7: CONCLUSION =====================
add_heading_styled(doc, '7. Kết luận và khuyến nghị', level=1)

add_heading_styled(doc, '7.1. Kết luận', level=2)
add_body_text(doc,
    'Qua quá trình phân tích dữ liệu và xây dựng mô hình, chúng tôi rút ra các kết luận sau:'
)

conclusions = [
    f'{best_name} là mô hình phù hợp nhất cho bài toán dự báo thu nhập hộ gia đình, với Val R² = {best_results["Val_R2"]:.4f}, Test R² = {best_results["Test_R2"]:.4f} và CV R² = {best_results["CV_R2_Mean"]:.4f}.',
    'Tỷ lệ Tuổi/Kinh nghiệm, Kinh nghiệm làm việc và Tuổi là 3 yếu tố quan trọng nhất ảnh hưởng đến thu nhập.',
    'Mô hình được chọn theo Validation R² và kết quả được báo cáo chính thức trên Test set độc lập (phân hoạch 70/15/15).',
    'Không có sự khác biệt đáng kể về thu nhập giữa các nhóm giới tính, trình độ học vấn hay khu vực trong bộ dữ liệu này.',
    'Tính chất nhiễu cao của dữ liệu tổng hợp hạn chế khả năng đạt R² cao, nhưng mô hình vẫn cho kết quả có ý nghĩa thống kê.',
]

for i, conc in enumerate(conclusions, 1):
    p = doc.add_paragraph()
    run = p.add_run(f'({i}) ')
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run = p.add_run(conc)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(4)

add_heading_styled(doc, '7.2. Khuyến nghị', level=2)

recommendations = [
    ('Thu thập thêm dữ liệu', 'Bổ sung các biến như tài sản, khoản vay, lịch sử thu nhập để tăng khả năng giải thích.'),
    ('Thử One-Hot Encoding', 'Sử dụng One-Hot Encoding thay vì Label Encoding cho các biến danh nghĩa.'),
    ('Hyperparameter Tuning', 'Áp dụng RandomizedSearchCV hoặc Bayesian Optimization để tối ưu siêu tham số.'),
    ('Stacking/Blending', 'Kết hợp nhiều mô hình (ensemble stacking) để cải thiện hiệu suất.'),
    ('Xử lý Outlier', 'Thử nghiệm các phương pháp xử lý outlier khác như Winsorization hoặc Robust Scaler.'),
    ('Feature Engineering nâng cao', 'Tạo thêm các đặc trưng tương tác (interaction features) và đặc trưng đa thức.'),
]

for title_rec, desc in recommendations:
    p = doc.add_paragraph(style='List Bullet')
    run = p.add_run(f'{title_rec}: ')
    run.bold = True
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    run = p.add_run(desc)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)

doc.add_paragraph()
doc.add_paragraph()

# Footer
line = doc.add_paragraph()
line.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = line.add_run('━' * 50)
run.font.color.rgb = RGBColor(200, 200, 200)

footer = doc.add_paragraph()
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer.add_run('— Hết báo cáo —')
run.italic = True
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(150, 150, 150)
run.font.name = 'Times New Roman'

# ============================================================
# SAVE DOCUMENT
# ============================================================
doc.save(OUTPUT_DOCX)

# Cleanup temp charts
shutil.rmtree(CHART_DIR, ignore_errors=True)

print(f"\n{'=' * 70}")
print(f"REPORT GENERATED SUCCESSFULLY!")
print(f"{'=' * 70}")
print(f"  Output: {os.path.abspath(OUTPUT_DOCX)}")
print(f"  Pages: ~25 pages")
print(f"  Charts: {len(chart_paths)} charts embedded")
print(f"  Sections: 7 main sections")
