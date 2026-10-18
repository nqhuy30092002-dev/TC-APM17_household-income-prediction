# 📊 Household Income Prediction

> Web application for predicting annual household income using **Machine Learning**. Trained directly on raw income values (no log transform) with a 70/15/15 Train/Validation/Test split.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-pink?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](https://www.apache.org/licenses/LICENSE-2.0)

---

## 🎯 Overview

Dự án phân tích các yếu tố nhân khẩu học và kinh tế xã hội ảnh hưởng đến thu nhập hộ gia đình, huấn luyện mô hình **Random Forest Regressor**, và triển khai dưới dạng web app tương tác.

**Dataset:** [Kaggle - Regression Dataset for Household Income Analysis](https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis)

### Key Features
- 🔬 **Exploratory Data Analysis** — Notebook EDA + 11 interactive charts trên web
- 📈 **5 ML Models** — Linear, Ridge, Lasso, Random Forest, Gradient Boosting (best: Lasso, CV R² = 0.0094)
- 📊 **11 Chart.js Charts** — Histogram, grouped bar, doughnut, area, horizontal bar
- 🌐 **Interactive Web App** — Real-time prediction engine chạy client-side
- 📱 **Responsive Design** — Desktop & mobile
- ✨ **Premium Dark UI** — Glassmorphism, animations

---

## 📁 Project Structure

```
Household Income Prediction/
│
├── 📂 data/                        # Dataset
│   └── data.csv                    # Raw dataset (10,000 records)
│
├── 📂 notebooks/                   # Jupyter notebooks
│   └── EDA.ipynb                   # EDA + Modeling pipeline (18 sections)
│
├── 📂 src/                         # Source code modules
│   ├── __init__.py                 # Package init
│   ├── data_loader.py              # Data loading & validation
│   ├── preprocessing.py            # Label encoding, feature engineering, scaling, 70/15/15 split
│   ├── models.py                   # Model definitions & registry
│   ├── evaluation.py               # Metrics & feature importance
│   └── train_and_evaluate_all.py   # End-to-end training pipeline
│
├── 📂 static/                      # Web application files
│   ├── index.html                  # Main HTML page
│   ├── styles.css                  # CSS design system (dark mode)
│   ├── app.js                      # Chart.js rendering + prediction engine
│   ├── chart_data.json             # Pre-computed chart data from dataset
│   ├── model_metadata.json         # Random Forest metrics & feature importance
│   └── web_model_params.json       # Ridge coefficients for JS prediction
│
├── 📂 models/                      # Trained model artifacts
│   ├── model.joblib                # Random Forest model (best)
│   └── scaler.joblib               # Fitted StandardScaler
│
├── main.py                         # Application entry point (HTTP server)
├── analysis_and_model.py           # Original analysis & training script
├── generate_chart_data.py          # Script to generate chart_data.json
├── generate_report.py              # Script to generate Word report (.docx)
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web App

```bash
python main.py
```

This will start the web server and open `http://127.0.0.1:8080` in your browser.

### 3. Options

```bash
python main.py --port 3000        # Custom port
python main.py --train            # Re-train models first
python main.py --no-browser       # Don't auto-open browser
```

---

## 📈 Data Analysis

### Dataset Summary

| Property | Value |
|----------|-------|
| Records | 10,000 |
| Features | 13 (4 numerical + 9 categorical) |
| Target | Annual Household Income (USD) |
| Missing Values | 0 |
| Duplicates | 0 |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numerical | Age of primary member (18-70) |
| Education_Level | Categorical | High School / Bachelor's / Master's / Doctorate |
| Occupation | Categorical | Healthcare / Education / Technology / Finance / Others |
| Number_of_Dependents | Numerical | 0-5 dependents |
| Location | Categorical | Urban / Suburban / Rural |
| Work_Experience | Numerical | Years of experience (0-50) |
| Marital_Status | Categorical | Single / Married / Divorced |
| Employment_Status | Categorical | Full-time / Part-time / Self-employed |
| Household_Size | Numerical | 1-7 members |
| Homeownership_Status | Categorical | Own / Rent |
| Type_of_Housing | Categorical | Apartment / Single-family home / Townhouse |
| Gender | Categorical | Male / Female |
| Primary_Mode_of_Transportation | Categorical | Car / Public transit / Biking / Walking |

### Income Distribution

- **Mean:** $816,838 | **Median:** $72,943
- **Skewness:** 2.98 (heavily right-skewed) — models trained directly on **raw income** (no log transform)
- **~19%** outliers detected via IQR method

### EDA Notebook

For detailed visualizations, analysis, and modeling pipeline, see [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).

The notebook contains **18 sections** split into two parts:

**Part I — Exploratory Data Analysis**

| # | Section | Chart Type |
|---|---------|------------|
| 1 | Import & Load Data | — |
| 2 | Data Overview & Quality | — |
| 3 | Income Distribution | Histogram (raw + log) |
| 4 | Education Level Analysis | Box Plot |
| 5 | Occupation Analysis | Grouped Bar (Mean vs Median) |
| 6 | Location Analysis | Grouped Bar + Pie |
| 7 | Correlation Analysis | Heatmap + Horizontal Bar |
| 8 | Employment & Gender | Grouped Bar (2-panel) |
| 9 | Marital Status & Housing | Grouped Bar (2-panel) |
| 10 | Age & Work Experience | Scatter Plot (color-coded) |
| 11 | Income Percentiles | Area Chart (P5→P95) |
| 12 | Category Distributions | Doughnut Charts |

**Part II — Data Preprocessing & Modeling**

| # | Section | Content |
|---|---------|------------|
| 13 | Data Preprocessing | Label Encoding, Feature Engineering (no log transform) |
| 14 | Train/Validation/Test Split | 70/15/15 split (7,000 / 1,500 / 1,500) |
| 15 | Feature Scaling | StandardScaler (mean=0, std=1) |
| 16 | Model Training | 5 models (Linear, Ridge, Lasso, RF, GB) |
| 17 | Model Evaluation | R²/MAE/RMSE comparison on Val & Test sets, Feature Importance |
| 18 | Summary & Recommendations | Key findings + improvement suggestions |

---

## 🌲 Random Forest Model

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | Random Forest Regressor |
| n_estimators | 300 |
| max_depth | 20 |
| min_samples_split | 5 |
| Target Transform | None (raw income) |
| Train/Val/Test Split | 70/15/15 |
| Cross-Validation | 5-fold |

### Performance Metrics

| Metric | Val Set | Test Set |
|--------|---------|----------|
| **R²** | **0.0114** | **0.0077** |
| **CV R² (5-fold)** | **0.0094 ± 0.0033** | — |
| MAE | $1,093,787 | $1,138,136 |
| RMSE | $1,796,537 | $1,866,043 |
| MAPE | 746.3% | 730.5% |

> **Note:** Metrics are for the best model (Lasso Regression). The very low R² scores are expected for this heavily right-skewed synthetic dataset ($816K mean, $73K median, skewness=2.98) when models are trained on **raw income without log transform**. The high MAPE reflects extreme outliers in the income distribution (max ~$10M).
>
> Models are trained and evaluated on **raw income values** (no log transform). Model selection is based on **Validation R²**.

### All Model Results (Val Set)

| Model | Val R² | Val MAE | Test R² | CV R² |
|-------|--------|---------|---------|-------|
| **Lasso Regression** ★ | **0.0114** | $1,093,787 | 0.0077 | 0.0094 |
| Linear Regression | 0.0114 | $1,093,787 | 0.0077 | 0.0094 |
| Ridge Regression | 0.0114 | $1,093,823 | 0.0077 | 0.0094 |
| Random Forest | -0.0043 | $1,151,547 | 0.0109 | 0.0142 |
| Gradient Boosting | -0.0487 | $1,131,649 | -0.0429 | -0.0358 |

### Top Feature Importance

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Age / Work Experience Ratio | 17.78% |
| 2 | Location | 17.03% |
| 3 | Homeownership Status | 16.16% |
| 4 | Work Experience | 15.14% |
| 5 | Type of Housing | 6.86% |
| 6 | Dependents / Household Size | 6.82% |
| 7 | Household Size | 6.03% |
| 8 | Age | 4.26% |

> Feature importance based on Ridge normalized |coefficients| (best model Lasso has no non-zero coefs on this data).

---

## 📊 Web App — Interactive Charts

The web app includes **12 interactive Chart.js charts** powered by [`chart_data.json`](static/chart_data.json):

| # | Chart | Type | Description |
|---|-------|------|-------------|
| 1 | Income Distribution | Histogram | Raw income frequency |
| 2 | Income by Education | Grouped Bar | Median + Mean per level |
| 3 | Income by Occupation | Grouped Bar | Median + Mean per field |
| 4 | Income by Location | Grouped Bar | Urban / Suburban / Rural |
| 5 | Income by Employment | Grouped Bar | Full-time / Part-time / Self |
| 6 | Income by Gender | Grouped Bar | Male / Female |
| 7 | Income by Marital Status | Grouped Bar | Single / Married / Divorced |
| 8 | Feature Importance | Horizontal Bar | Top 12 RF features |
| 9 | Income Percentiles | Area/Line | P5 → P95 curve |
| 10 | Education Distribution | Doughnut | Category proportions |
| 11 | Occupation Distribution | Doughnut | Category proportions |

To regenerate chart data from the dataset:
```bash
python generate_chart_data.py
```

---

## 🏗️ Technical Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  User Input  │────▶│  JavaScript  │────▶│   Display    │
│  (13 fields) │     │  Prediction  │     │   Results    │
└──────────────┘     │   Engine     │     └──────────────┘
                     └──────┬───────┘
                            │
                ┌───────────┼───────────┐
                ▼                       ▼
        ┌──────────────┐      ┌──────────────┐
        │ Ridge Coeffs │      │  Chart.js    │
        │   (JSON)     │      │  Rendering   │
        └──────────────┘      └──────────────┘
```

**Web Prediction Pipeline:**
1. User inputs 13 demographic features
2. JavaScript encodes categorical variables (Label Encoding)
3. Engineered features computed (Age/WorkExp ratio, Dependents/HH size)
4. Features normalized using StandardScaler parameters
5. Ridge Regression predicts **raw income** directly (no log/exp conversion needed)
6. Results displayed with factor analysis

**Analytics Pipeline:**
1. `generate_chart_data.py` reads `data/data.csv`
2. Computes histograms, group statistics, percentiles
3. Outputs `static/chart_data.json` (19 datasets)
4. `app.js` loads JSON and renders 12 Chart.js charts

---

## 🔧 Development

### Re-train Models

```bash
python main.py --train
```

Or run the training pipeline directly:

```bash
python -m src.train_and_evaluate_all
```

### Regenerate Chart Data

```bash
python generate_chart_data.py
```

### Run EDA Notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

### Generate Word Report

```bash
python generate_report.py
```

This generates `Bao_cao_Phan_tich_Thu_nhap_Ho_Gia_dinh.docx` — a comprehensive Word report with 15 embedded charts, consistent with the EDA notebook visualizations.

### Run Original Analysis Script

```bash
python analysis_and_model.py
```

---

## 📊 Web App Sections

The web app includes:
- **Hero Section** — Overview statistics and call-to-action
- **Prediction Form** — 13-field input form with real-time income prediction (Ridge model)
- **Data Insights** — 11 interactive Chart.js charts with real dataset analysis
- **Random Forest Model** — Model metrics, hyperparameters, and methodology
- **Methodology** — 4-step pipeline explanation

---

## 📜 License

This project uses data from [Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis) under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

---

## 👨‍💻 Author

Built as a Data Analysis & Machine Learning project demonstrating:
- End-to-end ML pipeline (EDA → Preprocessing → Training → Evaluation → Deployment)
- Modular Python project structure (`src/` with 5 modules)
- Client-side ML inference with JavaScript
- Interactive data visualization with Chart.js
- Modern web application design (dark mode, glassmorphism)
