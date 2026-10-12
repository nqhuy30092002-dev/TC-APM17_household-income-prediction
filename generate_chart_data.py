"""
Generate Chart Data — Household Income Prediction
===================================================
Reads data/data.csv and computes 19 chart datasets for the web app:
  - Income distribution histograms (raw + log-transformed)
  - Income grouped by Education, Occupation, Location, Employment, etc.
  - Feature importance (from model_metadata.json)
  - Income percentile curve (P5-P95)
  - Categorical distributions (for doughnut charts)
  - Summary statistics

Output: static/chart_data.json
"""
import pandas as pd
import numpy as np
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/data.csv')

chart_data = {}

# 1. Income Distribution Histogram
hist_counts, hist_edges = np.histogram(df['Income'], bins=30)
chart_data['income_histogram'] = {
    'labels': [f'${int(hist_edges[i]):,}' for i in range(len(hist_counts))],
    'values': hist_counts.tolist(),
    'bin_edges': hist_edges.tolist(),
}

# 2. Log Income Distribution
log_income = np.log1p(df['Income'])
log_counts, log_edges = np.histogram(log_income, bins=30)
chart_data['log_income_histogram'] = {
    'labels': [f'{log_edges[i]:.1f}' for i in range(len(log_counts))],
    'values': log_counts.tolist(),
}

# 3. Income by Education Level
edu_stats = df.groupby('Education_Level')['Income'].agg(['median', 'mean', 'count']).reset_index()
edu_order = ["High School", "Bachelor's", "Master's", "Doctorate"]
edu_stats['order'] = edu_stats['Education_Level'].map({v: i for i, v in enumerate(edu_order)})
edu_stats = edu_stats.sort_values('order')
chart_data['income_by_education'] = {
    'labels': edu_stats['Education_Level'].tolist(),
    'median': edu_stats['median'].tolist(),
    'mean': edu_stats['mean'].round(0).tolist(),
    'count': edu_stats['count'].tolist(),
}

# 4. Income by Occupation
occ_stats = df.groupby('Occupation')['Income'].agg(['median', 'mean', 'count']).reset_index()
occ_stats = occ_stats.sort_values('median', ascending=False)
chart_data['income_by_occupation'] = {
    'labels': occ_stats['Occupation'].tolist(),
    'median': occ_stats['median'].tolist(),
    'mean': occ_stats['mean'].round(0).tolist(),
    'count': occ_stats['count'].tolist(),
}

# 5. Income by Location
loc_stats = df.groupby('Location')['Income'].agg(['median', 'mean', 'count']).reset_index()
loc_stats = loc_stats.sort_values('median', ascending=False)
chart_data['income_by_location'] = {
    'labels': loc_stats['Location'].tolist(),
    'median': loc_stats['median'].tolist(),
    'mean': loc_stats['mean'].round(0).tolist(),
    'count': loc_stats['count'].tolist(),
}

# 6. Income by Employment Status
emp_stats = df.groupby('Employment_Status')['Income'].agg(['median', 'mean', 'count']).reset_index()
emp_stats = emp_stats.sort_values('median', ascending=False)
chart_data['income_by_employment'] = {
    'labels': emp_stats['Employment_Status'].tolist(),
    'median': emp_stats['median'].tolist(),
    'mean': emp_stats['mean'].round(0).tolist(),
    'count': emp_stats['count'].tolist(),
}

# 7. Income by Marital Status
mar_stats = df.groupby('Marital_Status')['Income'].agg(['median', 'mean', 'count']).reset_index()
mar_stats = mar_stats.sort_values('median', ascending=False)
chart_data['income_by_marital'] = {
    'labels': mar_stats['Marital_Status'].tolist(),
    'median': mar_stats['median'].tolist(),
    'mean': mar_stats['mean'].round(0).tolist(),
    'count': mar_stats['count'].tolist(),
}

# 8. Income by Gender
gen_stats = df.groupby('Gender')['Income'].agg(['median', 'mean', 'count']).reset_index()
chart_data['income_by_gender'] = {
    'labels': gen_stats['Gender'].tolist(),
    'median': gen_stats['median'].tolist(),
    'mean': gen_stats['mean'].round(0).tolist(),
    'count': gen_stats['count'].tolist(),
}

# 9. Income by Housing Type
hou_stats = df.groupby('Type_of_Housing')['Income'].agg(['median', 'mean', 'count']).reset_index()
hou_stats = hou_stats.sort_values('median', ascending=False)
chart_data['income_by_housing'] = {
    'labels': hou_stats['Type_of_Housing'].tolist(),
    'median': hou_stats['median'].tolist(),
    'mean': hou_stats['mean'].round(0).tolist(),
    'count': hou_stats['count'].tolist(),
}

# 10. Income by Homeownership
home_stats = df.groupby('Homeownership_Status')['Income'].agg(['median', 'mean', 'count']).reset_index()
chart_data['income_by_homeownership'] = {
    'labels': home_stats['Homeownership_Status'].tolist(),
    'median': home_stats['median'].tolist(),
    'mean': home_stats['mean'].round(0).tolist(),
    'count': home_stats['count'].tolist(),
}

# 11. Age distribution
age_counts, age_edges = np.histogram(df['Age'], bins=20)
chart_data['age_histogram'] = {
    'labels': [f'{int(age_edges[i])}-{int(age_edges[i+1])}' for i in range(len(age_counts))],
    'values': age_counts.tolist(),
}

# 12. Categorical distributions (for pie/doughnut charts)
for col in ['Education_Level', 'Occupation', 'Location', 'Employment_Status', 'Gender', 'Marital_Status']:
    counts = df[col].value_counts()
    chart_data[f'dist_{col.lower()}'] = {
        'labels': counts.index.tolist(),
        'values': counts.values.tolist(),
    }

# 13. Percentile data for income
percentiles = list(range(5, 100, 5))
chart_data['income_percentiles'] = {
    'labels': [f'P{p}' for p in percentiles],
    'values': [float(df['Income'].quantile(p / 100)) for p in percentiles],
}

# 14. Income stats summary
chart_data['summary_stats'] = {
    'total_records': len(df),
    'mean_income': float(df['Income'].mean()),
    'median_income': float(df['Income'].median()),
    'std_income': float(df['Income'].std()),
    'min_income': float(df['Income'].min()),
    'max_income': float(df['Income'].max()),
    'skewness': float(df['Income'].skew()),
    'kurtosis': float(df['Income'].kurtosis()),
    'mean_age': float(df['Age'].mean()),
    'mean_experience': float(df['Work_Experience'].mean()),
    'mean_household_size': float(df['Household_Size'].mean()),
}

with open('static/chart_data.json', 'w', encoding='utf-8') as f:
    json.dump(chart_data, f, indent=2)

print("Chart data generated: static/chart_data.json")
print(f"  {len(chart_data)} chart datasets")
