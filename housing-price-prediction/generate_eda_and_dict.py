import os
import re
import json
import pandas as pd
import numpy as np

DATA_DIR = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\data'
OUTPUT_DIR = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\artifacts'

train_path = os.path.join(DATA_DIR, 'Housing-project-train-data.csv')
test_path = os.path.join(DATA_DIR, 'Hosuing-project-test-data.csv')
desc_path = os.path.join(DATA_DIR, 'Housing-Data-Description.txt')
pred_path = os.path.join(DATA_DIR, 'predictions_test.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_preds = pd.read_csv(pred_path)

# Merge predictions with test data for rich viewing
df_test_full = df_test.merge(df_preds, on='Id', how='left')

# 1. Price Distribution Data (Histogram bins)
prices = df_train['SalePrice'].values
hist, bin_edges = np.histogram(prices, bins=30)
price_distribution = {
    'bins': [f"${int(bin_edges[i]):,} - ${int(bin_edges[i+1]):,}" for i in range(len(hist))],
    'bin_centers': [round((bin_edges[i] + bin_edges[i+1])/2, 1) for i in range(len(hist))],
    'counts': [int(c) for c in hist],
    'mean': float(np.mean(prices)),
    'median': float(np.median(prices)),
    'skewness': float(df_train['SalePrice'].skew()),
    'kurtosis': float(df_train['SalePrice'].kurt())
}

# 2. Neighborhood Price Stats
neigh_stats = df_train.groupby('Neighborhood')['SalePrice'].agg(
    count='count',
    mean='mean',
    median='median',
    min='min',
    max='max',
    std='std'
).reset_index().sort_values('median', ascending=False)
neigh_stats_list = [
    {
        'neighborhood': row['Neighborhood'],
        'count': int(row['count']),
        'mean': round(float(row['mean']), 2),
        'median': round(float(row['median']), 2),
        'min': round(float(row['min']), 2),
        'max': round(float(row['max']), 2)
    }
    for _, row in neigh_stats.iterrows()
]

# 3. Overall Quality vs Price
qual_stats = df_train.groupby('OverallQual')['SalePrice'].agg(
    count='count',
    mean='mean',
    median='median',
    min='min',
    max='max'
).reset_index().sort_values('OverallQual')
qual_stats_list = [
    {
        'quality': int(row['OverallQual']),
        'count': int(row['count']),
        'mean': round(float(row['mean']), 2),
        'median': round(float(row['median']), 2),
        'min': round(float(row['min']), 2),
        'max': round(float(row['max']), 2)
    }
    for _, row in qual_stats.iterrows()
]

# 4. Living Area vs Price (Sample 300 points for fast rendering in chart)
sample_df = df_train[['Id', 'GrLivArea', 'OverallQual', 'Neighborhood', 'SalePrice', 'YearBuilt']].dropna().sample(
    min(400, len(df_train)), random_state=42
)
scatter_living_area = [
    {
        'id': int(row['Id']),
        'x': int(row['GrLivArea']),
        'y': int(row['SalePrice']),
        'quality': int(row['OverallQual']),
        'neighborhood': str(row['Neighborhood']),
        'yearBuilt': int(row['YearBuilt'])
    }
    for _, row in sample_df.iterrows()
]

# 5. Year Built Trend (by Decade)
df_train['Decade'] = (df_train['YearBuilt'] // 10) * 10
decade_stats = df_train.groupby('Decade')['SalePrice'].agg(
    count='count',
    mean='mean',
    median='median'
).reset_index().sort_values('Decade')
decade_stats_list = [
    {
        'decade': f"{int(row['Decade'])}s",
        'count': int(row['count']),
        'mean': round(float(row['mean']), 2),
        'median': round(float(row['median']), 2)
    }
    for _, row in decade_stats.iterrows()
]

# 6. Numerical Feature Correlations with SalePrice
numeric_cols = df_train.select_dtypes(include=[np.number]).columns
corrs = df_train[numeric_cols].corr()['SalePrice'].drop('SalePrice').sort_values(ascending=False)
correlations_list = [
    {'feature': col, 'correlation': round(float(corrs[col]), 4)}
    for col in corrs.index if not np.isnan(corrs[col])
][:20]

# 7. Parse Housing-Data-Description.txt into structured JSON dictionary
with open(desc_path, 'r', encoding='utf-8', errors='ignore') as f:
    desc_text = f.read()

# Pattern for feature sections: FeatureName: Description \n \t code value
feature_dict = {}
blocks = re.split(r'\n(?=[A-Za-z0-9_]+:\s)', desc_text)
for block in blocks:
    lines = block.strip().split('\n')
    if not lines or ':' not in lines[0]:
        continue
    header = lines[0].split(':', 1)
    feat_name = header[0].strip()
    feat_desc = header[1].strip()
    
    # parse possible categories / codes if present
    codes = []
    for line in lines[1:]:
        line_clean = line.strip()
        if not line_clean:
            continue
        # Check if tab or spaced split
        parts = re.split(r'\s{2,}|\t+', line_clean)
        if len(parts) >= 2:
            codes.append({'code': parts[0].strip(), 'meaning': ' '.join(parts[1:]).strip()})
        elif parts:
            codes.append({'code': parts[0].strip(), 'meaning': ''})
            
    feature_dict[feat_name] = {
        'description': feat_desc,
        'options': codes[:15]  # limit length if too long
    }

# Also save sample of test predictions for frontend display
test_preview = []
for _, row in df_test_full.iterrows():
    test_preview.append({
        'id': int(row['Id']),
        'neighborhood': str(row.get('Neighborhood', 'N/A')),
        'overallQual': int(row.get('OverallQual', 5)),
        'yearBuilt': int(row.get('YearBuilt', 1980)),
        'grLivArea': int(row.get('GrLivArea', 1500)),
        'totalBsmtSF': int(row.get('TotalBsmtSF', 0)) if not pd.isna(row.get('TotalBsmtSF')) else 0,
        'garageCars': int(row.get('GarageCars', 0)) if not pd.isna(row.get('GarageCars')) else 0,
        'fullBath': int(row.get('FullBath', 1)),
        'bedroomAbvGr': int(row.get('BedroomAbvGr', 3)),
        'predictedPrice': round(float(row['SalePrice']), 2)
    })

eda_data = {
    'price_distribution': price_distribution,
    'neighborhood_stats': neigh_stats_list,
    'qual_stats': qual_stats_list,
    'scatter_living_area': scatter_living_area,
    'decade_stats': decade_stats_list,
    'top_correlations': correlations_list,
    'test_preview': test_preview,
    'feature_dictionary': feature_dict
}

with open(os.path.join(OUTPUT_DIR, 'eda_data.json'), 'w', encoding='utf-8') as f:
    json.dump(eda_data, f, indent=2)

print(f"Generated and saved eda_data.json successfully! ({len(feature_dict)} features in dict, {len(test_preview)} test preview items)")
