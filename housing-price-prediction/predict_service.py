import os
import json
import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\artifacts'
DATA_DIR = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\data'

# Load metadata, scaler, trained models
with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'scaler.joblib'))
trained_models = joblib.load(os.path.join(ARTIFACTS_DIR, 'trained_models.joblib'))
feature_names = metadata['feature_names']
weights = metadata['weights']

# Load train dataframe to get medians and modes for clean fallback
train_df = pd.read_csv(os.path.join(DATA_DIR, 'Housing-project-train-data.csv'))

none_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]

zero_cols = [
    'GarageYrBlt', 'GarageCars', 'GarageArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]

mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities', 'Functional', 'SaleCondition', 'HouseStyle', 'BldgType', 'Foundation']

# Medians for numerical
numeric_medians = {}
for col in train_df.select_dtypes(include=[np.number]).columns:
    if col != 'SalePrice' and col != 'Id':
        numeric_medians[col] = float(train_df[col].median(skipna=True))

qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

def preprocess_and_predict(house_dict: dict) -> dict:
    row = {}
    
    # Fill defaults from numeric_medians and modes
    for col, med in numeric_medians.items():
        row[col] = med
    for col in none_cols:
        row[col] = 'None'
    for col in zero_cols:
        row[col] = 0
    for col in mode_cols:
        if col in train_df.columns:
            row[col] = train_df[col].mode()[0]
            
    # Default other categoricals
    row['Neighborhood'] = 'CollgCr'
    row['LotConfig'] = 'Inside'
    row['LandContour'] = 'Lvl'
    row['LotShape'] = 'Reg'
    row['Street'] = 'Pave'
    row['LandSlope'] = 'Gtl'
    row['Condition1'] = 'Norm'
    row['Condition2'] = 'Norm'
    row['RoofStyle'] = 'Gable'
    row['RoofMatl'] = 'CompShg'
    row['Heating'] = 'GasA'
    row['CentralAir'] = 'Y'
    row['PavedDrive'] = 'Y'
    row['YrSold'] = 2010
    row['MoSold'] = 6
    row['MSSubClass'] = '20'

    # Apply user overrides
    row.update(house_dict)

    # Ensure consistency between floor areas and GrLivArea
    gr_liv = float(row.get('GrLivArea', 1500))
    first_flr = float(row.get('1stFlrSF', 0))
    second_flr = float(row.get('2ndFlrSF', 0))
    if first_flr == 0 and second_flr == 0:
        first_flr = gr_liv
        row['1stFlrSF'] = first_flr
        row['2ndFlrSF'] = 0.0
    elif first_flr > 0 and second_flr > 0:
        gr_liv = first_flr + second_flr
        row['GrLivArea'] = gr_liv

    df_row = pd.DataFrame([row])

    # Clean NaNs
    for col in none_cols:
        if col in df_row.columns:
            df_row[col] = df_row[col].fillna('None')
    for col in zero_cols:
        if col in df_row.columns:
            df_row[col] = df_row[col].fillna(0)
    for col in mode_cols:
        if col in df_row.columns:
            df_row[col] = df_row[col].fillna(train_df[col].mode()[0] if not train_df[col].mode().empty else 'None')

    # Feature Engineering
    df_row['TotalSF'] = df_row['TotalBsmtSF'] + df_row['1stFlrSF'] + df_row['2ndFlrSF']
    df_row['TotalBath'] = (df_row['FullBath'] + 0.5 * df_row['HalfBath'] + 
                          df_row['BsmtFullBath'] + 0.5 * df_row['BsmtHalfBath'])
    df_row['TotalPorchSF'] = (df_row['OpenPorchSF'] + df_row['EnclosedPorch'] + 
                             df_row['3SsnPorch'] + df_row['ScreenPorch'] + df_row['WoodDeckSF'])
    
    yr_sold = float(df_row['YrSold'].iloc[0])
    df_row['HouseAge'] = max(0.0, yr_sold - float(df_row['YearBuilt'].iloc[0]))
    df_row['RemodAge'] = max(0.0, yr_sold - float(df_row['YearRemodAdd'].iloc[0]))
    df_row['IsRemodeled'] = int(df_row['YearRemodAdd'].iloc[0] != df_row['YearBuilt'].iloc[0])
    df_row['IsNew'] = int(df_row['YearBuilt'].iloc[0] == yr_sold)
    
    df_row['HasPool'] = int(df_row['PoolArea'].iloc[0] > 0)
    df_row['Has2ndFloor'] = int(df_row['2ndFlrSF'].iloc[0] > 0)
    df_row['HasGarage'] = int(df_row['GarageArea'].iloc[0] > 0)
    df_row['HasBsmt'] = int(df_row['TotalBsmtSF'].iloc[0] > 0)
    df_row['HasFireplace'] = int(df_row['Fireplaces'].iloc[0] > 0)
    
    df_row['Qual_TotalSF'] = float(df_row['OverallQual'].iloc[0]) * float(df_row['TotalSF'].iloc[0])
    df_row['Qual_GrLivArea'] = float(df_row['OverallQual'].iloc[0]) * float(df_row['GrLivArea'].iloc[0])
    df_row['Cond_Qual'] = float(df_row['OverallCond'].iloc[0]) * float(df_row['OverallQual'].iloc[0])
    
    df_row['MSSubClass'] = str(df_row['MSSubClass'].iloc[0])
    
    for qcol in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
        if qcol in df_row.columns:
            df_row[qcol + '_Score'] = qual_map.get(str(df_row[qcol].iloc[0]), 3)

    # Encode
    row_encoded = pd.get_dummies(df_row)
    
    # Align to trained feature space
    aligned = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for col in feature_names:
        if col in row_encoded.columns:
            val = row_encoded[col].iloc[0]
            aligned[col] = float(val) if not pd.isna(val) else 0.0
            
    # Check any leftover NaNs and fill with 0.0
    aligned = aligned.fillna(0.0)

    # Scale
    scaled_feats = scaler.transform(aligned)
    
    # Model predictions
    model_predictions = {}
    blended_pred_log = 0.0
    
    for name, model in trained_models.items():
        p_log = model.predict(scaled_feats)[0]
        p_val = float(np.expm1(p_log))
        model_predictions[name] = round(p_val, 2)
        if name in weights:
            blended_pred_log += weights[name] * p_log
            
    blended_price = float(np.expm1(blended_pred_log))
    
    low_price = round(blended_price * 0.935, 2)
    high_price = round(blended_price * 1.065, 2)
    price_per_sqft = round(blended_price / gr_liv, 2) if gr_liv > 0 else 0
    
    return {
        'predicted_price': round(blended_price, 2),
        'low_price': low_price,
        'high_price': high_price,
        'price_per_sqft': price_per_sqft,
        'model_breakdown': model_predictions,
        'inputs_summary': {
            'living_area_sqft': gr_liv,
            'total_sqft': float(df_row['TotalSF'].iloc[0]),
            'overall_quality': int(df_row['OverallQual'].iloc[0]),
            'year_built': int(df_row['YearBuilt'].iloc[0]),
            'neighborhood': str(df_row['Neighborhood'].iloc[0]),
            'bedrooms': int(df_row['BedroomAbvGr'].iloc[0]),
            'bathrooms': float(df_row['TotalBath'].iloc[0]),
            'garage_cars': int(df_row['GarageCars'].iloc[0])
        }
    }

if __name__ == '__main__':
    sample = {
        'OverallQual': 8,
        'GrLivArea': 2200,
        'TotalBsmtSF': 1100,
        '1stFlrSF': 1100,
        '2ndFlrSF': 1100,
        'YearBuilt': 2005,
        'YearRemodAdd': 2005,
        'Neighborhood': 'NridgHt',
        'GarageCars': 3,
        'GarageArea': 750,
        'FullBath': 2,
        'HalfBath': 1,
        'KitchenQual': 'Gd'
    }
    res = preprocess_and_predict(sample)
    print("Inference Success! Predicted:")
    print(json.dumps(res, indent=2))
