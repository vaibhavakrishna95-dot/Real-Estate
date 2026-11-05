import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

DATA_DIR = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\data'
OUTPUT_DIR = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, 'Housing-project-train-data.csv')
test_path = os.path.join(DATA_DIR, 'Hosuing-project-test-data.csv')

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"Loaded train shape: {df_train.shape}")
print(f"Loaded test shape: {df_test.shape}")

# Target variable
y_train_orig = df_train['SalePrice'].values
y_train_log = np.log1p(y_train_orig)

# Separate features
X_train_raw = df_train.drop(columns=['SalePrice'])
X_test_raw = df_test.copy()

# Store IDs
train_ids = df_train['Id'].values
test_ids = df_test['Id'].values

# Combine for consistent feature engineering and encoding
combined = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
print(f"Combined shape: {combined.shape}")

# 1. Handling Missing Values properly according to Housing-Data-Description
# Many NA values actually mean "None" / absent feature
none_cols = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType'
]
for col in none_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna('None')

# Numerical garage / basement / masonry features that are 0 when feature is None
zero_cols = [
    'GarageYrBlt', 'GarageCars', 'GarageArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
]
for col in zero_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna(0)

# LotFrontage is often correlated with Neighborhood
if 'LotFrontage' in combined.columns and 'Neighborhood' in combined.columns:
    combined['LotFrontage'] = combined.groupby('Neighborhood')['LotFrontage'].transform(
        lambda val: val.fillna(val.median() if not val.median() != val.median() else 60)
    )
    combined['LotFrontage'] = combined['LotFrontage'].fillna(combined['LotFrontage'].median())

# Other categorical columns: fill with mode
mode_cols = ['MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities', 'Functional']
for col in mode_cols:
    if col in combined.columns:
        mode_val = combined[col].mode()[0] if not combined[col].mode().empty else 'None'
        combined[col] = combined[col].fillna(mode_val)

# 2. Feature Engineering
# Total square feet
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined['TotalBath'] = (combined['FullBath'] + 0.5 * combined['HalfBath'] + 
                        combined['BsmtFullBath'] + 0.5 * combined['BsmtHalfBath'])
combined['TotalPorchSF'] = (combined['OpenPorchSF'] + combined['EnclosedPorch'] + 
                           combined['3SsnPorch'] + combined['ScreenPorch'] + combined['WoodDeckSF'])

# Age features (reference YrSold, or max YearBuilt if YrSold < YearBuilt due to typo)
combined['HouseAge'] = combined['YrSold'] - combined['YearBuilt']
combined['HouseAge'] = combined['HouseAge'].apply(lambda x: max(x, 0))

combined['RemodAge'] = combined['YrSold'] - combined['YearRemodAdd']
combined['RemodAge'] = combined['RemodAge'].apply(lambda x: max(x, 0))

combined['IsRemodeled'] = (combined['YearRemodAdd'] != combined['YearBuilt']).astype(int)
combined['IsNew'] = (combined['YearBuilt'] == combined['YrSold']).astype(int)

# Binary indicators
combined['HasPool'] = (combined['PoolArea'] > 0).astype(int)
combined['Has2ndFloor'] = (combined['2ndFlrSF'] > 0).astype(int)
combined['HasGarage'] = (combined['GarageArea'] > 0).astype(int)
combined['HasBsmt'] = (combined['TotalBsmtSF'] > 0).astype(int)
combined['HasFireplace'] = (combined['Fireplaces'] > 0).astype(int)

# Quality interaction
combined['Qual_TotalSF'] = combined['OverallQual'] * combined['TotalSF']
combined['Qual_GrLivArea'] = combined['OverallQual'] * combined['GrLivArea']
combined['Cond_Qual'] = combined['OverallCond'] * combined['OverallQual']

# Convert MSSubClass to string (categorical)
combined['MSSubClass'] = combined['MSSubClass'].astype(str)

# Quality ordinal mapping
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for qcol in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:
    if qcol in combined.columns:
        combined[qcol + '_Score'] = combined[qcol].map(lambda x: qual_map.get(x, 0))

# Drop Id column for training
features_df = combined.drop(columns=['Id'])

# One-hot encode categorical features
features_encoded = pd.get_dummies(features_df, drop_first=True)
print(f"Features encoded shape: {features_encoded.shape}")

# Split back to train and test
n_train = len(df_train)
X_train = features_encoded.iloc[:n_train].copy()
X_test = features_encoded.iloc[n_train:].copy()

# Robust scaling of numerical columns
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Define 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model(name, model, X, y_log, y_orig):
    rmse_scores = []
    rmsle_scores = []
    mae_scores = []
    r2_scores = []
    
    oof_preds_log = np.zeros(len(y_log))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_log[train_idx], y_log[val_idx]
        y_val_orig = y_orig[val_idx]
        
        model.fit(X_tr, y_tr)
        pred_log = model.predict(X_val)
        oof_preds_log[val_idx] = pred_log
        pred_orig = np.expm1(pred_log)
        
        # calculate metrics
        rmsle = np.sqrt(mean_squared_error(y_val, pred_log))
        rmse = np.sqrt(mean_squared_error(y_val_orig, pred_orig))
        mae = mean_absolute_error(y_val_orig, pred_orig)
        r2 = r2_score(y_val_orig, pred_orig)
        
        rmsle_scores.append(rmsle)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
    return {
        'name': name,
        'rmsle_mean': float(np.mean(rmsle_scores)),
        'rmse_mean': float(np.mean(rmse_scores)),
        'mae_mean': float(np.mean(mae_scores)),
        'r2_mean': float(np.mean(r2_scores)),
        'oof_preds_log': oof_preds_log
    }

models = {
    'Ridge': Ridge(alpha=15.0, random_state=42),
    'Lasso': Lasso(alpha=0.0005, random_state=42, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000),
    'RandomForest': RandomForestRegressor(n_estimators=250, max_depth=16, min_samples_split=4, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.04, max_depth=4, subsample=0.85, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=350, learning_rate=0.03, num_leaves=28, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=350, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
}

results = []
oof_dict = {}

print("\n--- Training and Evaluating Models (5-Fold CV) ---")
for name, model in models.items():
    res = evaluate_model(name, model, X_train_scaled, y_train_log, y_train_orig)
    oof_dict[name] = res['oof_preds_log']
    results.append({
        'name': name,
        'RMSLE': round(res['rmsle_mean'], 4),
        'RMSE ($)': round(res['rmse_mean'], 2),
        'MAE ($)': round(res['mae_mean'], 2),
        'R2': round(res['r2_mean'], 4)
    })
    print(f"[{name}] RMSLE: {res['rmsle_mean']:.4f} | RMSE: ${res['rmse_mean']:,.2f} | MAE: ${res['mae_mean']:,.2f} | R²: {res['r2_mean']:.4f}")

# Weighted Blended Ensemble
# Weights based on performance
weights = {
    'Ridge': 0.10,
    'Lasso': 0.10,
    'ElasticNet': 0.05,
    'GradientBoosting': 0.25,
    'LightGBM': 0.25,
    'XGBoost': 0.25
}

ensemble_oof_log = np.zeros(len(y_train_log))
for name, w in weights.items():
    ensemble_oof_log += w * oof_dict[name]

ens_pred_orig = np.expm1(ensemble_oof_log)
ens_rmsle = np.sqrt(mean_squared_error(y_train_log, ensemble_oof_log))
ens_rmse = np.sqrt(mean_squared_error(y_train_orig, ens_pred_orig))
ens_mae = mean_absolute_error(y_train_orig, ens_pred_orig)
ens_r2 = r2_score(y_train_orig, ens_pred_orig)

results.append({
    'name': 'Ensemble Blend',
    'RMSLE': round(float(ens_rmsle), 4),
    'RMSE ($)': round(float(ens_rmse), 2),
    'MAE ($)': round(float(ens_mae), 2),
    'R2': round(float(ens_r2), 4)
})
print(f"\n[Ensemble Blend] RMSLE: {ens_rmsle:.4f} | RMSE: ${ens_rmse:,.2f} | MAE: ${ens_mae:,.2f} | R²: {ens_r2:.4f}")

# Fit all models on full training data and predict on test
trained_models = {}
test_preds_log = np.zeros(len(X_test))

for name, model in models.items():
    model.fit(X_train_scaled, y_train_log)
    trained_models[name] = model
    if name in weights:
        pred_test = model.predict(X_test_scaled)
        test_preds_log += weights[name] * pred_test

test_preds_orig = np.expm1(test_preds_log)

# Save predictions to CSV
df_predictions = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': np.round(test_preds_orig, 2)
})
pred_path = os.path.join(DATA_DIR, 'predictions_test.csv')
df_predictions.to_csv(pred_path, index=False)
print(f"\nSaved test predictions to {pred_path} ({len(df_predictions)} rows)")

# Feature Importances from XGBoost & LightGBM & GradientBoosting
feature_names = features_encoded.columns.tolist()
xgb_importances = trained_models['XGBoost'].feature_importances_
lgb_importances = trained_models['LightGBM'].feature_importances_
gb_importances = trained_models['GradientBoosting'].feature_importances_

# Normalize importances
xgb_norm = xgb_importances / np.sum(xgb_importances)
lgb_norm = lgb_importances / np.sum(lgb_importances)
gb_norm = gb_importances / np.sum(gb_importances)
avg_importance = (xgb_norm + lgb_norm + gb_norm) / 3.0

top_feat_indices = np.argsort(avg_importance)[::-1][:20]
top_features = [
    {'feature': feature_names[i], 'importance': float(avg_importance[i])}
    for i in top_feat_indices
]

# Save pipeline artifacts
artifacts = {
    'feature_names': feature_names,
    'weights': weights,
    'results': results,
    'top_features': top_features,
    'train_stats': {
        'count': int(len(df_train)),
        'mean_price': float(np.mean(y_train_orig)),
        'median_price': float(np.median(y_train_orig)),
        'min_price': float(np.min(y_train_orig)),
        'max_price': float(np.max(y_train_orig)),
        'std_price': float(np.std(y_train_orig))
    },
    'test_stats': {
        'count': int(len(df_predictions)),
        'mean_price': float(np.mean(test_preds_orig)),
        'median_price': float(np.median(test_preds_orig)),
        'min_price': float(np.min(test_preds_orig)),
        'max_price': float(np.max(test_preds_orig)),
        'std_price': float(np.std(test_preds_orig))
    }
}

with open(os.path.join(OUTPUT_DIR, 'model_metadata.json'), 'w') as f:
    json.dump(artifacts, f, indent=2)

# Save scaler and models
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.joblib'))
joblib.dump(trained_models, os.path.join(OUTPUT_DIR, 'trained_models.joblib'))

print("All models, metadata, and predictions successfully saved!")
