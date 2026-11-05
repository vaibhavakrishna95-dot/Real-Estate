import pandas as pd
import numpy as np

train_path = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\data\Housing-project-train-data.csv'
test_path = r'C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction\data\Hosuing-project-test-data.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")
print("\nTrain columns:", list(df_train.columns))
print("\nTest columns:", list(df_test.columns))

# Check difference in columns
diff_cols = set(df_train.columns) - set(df_test.columns)
print(f"\nColumns in train but not test: {diff_cols}")

# Target summary
if 'SalePrice' in df_train.columns:
    print("\nSalePrice summary:")
    print(df_train['SalePrice'].describe())

print("\nMissing values in train (top 15):")
print(df_train.isnull().sum()[df_train.isnull().sum() > 0].sort_values(ascending=False).head(15))

print("\nMissing values in test (top 15):")
print(df_test.isnull().sum()[df_test.isnull().sum() > 0].sort_values(ascending=False).head(15))
