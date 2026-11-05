# Ames Real Estate AI Valuation & Analytics Platform

An end-to-end Machine Learning Solution and interactive Web Application built on the comprehensive 80-feature Ames Housing dataset.

---

## 📌 Project Overview & Source Data

- **Google Drive Data Source**: [Dataset Folder Link](https://drive.google.com/drive/folders/1lXB1PUJThjo6DnnoH0QdvRf_qHMjJm2M?usp=sharing)
- **Files Ingested**:
  - `Housing-project-train-data.csv` (1,168 properties, 81 columns including `SalePrice`)
  - `Hosuing-project-test-data.csv` (292 properties, 80 columns)
  - `Housing-Data-Description.txt` (Complete documentation for all 80 attributes)
- **Output Predictions**: `predictions_test.csv` (Predictions generated for all 292 test houses)

---

## 🏗️ Project Architecture & Directory Structure

```text
housing-price-prediction/
├── data/
│   ├── Housing-project-train-data.csv      # Training dataset (1,168 rows)
│   ├── Hosuing-project-test-data.csv       # Test dataset (292 rows)
│   ├── Housing-Data-Description.txt        # Full attribute glossary
│   └── predictions_test.csv                # Model predictions on test set
├── artifacts/
│   ├── model_metadata.json                 # Evaluation metrics & top features
│   ├── eda_data.json                       # Pre-aggregated charts & data dictionary
│   ├── scaler.joblib                       # RobustScaler transformer
│   └── trained_models.joblib               # Serialized trained model ensemble
├── static/
│   ├── css/style.css                       # Modern executive UI design
│   └── js/app.js                           # Interactive charts & app logic
├── templates/
│   └── index.html                          # Single Page Web Application
├── train_models.py                         # 5-Fold CV training & ensembling
├── generate_eda_and_dict.py                # EDA aggregation & dictionary parser
├── predict_service.py                      # Real-time inference engine
├── main.py                                 # FastAPI backend application
├── start_website.bat                       # One-click Windows launcher
└── README.md                               # Project documentation
```

---

## 🧠 Machine Learning Methodology

### 1. Data Cleaning & Domain-Specific Imputation
- Distinguished **true absent features** (e.g. `None` for missing Garage, Fireplace, Basement, Pool, Alley, Fence) from missing values to prevent invalid default imputation.
- Group-based median imputation for `LotFrontage` using neighboring properties.
- Proper zero-imputation for numerical garage and basement metrics when the structure is absent.

### 2. Feature Engineering
- **Total Square Footage**: `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- **Total Bathrooms**: `TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
- **Composite Porch Area**: Aggregation of wood deck, open porch, enclosed porch, 3-season, and screen porch.
- **Property Age Features**: `HouseAge = YrSold - YearBuilt`, `RemodAge = YrSold - YearRemodAdd`.
- **Non-Linear Interactions**: `Qual_TotalSF = OverallQual * TotalSF` and `Qual_GrLivArea = OverallQual * GrLivArea`.
- **Ordinal Quality Encodings**: Converted qualitative metrics (`Ex`, `Gd`, `TA`, `Fa`, `Po`) into numeric integer scores.

### 3. Target Transformation & Metric Optimization
- House prices exhibit significant positive skewness (\(\gamma_1 = 1.88\)).
- Trained all models on \(\log(1 + \text{SalePrice})\) to stabilize variance, equalize relative percent errors across budget and luxury segments, and optimize for **RMSLE** (Root Mean Squared Logarithmic Error).

---

## 🏆 Model Performance & 5-Fold Cross-Validation Leaderboard

| Model Algorithm | 5-Fold CV RMSLE | 5-Fold CV RMSE ($) | 5-Fold CV MAE ($) | R² Score | Role |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble Blend** 👑 | **0.1279** | **$28,152.72** | **$15,159.56** | **0.8732** | **Production Ensemble** |
| **XGBoost Regressor** | **0.1282** | **$26,127.54** | **$15,706.80** | **0.8873** | Base Model (Weight: 25%) |
| **Gradient Boosting** | **0.1292** | **$26,286.82** | **$15,729.20** | **0.8851** | Base Model (Weight: 25%) |
| **LightGBM Regressor**| **0.1314** | **$27,082.18** | **$16,294.54** | **0.8811** | Base Model (Weight: 25%) |
| **Random Forest** | 0.1448 | $29,919.26 | $17,882.65 | 0.8535 | Non-linear Tree Baseline |
| **Ridge Regression** | 0.1467 | $44,579.08 | $17,973.57 | 0.4645 | Regularized L2 (Weight: 10%) |
| **ElasticNet** | 0.1479 | $46,960.36 | $18,168.58 | 0.3690 | L1+L2 (Weight: 5%) |
| **Lasso Regression** | 0.1515 | $52,411.21 | $18,606.69 | 0.1042 | Regularized L1 (Weight: 10%) |

---

## 📊 Top 10 Feature Importances

1. **`Qual_TotalSF`** (Overall Quality \(\times\) Total Square Footage) — **30.2%**
2. **`Qual_GrLivArea`** (Overall Quality \(\times\) Living Area) — **6.87%**
3. **`LotArea`** (Lot size in square feet) — **2.62%**
4. **`TotalSF`** (Combined basement + 1st + 2nd floor area) — **2.02%**
5. **`TotalBath`** (Total composite bathrooms) — **1.98%**
6. **`Cond_Qual`** (Overall Condition \(\times\) Overall Quality) — **1.77%**
7. **`GarageArea`** (Size of garage in square feet) — **1.72%**
8. **`1stFlrSF`** (First floor square footage) — **1.66%**
9. **`KitchenQual_Score`** (Kitchen finish quality score) — **1.61%**
10. **`BsmtFinSF1`** (Type 1 finished basement square feet) — **1.57%**

---

## 🌐 Web Application Features

The website is running on `http://127.0.0.1:8000`:

1. **Property Valuation Estimator**:
   - Interactive property appraisal tool with instant real-time predictions.
   - 1-click **Representative Presets**:
     - *Northridge Luxury Estate* ($450k+ tier)
     - *College Creek Family Home* ($215k tier)
     - *Old Town Historic Craftsman* ($145k tier)
     - *Ames Starter Bungalow* ($115k tier)
   - Real-time confidence band, price per square foot, and consensus comparison chart across all 7 models.
2. **Market Analytics (EDA)**:
   - Interactive Chart.js charts: Price distribution, Neighborhood median rankings, Quality vs Price curves, Living Area scatter with quality color mapping, and Decade of construction trends.
3. **Test Predictions Explorer (292 Properties)**:
   - Searchable and filterable data table of all 292 test homes.
   - Ability to load any test home directly into the Valuation Estimator with 1 click.
   - Direct button to download `predictions_test.csv`.
4. **Model Architecture & Benchmarks**:
   - 5-Fold Cross Validation leaderboard and interactive horizontal feature importance chart.
5. **Data Dictionary**:
   - Searchable glossaries for all 79 attributes parsed from `Housing-Data-Description.txt`.

---

## 🚀 How to Run the Website

### Option 1: Double-Click Batch Launcher
Double-click `start_website.bat` in the project folder.

### Option 2: Command Line
```powershell
cd "C:\Users\VAIBHAV KRISHNA S K\.gemini\antigravity\scratch\housing-price-prediction"
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in any modern web browser.
