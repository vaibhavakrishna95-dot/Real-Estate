import os
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict_service import preprocess_and_predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
DATA_DIR = os.path.join(BASE_DIR, 'data')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

app = FastAPI(
    title="Ames Housing Price AI Valuation Platform",
    description="End-to-end Machine Learning Solution & Analytics Dashboard for Ames Housing dataset",
    version="1.0.0"
)

# Enable CORS for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load cached data
with open(os.path.join(ARTIFACTS_DIR, 'model_metadata.json'), 'r', encoding='utf-8') as f:
    metadata_cache = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, 'eda_data.json'), 'r', encoding='utf-8') as f:
    eda_cache = json.load(f)

test_records = eda_cache.get('test_preview', [])

class PredictRequest(BaseModel):
    Neighborhood: Optional[str] = "CollgCr"
    OverallQual: Optional[int] = 7
    OverallCond: Optional[int] = 5
    YearBuilt: Optional[int] = 2005
    YearRemodAdd: Optional[int] = 2005
    GrLivArea: Optional[float] = 1800.0
    TotalBsmtSF: Optional[float] = 900.0
    firstFlrSF: Optional[float] = 900.0
    secondFlrSF: Optional[float] = 900.0
    LotArea: Optional[float] = 9500.0
    LotFrontage: Optional[float] = 70.0
    FullBath: Optional[int] = 2
    HalfBath: Optional[int] = 1
    BsmtFullBath: Optional[int] = 0
    BedroomAbvGr: Optional[int] = 3
    KitchenQual: Optional[str] = "Gd"
    TotRmsAbvGrd: Optional[int] = 7
    Fireplaces: Optional[int] = 1
    GarageCars: Optional[int] = 2
    GarageArea: Optional[float] = 500.0
    CentralAir: Optional[str] = "Y"
    HeatingQC: Optional[str] = "Ex"
    ExterQual: Optional[str] = "Gd"
    WoodDeckSF: Optional[float] = 120.0
    OpenPorchSF: Optional[float] = 50.0
    MSSubClass: Optional[str] = "60"
    BldgType: Optional[str] = "1Fam"
    HouseStyle: Optional[str] = "2Story"
    PoolArea: Optional[float] = 0.0

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(TEMPLATES_DIR, 'index.html')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Ames Housing AI Valuation Platform</h1><p>Frontend template not found.</p>")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": ["Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting", "LightGBM", "XGBoost", "Ensemble Blend"],
        "train_samples": metadata_cache["train_stats"]["count"],
        "test_samples": metadata_cache["test_stats"]["count"]
    }

@app.get("/api/metadata")
async def get_metadata():
    return metadata_cache

@app.get("/api/eda")
async def get_eda():
    return {
        "price_distribution": eda_cache.get("price_distribution"),
        "neighborhood_stats": eda_cache.get("neighborhood_stats"),
        "qual_stats": eda_cache.get("qual_stats"),
        "scatter_living_area": eda_cache.get("scatter_living_area"),
        "decade_stats": eda_cache.get("decade_stats"),
        "top_correlations": eda_cache.get("top_correlations")
    }

@app.get("/api/dictionary")
async def get_dictionary():
    return eda_cache.get("feature_dictionary", {})

@app.get("/api/test-data")
async def get_test_data(
    search: Optional[str] = Query(None, description="Search by ID or Neighborhood"),
    neighborhood: Optional[str] = Query(None, description="Filter by neighborhood"),
    min_price: Optional[float] = Query(None, description="Min predicted price"),
    max_price: Optional[float] = Query(None, description="Max predicted price"),
    sort_by: Optional[str] = Query("id", description="Field to sort by"),
    order: Optional[str] = Query("asc", description="Sort order: asc or desc"),
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=292)
):
    filtered = list(test_records)

    # Search filter
    if search:
        s = search.lower().strip()
        filtered = [
            r for r in filtered 
            if s in str(r['id']).lower() or s in r['neighborhood'].lower()
        ]

    # Neighborhood filter
    if neighborhood and neighborhood.lower() != "all":
        filtered = [r for r in filtered if r['neighborhood'].lower() == neighborhood.lower()]

    # Price range filter
    if min_price is not None:
        filtered = [r for r in filtered if r['predictedPrice'] >= min_price]
    if max_price is not None:
        filtered = [r for r in filtered if r['predictedPrice'] <= max_price]

    # Sorting
    key_mapping = {
        'id': 'id',
        'neighborhood': 'neighborhood',
        'overallQual': 'overallQual',
        'yearBuilt': 'yearBuilt',
        'grLivArea': 'grLivArea',
        'predictedPrice': 'predictedPrice'
    }
    sort_key = key_mapping.get(sort_by, 'id')
    reverse = (order.lower() == 'desc')
    filtered.sort(key=lambda x: x.get(sort_key, 0), reverse=reverse)

    total_count = len(filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_items = filtered[start_idx:end_idx]

    return {
        "total": total_count,
        "page": page,
        "limit": limit,
        "pages": (total_count + limit - 1) // limit,
        "items": paginated_items
    }

@app.post("/api/predict")
async def predict_price(payload: Dict[str, Any]):
    try:
        # Convert any camelCase keys to PascalCase matching features
        converted_payload = {}
        for k, v in payload.items():
            if k == 'firstFlrSF':
                converted_payload['1stFlrSF'] = v
            elif k == 'secondFlrSF':
                converted_payload['2ndFlrSF'] = v
            else:
                converted_payload[k] = v
                
        result = preprocess_and_predict(converted_payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/download-predictions")
async def download_predictions():
    pred_file = os.path.join(DATA_DIR, 'predictions_test.csv')
    if os.path.exists(pred_file):
        return FileResponse(
            path=pred_file,
            filename="predictions_test.csv",
            media_type="text/csv"
        )
    raise HTTPException(status_code=404, detail="Prediction file not found.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
