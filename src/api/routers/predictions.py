# api/routers/predictions.py

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from src.api.models import PredictionRequest
from src.api.dependencies import get_model_data

router = APIRouter()

@router.post("/predict", response_model=Dict[str, float])
def get_predictions(request: PredictionRequest, model_data: Dict = Depends(get_model_data)):
    """Provides personalized predicted ratings for a given user and a list of conferences."""
    if "prediction_matrix" not in model_data or model_data["prediction_matrix"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Service is unavailable.")
    if request.user_id not in model_data["user_map"]:
        raise HTTPException(status_code=404, detail=f"User '{request.user_id}' not found.")

    user_idx = model_data["user_map"][request.user_id]
    predictions = {}
    for conf_id in request.conference_ids:
        item_idx = model_data["item_map"].get(conf_id)
        if item_idx is not None:
            score = model_data["prediction_matrix"][user_idx, item_idx]
            predictions[conf_id] = round(score, 4)
        else:
            predictions[conf_id] = 2.5 # Neutral default for unknown items
    return predictions

@router.get("/health")
def health_check(model_data: Dict = Depends(get_model_data)):
    """Checks if the server is running and the model is loaded."""
    if "prediction_matrix" in model_data and model_data["prediction_matrix"] is not None:
        return {"status": "ok", "model_loaded": True}
    return {"status": "degraded", "model_loaded": False}