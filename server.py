from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np

# Import our project's configuration and file handlers
import config.config as config
from src.utils.file_handlers import load_pickle

# --- 1. Global State and Lifespan Management ---

# This dictionary will hold our "live" model data. It's defined in the global scope.
model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This is the new, recommended way to manage startup and shutdown events.
    The code before the 'yield' statement is executed on startup.
    The code after the 'yield' statement is executed on shutdown.
    """
    # --- Startup Logic ---
    print("--- 🚀 API Server is starting up... ---")
    print(f"📂 Loading prediction model from: {config.MUTIFACTOR_PREDICTIONS_PATH}")
    
    # We load the mutifactor model by default as it's our best one.
    prediction_artifact = load_pickle(config.MUTIFACTOR_PREDICTIONS_PATH)
    
    if prediction_artifact is None:
        print("❌ CRITICAL ERROR: Could not load the prediction model artifact. The API will not be able to serve predictions.")
        # To prevent the server from running in a broken state, we can leave the model_data empty.
        # The /predict endpoint will then correctly raise a 503 Service Unavailable error.
    else:
        # Populate the global model_data dictionary
        model_data["prediction_matrix"] = prediction_artifact.get("prediction_matrix")
        model_data["user_map"] = prediction_artifact.get("user_map")
        model_data["item_map"] = prediction_artifact.get("item_map")
        print("✅ Model loaded successfully. The API is ready to serve requests.")
    print("="*50)
    
    yield # The application runs after this point
    
    # --- Shutdown Logic ---
    print("\n" + "="*50)
    print("---  shutting down API Server... ---")
    model_data.clear() # Clear the model data from memory
    print("✅ Model data cleared. Server shutdown complete.")


# --- 2. Initialize the FastAPI Application with the Lifespan Manager ---
app = FastAPI(
    title="Recommendation System API",
    description="An API to serve real-time conference recommendations.",
    version="1.0.0",
    lifespan=lifespan # Connect the lifespan manager to the app
)

# --- 3. Define the Pydantic Model for the POST Request Body ---
class PredictionRequest(BaseModel):
    user_id: str
    conference_ids: List[str]

# --- 4. Define the API Endpoint using POST ---
@app.post("/predict", response_model=Dict[str, float])
def get_predictions(request: PredictionRequest):
    """
    Provides personalized predicted ratings for a given user and a list of conferences.
    This endpoint accepts a POST request with a JSON body.
    """
    # --- Input Validation ---
    if "prediction_matrix" not in model_data or model_data["prediction_matrix"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. The service is currently unavailable.")
        
    if request.user_id not in model_data["user_map"]:
        raise HTTPException(status_code=404, detail=f"User '{request.user_id}' not found.")

    # --- Prediction Logic (Fast, In-Memory Lookups) ---
    user_idx = model_data["user_map"][request.user_id]
    predictions = {}
    
    for conf_id in request.conference_ids:
        item_idx = model_data["item_map"].get(conf_id)
        
        if item_idx is not None:
            # Direct array lookup - this is extremely fast
            score = model_data["prediction_matrix"][user_idx, item_idx]
            predictions[conf_id] = round(score, 4)
        else:
            # If a conference from the search isn't in our model, assign a neutral default score
            predictions[conf_id] = 2.5
            
    return predictions

# --- 5. Health Check Endpoint (Good Practice) ---
@app.get("/health")
def health_check():
    """A simple endpoint to check if the server is running and the model is loaded."""
    if "prediction_matrix" in model_data and model_data["prediction_matrix"] is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "degraded", "model_loaded": False}

# To run this server:
# 1. Open your terminal in the project's root directory.
# 2. Run the command: uvicorn server:app --reload