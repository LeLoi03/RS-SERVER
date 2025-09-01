# api/dependencies.py

from typing import Dict, Any
import config.config as config
from src.utils.file_handlers import load_pickle

# Biến state toàn cục, nhưng được quản lý tập trung tại đây
model_data: Dict[str, Any] = {}

def load_model_data():
    """Helper function to load or reload model data into the global state."""
    print("🔄 Attempting to load prediction model...")
    # Sử dụng model mặc định là mutifactor để tải
    prediction_artifact = load_pickle(config.MUTIFACTOR_PREDICTIONS_PATH)
    if prediction_artifact:
        model_data["prediction_matrix"] = prediction_artifact.get("prediction_matrix")
        model_data["user_map"] = prediction_artifact.get("user_map")
        model_data["item_map"] = prediction_artifact.get("item_map")
        print("✅ Model loaded successfully.")
        return True
    else:
        print("⚠️ Could not load model artifact. API might be in a degraded state.")
        model_data.clear()
        return False

def get_model_data() -> Dict[str, Any]:
    """Dependency function to provide model data to endpoints."""
    return model_data