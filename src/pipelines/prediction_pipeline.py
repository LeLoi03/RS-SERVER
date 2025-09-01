"""
================================================================================
Pipeline Step 4: Pre-computing Predictions
================================================================================
"""

import pandas as pd
import numpy as np
from config import config
from src.utils.file_handlers import load_pickle, save_pickle
from src.utils.live_logger import LiveLogger

# --- Helper Functions ---
# (Các hàm helper _predict_rating_for_user_item và _create_rating_matrix giữ nguyên)
def _predict_rating_for_user_item(target_user_idx: int, target_item_idx: int, rating_matrix: np.ndarray, user_map: dict, rev_user_map: dict, similarity_scores: dict, k: int) -> float | None:
    target_user_id = rev_user_map.get(target_user_idx)
    if not target_user_id: return None
    
    similar_users = similarity_scores.get(target_user_id, [])
    
    numerator = 0.0
    denominator = 0.0
    neighbors_found = 0
    
    for neighbor_id, sim_score in similar_users:
        if neighbors_found >= k or sim_score <= 0:
            break
        if neighbor_id == target_user_id:
            continue
            
        neighbor_idx = user_map.get(neighbor_id)
        if neighbor_idx is None:
            continue
            
        neighbor_rating = rating_matrix[neighbor_idx, target_item_idx]
        if neighbor_rating > 0:
            numerator += sim_score * neighbor_rating
            denominator += sim_score
            neighbors_found += 1
            
    if denominator == 0:
        return None
        
    predicted_rating = numerator / denominator
    return np.clip(predicted_rating, 1, 5)

def _create_rating_matrix(df: pd.DataFrame, user_map: dict, item_map: dict) -> np.ndarray:
    rating_matrix = np.zeros((len(user_map), len(item_map)))
    for _, row in df.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            rating_matrix[uidx, iidx] = row['rating']
    return rating_matrix

# --- Main Pipeline Function ---
# THAY ĐỔI: Hàm bây giờ nhận DataFrame và model_type
def run_prediction_pipeline(df: pd.DataFrame, model_type: str) -> bool:
    """
    Generates and saves the full prediction matrix for all users and items.
    
    Args:
        df (pd.DataFrame): The source DataFrame containing user feedback data.
        model_type (str): The model to use, either 'pearson' or 'mutifactor'.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # --- 1. Determine file paths based on model type ---
        similarity_path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
        output_path = config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else config.PEARSON_PREDICTIONS_PATH

        # --- 2. Load Prerequisite Artifacts ---
        # THAY ĐỔI: Không đọc source data từ file nữa
        LiveLogger.log("📂 Loading prerequisite artifacts (clustering, similarity scores)...")
        clustering_artifacts = load_pickle(config.CLUSTERING_ARTIFACT_PATH)
        similarity_scores = load_pickle(similarity_path)

        # THAY ĐỔI: Cập nhật kiểm tra đầu vào
        if not all([clustering_artifacts, similarity_scores, not df.empty]):
            LiveLogger.log("❌ Error: One or more required inputs are missing or empty (DataFrame, clustering, or similarity scores).")
            return False
        LiveLogger.log("   - All prerequisite artifacts loaded successfully.")
        LiveLogger.log("✅ Source data received as DataFrame.")

        user_map = clustering_artifacts['user_map']
        item_map = clustering_artifacts['item_map']
        rev_user_map = clustering_artifacts['rev_user_map']
        
        # --- 3. Prepare Original Rating Matrix ---
        # Logic này giữ nguyên, vì nó hoạt động trên DataFrame đầu vào
        LiveLogger.log("🛠️  Preparing original rating matrix for prediction context...")
        rating_matrix = _create_rating_matrix(df, user_map, item_map)
        num_users, num_items = rating_matrix.shape
        LiveLogger.log(f"   - Matrix with shape {rating_matrix.shape} created.")

        # --- 4. Pre-compute Fallback Values for Cold Starts ---
        # Logic này giữ nguyên
        LiveLogger.log("📊 Pre-calculating user average ratings for fallback...")
        user_means = np.true_divide(rating_matrix.sum(1), (rating_matrix != 0).sum(1))
        # Xử lý trường hợp không có rating nào trong toàn bộ ma trận
        global_mean = np.mean(rating_matrix[rating_matrix > 0]) if np.any(rating_matrix > 0) else 2.5
        user_means[np.isnan(user_means)] = global_mean
        LiveLogger.log("   - Fallback values calculated.")

         # --- 5. Generate Full Prediction Matrix ---
         # Logic này giữ nguyên
        LiveLogger.log(f"🧠 Starting prediction for all {num_users} users (k={config.NUM_NEIGHBORS})...")
        full_prediction_matrix = np.copy(rating_matrix)

        LiveLogger.start_progress(description="Predicting for Users", total=num_users)

        for user_idx in range(num_users):
            for item_idx in range(num_items):
                if rating_matrix[user_idx, item_idx] == 0:
                    predicted_rating = _predict_rating_for_user_item(
                        user_idx, item_idx, rating_matrix, user_map, 
                        rev_user_map, similarity_scores, k=config.NUM_NEIGHBORS
                    )
                    full_prediction_matrix[user_idx, item_idx] = predicted_rating if predicted_rating is not None else user_means[user_idx]
            
            LiveLogger.update_progress(current=user_idx + 1)

        full_prediction_matrix = np.clip(full_prediction_matrix, 1, 5)
        LiveLogger.log(f"✅ Finished predicting ratings for all {num_users} users.")

        # --- 6. Save the Final Artifact ---
        # Logic này giữ nguyên
        LiveLogger.log(f"💾 Saving final prediction artifact to '{output_path.name}'...")
        prediction_artifact = {
            "prediction_matrix": full_prediction_matrix,
            "user_map": user_map,
            "item_map": item_map
        }
        save_pickle(prediction_artifact, output_path)
        LiveLogger.log(f"   - Artifact saved successfully.")
        
        return True

    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the prediction pipeline: {e}")
        return False

# THAY ĐỔI: Cập nhật khối chạy standalone
if __name__ == '__main__':
    class ConsoleLogger:
        @staticmethod
        def log(message): print(message)
        @staticmethod
        def start_progress(description, total): print(f"--- Starting Progress: {description} (Total: {total}) ---")
        @staticmethod
        def update_progress(current):
            print(f"--- Progress: {current} ---", end='\r')

    import sys
    sys.modules['src.utils.live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    from src.utils.live_logger import LiveLogger
    
    # Import data loader để chạy thử nghiệm
    from utils.feedbacks_data_loader import fetch_and_prepare_data

    print("--- Running Prediction Pipeline in Standalone Mode ---")
    try:
        print("Step 0: Fetching data for standalone test...")
        source_df = fetch_and_prepare_data(force_fetch=False)
        
        if source_df is not None and not source_df.empty:
            # Giả định rằng các artifact trước đó (clustering, similarity) đã tồn tại
            # để có thể chạy bước này độc lập.
            print("\nRunning PEARSON model predictions for testing...")
            run_prediction_pipeline(source_df, model_type='pearson')
            
            print("\n\nRunning MUTIFACTOR model predictions for testing...")
            run_prediction_pipeline(source_df, model_type='mutifactor')
            
            print("\n\n✅ Prediction pipeline completed in standalone mode.")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")
            
    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")
        
    print("--- Standalone run finished ---")