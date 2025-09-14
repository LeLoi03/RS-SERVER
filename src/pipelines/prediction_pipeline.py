"""
================================================================================
Pipeline Step 4: Pre-computing Predictions
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Import config tĩnh để lấy đường dẫn
from config import config as static_config
from src.utils.file_handlers import load_pickle, save_pickle
from src.utils.live_logger import LiveLogger

# --- Helper Functions ---
def _predict_rating_for_user_item(target_user_idx: int, target_item_idx: int, rating_matrix: np.ndarray, user_map: dict, rev_user_map: dict, similarity_scores: dict, k: int) -> float | None:
    """Dự đoán rating cho một cặp (người dùng, mục) cụ thể."""
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
    """Tạo ma trận rating từ DataFrame."""
    rating_matrix = np.zeros((len(user_map), len(item_map)))
    for _, row in df.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            rating_matrix[uidx, iidx] = row['rating']
    return rating_matrix

# --- Main Pipeline Function ---
# THAY ĐỔI: Hàm nhận thêm tham số config_data
def run_prediction_pipeline(df: pd.DataFrame, model_type: str, config_data: Dict[str, Any]) -> bool:
    """
    Tạo và lưu ma trận dự đoán đầy đủ cho tất cả người dùng và mục.

    Args:
        df (pd.DataFrame): DataFrame nguồn chứa dữ liệu phản hồi của người dùng.
        model_type (str): Mô hình sử dụng, 'pearson' hoặc 'mutifactor'.
        config_data (Dict[str, Any]): Dictionary chứa các tham số cấu hình (ví dụ: NUM_NEIGHBORS).

    Returns:
        bool: True nếu thành công, False nếu ngược lại.
    """
    try:
        # --- 1. Xác định đường dẫn file dựa trên loại mô hình ---
        similarity_path = static_config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else static_config.PEARSON_SIMILARITY_PATH
        output_path = static_config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else static_config.PEARSON_PREDICTIONS_PATH

        # --- 2. Tải các Artifact cần thiết ---
        LiveLogger.log("📂 Loading prerequisite artifacts (clustering, similarity scores)...")
        clustering_artifacts = load_pickle(static_config.CLUSTERING_ARTIFACT_PATH)
        similarity_scores = load_pickle(similarity_path)

        if not all([clustering_artifacts, similarity_scores, not df.empty]):
            LiveLogger.log("❌ Error: One or more required inputs are missing or empty (DataFrame, clustering, or similarity scores).")
            return False
        LiveLogger.log("   - All prerequisite artifacts loaded successfully.")
        LiveLogger.log("✅ Source data received as DataFrame.")

        user_map = clustering_artifacts['user_map']
        item_map = clustering_artifacts['item_map']
        rev_user_map = clustering_artifacts['rev_user_map']

        # --- 3. Chuẩn bị Ma trận Rating Gốc ---
        LiveLogger.log("🛠️  Preparing original rating matrix for prediction context...")
        rating_matrix = _create_rating_matrix(df, user_map, item_map)
        num_users, num_items = rating_matrix.shape
        LiveLogger.log(f"   - Matrix with shape {rating_matrix.shape} created.")

        # --- 4. Tính toán trước các giá trị Fallback ---
        LiveLogger.log("📊 Pre-calculating user average ratings for fallback...")
        user_means = np.true_divide(rating_matrix.sum(1), (rating_matrix != 0).sum(1))
        global_mean = np.mean(rating_matrix[rating_matrix > 0]) if np.any(rating_matrix > 0) else 2.5
        user_means[np.isnan(user_means)] = global_mean
        LiveLogger.log("   - Fallback values calculated.")

        # --- 5. Tạo Ma trận Dự đoán Đầy đủ ---
        # THAY ĐỔI: Sử dụng NUM_NEIGHBORS từ config_data
        num_neighbors = config_data['NUM_NEIGHBORS']
        LiveLogger.log(f"🧠 Starting prediction for all {num_users} users (k={num_neighbors})...")
        full_prediction_matrix = np.copy(rating_matrix)

        LiveLogger.start_progress(description="Predicting for Users", total=num_users)

        for user_idx in range(num_users):
            for item_idx in range(num_items):
                if rating_matrix[user_idx, item_idx] == 0:
                    predicted_rating = _predict_rating_for_user_item(
                        user_idx, item_idx, rating_matrix, user_map,
                        rev_user_map, similarity_scores, k=num_neighbors
                    )
                    full_prediction_matrix[user_idx, item_idx] = predicted_rating if predicted_rating is not None else user_means[user_idx]

            LiveLogger.update_progress(current=user_idx + 1)

        full_prediction_matrix = np.clip(full_prediction_matrix, 1, 5)
        LiveLogger.log(f"✅ Finished predicting ratings for all {num_users} users.")

        # --- 6. Lưu Artifact Cuối cùng ---
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
    sys.modules['src/utils/live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    from src.utils.live_logger import LiveLogger

    from utils.feedbacks_data_loader import fetch_and_prepare_data
    from config.config import get_pipeline_config

    print("--- Running Prediction Pipeline in Standalone Mode ---")
    try:
        print("Step 0a: Fetching data for standalone test...")
        source_df = fetch_and_prepare_data(force_fetch=False)

        print("Step 0b: Loading pipeline configuration...")
        test_config = get_pipeline_config()
        print(f"   - Loaded config with NUM_NEIGHBORS = {test_config['NUM_NEIGHBORS']}")

        if source_df is not None and not source_df.empty:
            print("\nRunning PEARSON model predictions for testing...")
            run_prediction_pipeline(source_df, model_type='pearson', config_data=test_config)

            print("\n\nRunning MUTIFACTOR model predictions for testing...")
            run_prediction_pipeline(source_df, model_type='mutifactor', config_data=test_config)

            print("\n\n✅ Prediction pipeline completed in standalone mode.")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")

    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")

    print("--- Standalone run finished ---")
