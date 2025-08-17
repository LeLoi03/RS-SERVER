import pandas as pd
import numpy as np
from tqdm import tqdm
import config.config as config  # Import centralized configuration
from src.utils.file_handlers import load_pickle, save_pickle # Import utilities

# --- Helper Functions ---

def _predict_rating_for_user_item(target_user_idx, target_item_idx, rating_matrix, user_map, rev_user_map, similarity_scores, k):
    """
    Predicts a rating for a single user-item pair using user-based collaborative filtering.
    Returns None if a prediction cannot be made.
    """
    target_user_id = rev_user_map.get(target_user_idx)
    if not target_user_id: return None
    
    similar_users = similarity_scores.get(target_user_id, [])
    
    numerator = 0
    denominator = 0
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

def _create_rating_matrix(df, user_map, item_map):
    """Creates a user-item rating matrix from the dataframe."""
    rating_matrix = np.zeros((len(user_map), len(item_map)))
    for _, row in df.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            rating_matrix[uidx, iidx] = row['rating']
    return rating_matrix

# --- Main Pipeline Function ---

def run_prediction_pipeline(model_type: str):
    """
    Generates and saves the full prediction matrix for all users and items.
    
    Args:
        model_type (str): The model to use, either 'pearson' or 'mutifactor'.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        print("\n" + "="*60)
        print(f"--- 🚀 Running Step 4: Prediction Pipeline ({model_type.upper()}) ---")
        print("="*60)

        # --- 1. Determine file paths based on model type ---
        similarity_path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
        output_path = config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else config.PEARSON_PREDICTIONS_PATH

        # --- 2. Load Prerequisite Artifacts and Data ---
        print("📂 Loading prerequisite artifacts (clustering, similarity scores) and source data...")
        clustering_artifacts = load_pickle(config.CLUSTERING_ARTIFACT_PATH)
        similarity_scores = load_pickle(similarity_path)
        df = pd.read_csv(config.SOURCE_DATASET_PATH)

        if not all([clustering_artifacts, similarity_scores, not df.empty]):
            print("❌ Error: One or more required inputs are missing or empty.")
            return False

        user_map = clustering_artifacts['user_map']
        item_map = clustering_artifacts['item_map']
        rev_user_map = clustering_artifacts['rev_user_map']
        
        # --- 3. Prepare Data ---
        print("🛠️  Preparing original rating matrix for prediction context...")
        # We need the original ratings to know which items to predict and to find neighbors' ratings
        rating_matrix = _create_rating_matrix(df, user_map, item_map)
        num_users, num_items = rating_matrix.shape
        print(f"   - Loaded data for {num_users} users and {num_items} conferences.")

        # --- 4. Pre-compute Fallback Values ---
        print("📊 Pre-calculating user average ratings for fallback...")
        user_means = np.true_divide(rating_matrix.sum(1), (rating_matrix != 0).sum(1))
        global_mean = np.mean(rating_matrix[rating_matrix > 0])
        user_means[np.isnan(user_means)] = global_mean

        # --- 5. Generate Full Prediction Matrix ---
        print(f"🧠 Starting prediction for all missing ratings using k={config.NUM_NEIGHBORS} neighbors...")
        # Start with a copy of the original matrix to preserve existing ratings
        full_prediction_matrix = np.copy(rating_matrix)

        for user_idx in tqdm(range(num_users), desc="Processing Users"):
            for item_idx in range(num_items):
                if rating_matrix[user_idx, item_idx] == 0: # Only predict if missing
                    predicted_rating = _predict_rating_for_user_item(
                        user_idx, item_idx, rating_matrix, user_map, 
                        rev_user_map, similarity_scores, k=config.NUM_NEIGHBORS
                    )
                    
                    # Apply prediction or fallback
                    full_prediction_matrix[user_idx, item_idx] = predicted_rating if predicted_rating is not None else user_means[user_idx]

        # Final clamp to ensure all values are within the valid range [1, 5]
        full_prediction_matrix = np.clip(full_prediction_matrix, 1, 5)

        # --- 6. Save the Final Artifact ---
        print(f"💾 Saving final prediction artifact to '{output_path}'...")
        prediction_artifact = {
            "prediction_matrix": full_prediction_matrix,
            "user_map": user_map,
            "item_map": item_map
        }
        save_pickle(prediction_artifact, output_path)
        
        print("\n--- ✅ Step 4: Prediction Pipeline Completed Successfully ---\n")
        return True

    except Exception as e:
        print(f"❌ An unexpected error occurred in the prediction pipeline: {e}")
        return False

if __name__ == '__main__':
    # This allows the script to be run standalone for testing/debugging
    print("Running PEARSON model predictions for testing...")
    run_prediction_pipeline(model_type='pearson')
    
    print("\nRunning MUTIFACTOR model predictions for testing...")
    run_prediction_pipeline(model_type='mutifactor')