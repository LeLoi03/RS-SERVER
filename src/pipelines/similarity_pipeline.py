"""
================================================================================
Pipeline Step 3: User Similarity Calculation
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

from config import config
from src.utils.file_handlers import load_pickle, save_pickle
# THAY ĐỔI: Import data loader mới và xóa cái cũ
# from src.utils.behavioral_data_simulator import fetch_behavioral_data # <-- XÓA DÒNG NÀY
from src.utils.behavioral_data_loader import fetch_behavioral_data # <-- THÊM DÒNG NÀY
from src.utils.live_logger import LiveLogger

# --- Similarity Calculation Functions ---
# (Toàn bộ các hàm _calculate_... và _create_... giữ nguyên, không cần thay đổi)
def _calculate_pearson_sim(u1_ratings: np.ndarray, u2_ratings: np.ndarray) -> float:
    common_mask = (u1_ratings != 0) & (u2_ratings != 0)
    if np.sum(common_mask) < 2: return 0.0
    u1_common, u2_common = u1_ratings[common_mask], u2_ratings[common_mask]
    if np.std(u1_common) == 0 or np.std(u2_common) == 0: return 0.0
    corr, _ = pearsonr(u1_common, u2_common)
    return (np.nan_to_num(corr) + 1) / 2

def _calculate_mutifactor_sim(u1_ratings: np.ndarray, u2_ratings: np.ndarray, u1_ts: np.ndarray, u2_ts: np.ndarray) -> float:
    rated1_mask, rated2_mask = u1_ratings != 0, u2_ratings != 0
    common_mask = rated1_mask & rated2_mask
    cuv = np.sum(common_mask)
    suv = np.sum(rated1_mask & ~rated2_mask) + np.sum(~rated1_mask & rated2_mask)
    u1_common_ratings, u2_common_ratings = u1_ratings[common_mask], u2_ratings[common_mask]
    agree_positive = (u1_common_ratings > 3) & (u2_common_ratings > 3)
    agree_negative = (u1_common_ratings < 3) & (u2_common_ratings < 3)
    duv = np.sum(agree_positive) + np.sum(agree_negative)
    u1_common_ts, u2_common_ts = u1_ts[common_mask], u2_ts[common_mask]
    time_diff_days = np.abs(u1_common_ts - u2_common_ts) / (60 * 60 * 24)
    tuv = np.sum(np.exp(-config.MUTIFACTOR_ALPHA * time_diff_days))
    term_c = 1 / cuv if cuv > 0 else 0
    term_s = 1 / suv if suv > 0 else 0
    term_d = 1 / duv if duv > 0 else 0
    term_t = 1 / tuv if tuv > 0 else 0
    denominator = 1 + term_c + term_s + term_d + term_t
    return 1 / denominator if denominator != 0 else 0

def _create_matrices(df: pd.DataFrame, user_map: dict, item_map: dict, create_timestamp_matrix: bool = False) -> tuple:
    num_users, num_items = len(user_map), len(item_map)
    rating_matrix = np.zeros((num_users, num_items))
    timestamp_matrix = np.zeros((num_users, num_items)) if create_timestamp_matrix else None
    for _, row in df.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            rating_matrix[uidx, iidx] = row['rating']
            if create_timestamp_matrix:
                timestamp_matrix[uidx, iidx] = row['timestamp']
    return rating_matrix, timestamp_matrix

def _calculate_hybrid_sims(user_idx: int, influencer_indices: list, rating_matrix: np.ndarray, user_embeddings: dict, user_id: str, influencer_ids: list, model_type: str, timestamp_matrix: np.ndarray = None) -> np.ndarray:
    if model_type == 'mutifactor':
        rating_sims = [_calculate_mutifactor_sim(rating_matrix[user_idx], rating_matrix[inf_idx], timestamp_matrix[user_idx], timestamp_matrix[inf_idx]) for inf_idx in influencer_indices]
    else:
        rating_sims = [_calculate_pearson_sim(rating_matrix[user_idx], rating_matrix[inf_idx]) for inf_idx in influencer_indices]
    user_emb = user_embeddings.get(user_id)
    review_sims = np.zeros(len(influencer_ids))
    if user_emb is not None:
        influencer_embs = np.array([user_embeddings.get(inf_id, np.zeros(config.EMBEDDING_DIM)) for inf_id in influencer_ids])
        review_sims = cosine_similarity([user_emb], influencer_embs)[0]
    return (np.array(rating_sims) + review_sims) / 2.0

def _create_behavioral_rating_matrix(behavioral_df: pd.DataFrame, user_map: dict, item_map: dict) -> np.ndarray:
    LiveLogger.log("   - Converting behavioral data into an implicit rating matrix...")
    num_users, num_items = len(user_map), len(item_map)
    behavioral_matrix = np.zeros((num_users, num_items))
    behavioral_df['weighted_score'] = behavioral_df['behavior_type'].map(config.BEHAVIORAL_WEIGHTS) * behavioral_df['count']
    implicit_ratings = behavioral_df.groupby(['user_id', 'conference_key'])['weighted_score'].sum().reset_index()
    for _, row in implicit_ratings.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            behavioral_matrix[uidx, iidx] = row['weighted_score']
    min_val, max_val = behavioral_matrix.min(), behavioral_matrix.max()
    if max_val > min_val:
        behavioral_matrix = 5 * (behavioral_matrix - min_val) / (max_val - min_val)
    return behavioral_matrix

# --- Main Pipeline Function ---
# THAY ĐỔI: Hàm bây giờ nhận DataFrame và model_type
def run_similarity_pipeline(df: pd.DataFrame, model_type: str) -> bool:
    """
    Calculates user-user similarities using the specified model type.
    Receives a DataFrame and loads other prerequisite artifacts.
    """
    try:
        # --- 1. Load Prerequisite Artifacts ---
        # THAY ĐỔI: Không đọc source data từ file nữa
        LiveLogger.log("📂 Loading prerequisite artifacts (clustering, embeddings)...")
        clustering_artifacts = load_pickle(config.CLUSTERING_ARTIFACT_PATH)
        user_embeddings = load_pickle(config.EMBEDDINGS_ARTIFACT_PATH)
        
        # THAY ĐỔI: Cập nhật kiểm tra đầu vào
        if not all([clustering_artifacts, user_embeddings is not None, not df.empty]):
            LiveLogger.log("❌ Error: One or more required inputs are missing or empty (DataFrame, clustering, or embeddings).")
            return False
        LiveLogger.log("   - All prerequisite artifacts loaded successfully.")
        LiveLogger.log("✅ Source data received as DataFrame.")

        clusters = clustering_artifacts['clusters']
        user_map = clustering_artifacts['user_map']
        item_map = clustering_artifacts['item_map']
        
        # --- 2. Prepare Data and Identify Influencers ---
        # Logic này giữ nguyên, vì nó hoạt động trên DataFrame đầu vào
        LiveLogger.log("🛠️  Preparing data matrices and identifying influencers...")
        use_timestamps = (model_type == 'mutifactor')
        rating_matrix, timestamp_matrix = _create_matrices(df, user_map, item_map, create_timestamp_matrix=use_timestamps)
        
        user_activity = df['user_id'].value_counts()
        influencer_ids = user_activity.head(config.NUM_INFLUENCERS).index.tolist()
        influencer_indices = [user_map[uid] for uid in influencer_ids if uid in user_map] # Thêm kiểm tra an toàn
        LiveLogger.log(f"   - Identified {len(influencer_ids)} influencers for preference similarity.")

         # --- 3. Process Behavioral Data (if enabled) ---
        behavioral_rating_matrix = None
        if config.USE_BEHAVIORAL_DATA:
            LiveLogger.log("📈 Processing behavioral data component...")
            
            # THAY ĐỔI: Lời gọi hàm không cần tham số nữa
            behavioral_df = fetch_behavioral_data() 
            
            if not behavioral_df.empty:
                # user_map và item_map được lấy từ clustering_artifacts
                behavioral_rating_matrix = _create_behavioral_rating_matrix(behavioral_df, user_map, item_map)
            else:
                LiveLogger.log("   - No behavioral data returned from the database. Skipping this component.")
        else:
            LiveLogger.log("   - Behavioral data component is disabled in config. Skipping.")

         # --- 4. Calculate Similarities within each Cluster ---
         # Logic này giữ nguyên
        LiveLogger.log("🧠 Calculating final combined similarity scores within each cluster...")
        final_similarity_scores = {}
        
        num_clusters = len(clusters)
        LiveLogger.start_progress(description="Processing Clusters", total=num_clusters)
        
        for i, (cluster_id, user_ids_in_cluster) in enumerate(clusters.items()):
            if len(user_ids_in_cluster) < 2:
                LiveLogger.update_progress(current=i + 1)
                continue

            user_influencer_sims = {
                user_id: _calculate_hybrid_sims(user_map[user_id], influencer_indices, rating_matrix, user_embeddings, user_id, influencer_ids, model_type, timestamp_matrix)
                for user_id in user_ids_in_cluster
            }
            cluster_user_list = list(user_influencer_sims.keys())
            influencer_vectors = np.array([user_influencer_sims[uid] for uid in cluster_user_list])
            preference_sim_matrix = cosine_similarity(influencer_vectors)
            
            if config.USE_BEHAVIORAL_DATA and behavioral_rating_matrix is not None:
                cluster_user_indices = [user_map[uid] for uid in cluster_user_list]
                cluster_behavioral_slice = behavioral_rating_matrix[cluster_user_indices, :]
                
                behavioral_sim_matrix = np.identity(len(cluster_user_list))
                for u_idx in range(len(cluster_user_list)):
                    for v_idx in range(u_idx + 1, len(cluster_user_list)):
                        sim = _calculate_pearson_sim(cluster_behavioral_slice[u_idx], cluster_behavioral_slice[v_idx])
                        behavioral_sim_matrix[u_idx, v_idx] = behavioral_sim_matrix[v_idx, u_idx] = sim
                
                final_sim_matrix = (
                    (1 - config.BEHAVIORAL_SIM_WEIGHT) * preference_sim_matrix +
                    config.BEHAVIORAL_SIM_WEIGHT * behavioral_sim_matrix
                )
            else:
                final_sim_matrix = preference_sim_matrix
            
            for u_idx, user1_id in enumerate(cluster_user_list):
                user_score_pairs = sorted(list(zip(cluster_user_list, final_sim_matrix[u_idx])), key=lambda x: x[1], reverse=True)
                final_similarity_scores[user1_id] = user_score_pairs
            
            LiveLogger.update_progress(current=i + 1)

        LiveLogger.log(f"✅ Finished processing all {num_clusters} clusters.")

        # --- 5. Save the Final Artifact ---
        # Logic này giữ nguyên
        output_path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
        LiveLogger.log(f"💾 Saving final combined similarity scores artifact to '{output_path.name}'...")
        save_pickle(final_similarity_scores, output_path)
        LiveLogger.log(f"   - Artifact saved successfully.")
        
        return True

    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the similarity pipeline: {e}")
        return False
    
# THAY ĐỔI: Cập nhật khối chạy standalone
if __name__ == '__main__':
    class ConsoleLogger:
        @staticmethod
        def log(message): print(message)
        @staticmethod
        def start_progress(description, total): print(f"--- Starting Progress: {description} (Total: {total}) ---")
        @staticmethod
        def update_progress(current): print(f"--- Progress: {current} ---")
    
    import sys
    sys.modules['src.utils.live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    from src.utils.live_logger import LiveLogger
    
    # Import data loader để chạy thử nghiệm
    from utils.feedbacks_data_loader import fetch_and_prepare_data

    print("--- Running Similarity Pipeline in Standalone Mode ---")
    try:
        print("Step 0: Fetching data for standalone test...")
        source_df = fetch_and_prepare_data(force_fetch=False)
        
        if source_df is not None and not source_df.empty:
            print("\nRunning PEARSON model for testing...")
            run_similarity_pipeline(source_df, model_type='pearson')
            
            print("\nRunning MUTIFACTOR model for testing...")
            run_similarity_pipeline(source_df, model_type='mutifactor')
            
            print("\n✅ Similarity pipeline completed in standalone mode.")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")
            
    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")
        
    print("--- Standalone run finished ---")