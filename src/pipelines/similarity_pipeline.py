"""
================================================================================
Pipeline Step 3: User Similarity Calculation
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from typing import Dict, Any

# Import config tĩnh để lấy đường dẫn và các hằng số không thay đổi
from config import config as static_config
from src.utils.file_handlers import load_pickle, save_pickle
from src.utils.behavioral_data_loader import fetch_behavioral_data
from src.utils.live_logger import LiveLogger

# --- Các hàm tính toán độ tương đồng ---

def _calculate_pearson_sim(u1_ratings: np.ndarray, u2_ratings: np.ndarray) -> float:
    """Tính toán độ tương đồng Pearson."""
    common_mask = (u1_ratings != 0) & (u2_ratings != 0)
    if np.sum(common_mask) < 2:
        return 0.0
    u1_common, u2_common = u1_ratings[common_mask], u2_ratings[common_mask]
    if np.std(u1_common) == 0 or np.std(u2_common) == 0:
        return 0.0
    corr, _ = pearsonr(u1_common, u2_common)
    return (np.nan_to_num(corr) + 1) / 2

# THAY ĐỔI: Hàm này cần MUTIFACTOR_ALPHA từ config
def _calculate_mutifactor_sim(u1_ratings: np.ndarray, u2_ratings: np.ndarray, u1_ts: np.ndarray, u2_ts: np.ndarray, mutifactor_alpha: float) -> float:
    """Tính toán điểm tương đồng đa yếu tố."""
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
    # THAY ĐỔI: Sử dụng mutifactor_alpha được truyền vào
    tuv = np.sum(np.exp(-mutifactor_alpha * time_diff_days))
    term_c = 1 / cuv if cuv > 0 else 0
    term_s = 1 / suv if suv > 0 else 0
    term_d = 1 / duv if duv > 0 else 0
    term_t = 1 / tuv if tuv > 0 else 0
    denominator = 1 + term_c + term_s + term_d + term_t
    return 1 / denominator if denominator != 0 else 0

def _create_matrices(df: pd.DataFrame, user_map: dict, item_map: dict, create_timestamp_matrix: bool = False) -> tuple:
    """Tạo ma trận người dùng-mục."""
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

# THAY ĐỔI: Hàm này cần MUTIFACTOR_ALPHA từ config
def _calculate_hybrid_sims(user_idx: int, influencer_indices: list, rating_matrix: np.ndarray, user_embeddings: dict, user_id: str, influencer_ids: list, model_type: str, config_data: Dict[str, Any], timestamp_matrix: np.ndarray = None) -> np.ndarray:
    """Tính toán vector độ tương đồng lai."""
    if model_type == 'mutifactor':
        rating_sims = [_calculate_mutifactor_sim(rating_matrix[user_idx], rating_matrix[inf_idx], timestamp_matrix[user_idx], timestamp_matrix[inf_idx], config_data['MUTIFACTOR_ALPHA']) for inf_idx in influencer_indices]
    else:
        rating_sims = [_calculate_pearson_sim(rating_matrix[user_idx], rating_matrix[inf_idx]) for inf_idx in influencer_indices]

    user_emb = user_embeddings.get(user_id)
    review_sims = np.zeros(len(influencer_ids))
    if user_emb is not None:
        influencer_embs = np.array([user_embeddings.get(inf_id, np.zeros(static_config.EMBEDDING_DIM)) for inf_id in influencer_ids])
        review_sims = cosine_similarity([user_emb], influencer_embs)[0]

    return (np.array(rating_sims) + review_sims) / 2.0

# THAY ĐỔI: Hàm này cần BEHAVIORAL_WEIGHTS từ config
def _create_behavioral_rating_matrix(behavioral_df: pd.DataFrame, user_map: dict, item_map: dict, config_data: Dict[str, Any]) -> np.ndarray:
    """Tạo ma trận đánh giá ngầm từ dữ liệu hành vi."""
    LiveLogger.log("   - Converting behavioral data to implicit rating matrix...")
    num_users, num_items = len(user_map), len(item_map)
    behavioral_matrix = np.zeros((num_users, num_items))

    # THAY ĐỔI: Sử dụng BEHAVIORAL_WEIGHTS từ config_data
    behavioral_df['weighted_score'] = behavioral_df['behavior_type'].map(config_data['BEHAVIORAL_WEIGHTS']) * behavioral_df['count']

    implicit_ratings = behavioral_df.groupby(['user_id', 'conference_key'])['weighted_score'].sum().reset_index()
    for _, row in implicit_ratings.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            behavioral_matrix[uidx, iidx] = row['weighted_score']

    min_val, max_val = behavioral_matrix.min(), behavioral_matrix.max()
    if max_val > min_val:
        behavioral_matrix = 5 * (behavioral_matrix - min_val) / (max_val - min_val)
    return behavioral_matrix

# --- Hàm Pipeline chính ---
# THAY ĐỔI: Chữ ký hàm nhận thêm config_data
def run_similarity_pipeline(df: pd.DataFrame, model_type: str, config_data: Dict[str, Any]) -> bool:
    """Thực thi pipeline tính toán độ tương đồng người dùng."""
    try:
        # --- 1. Tải các Tạo phẩm Tiền đề ---
        LiveLogger.log("📂 Loading prerequisite artifacts (clustering, embeddings)...")
        clustering_artifacts = load_pickle(static_config.CLUSTERING_ARTIFACT_PATH)
        user_embeddings = load_pickle(static_config.EMBEDDINGS_ARTIFACT_PATH)

        if not all([clustering_artifacts, user_embeddings is not None, not df.empty]):
            LiveLogger.log("❌ Error: One or more required inputs are missing or empty (DataFrame, clustering, or embeddings).")
            return False
        LiveLogger.log("   - All prerequisite artifacts loaded successfully.")
        LiveLogger.log("✅ Source data received as DataFrame.")

        clusters = clustering_artifacts['clusters']
        user_map = clustering_artifacts['user_map']
        item_map = clustering_artifacts['item_map']

        # --- 2. Chuẩn bị Dữ liệu và Xác định Người có ảnh hưởng ---
        LiveLogger.log("🛠️  Preparing data matrices and identifying influencers...")
        use_timestamps = (model_type == 'mutifactor')
        rating_matrix, timestamp_matrix = _create_matrices(df, user_map, item_map, create_timestamp_matrix=use_timestamps)

        user_activity = df['user_id'].value_counts()
        # THAY ĐỔI: Sử dụng NUM_INFLUENCERS từ config_data
        influencer_ids = user_activity.head(config_data['NUM_INFLUENCERS']).index.tolist()
        influencer_indices = [user_map[uid] for uid in influencer_ids if uid in user_map]
        LiveLogger.log(f"   - Identified {len(influencer_ids)} influencers for preference similarity.")

        # --- 3. Xử lý Dữ liệu Hành vi ---
        behavioral_rating_matrix = None
        if config_data['USE_BEHAVIORAL_DATA']:
            LiveLogger.log("📈 Processing behavioral data component...")

            # THAY ĐỔI: Truyền config_data vào hàm fetch_behavioral_data
            behavioral_df = fetch_behavioral_data(config_data)

            if not behavioral_df.empty:
                behavioral_rating_matrix = _create_behavioral_rating_matrix(behavioral_df, user_map, item_map, config_data)
            else:
                LiveLogger.log("   - No behavioral data returned from database. Skipping this component.")
        else:
            LiveLogger.log("   - Behavioral data component is disabled in config. Skipping.")


        # --- 4. Tính toán Độ tương đồng trong mỗi Cụm ---
        LiveLogger.log("🧠 Calculating final combined similarity scores within each cluster...")
        final_similarity_scores = {}
        num_clusters = len(clusters)
        LiveLogger.start_progress(description="Processing clusters", total=num_clusters)

        for i, (cluster_id, user_ids_in_cluster) in enumerate(clusters.items()):
            if len(user_ids_in_cluster) < 2:
                LiveLogger.update_progress(current=i + 1)
                continue

            # THAY ĐỔI: Truyền config_data vào hàm con
            user_influencer_sims = {
                user_id: _calculate_hybrid_sims(user_map[user_id], influencer_indices, rating_matrix, user_embeddings, user_id, influencer_ids, model_type, config_data, timestamp_matrix)
                for user_id in user_ids_in_cluster
            }

            cluster_user_list = list(user_influencer_sims.keys())
            influencer_vectors = np.array([user_influencer_sims[uid] for uid in cluster_user_list])
            preference_sim_matrix = cosine_similarity(influencer_vectors)

            # THAY ĐỔI: Sử dụng USE_BEHAVIORAL_DATA và BEHAVIORAL_SIM_WEIGHT từ config_data
            if config_data['USE_BEHAVIORAL_DATA'] and behavioral_rating_matrix is not None:
                cluster_user_indices = [user_map[uid] for uid in cluster_user_list]
                cluster_behavioral_slice = behavioral_rating_matrix[cluster_user_indices, :]

                behavioral_sim_matrix = np.identity(len(cluster_user_list))
                for u_idx in range(len(cluster_user_list)):
                    for v_idx in range(u_idx + 1, len(cluster_user_list)):
                        sim = _calculate_pearson_sim(cluster_behavioral_slice[u_idx], cluster_behavioral_slice[v_idx])
                        behavioral_sim_matrix[u_idx, v_idx] = behavioral_sim_matrix[v_idx, u_idx] = sim

                final_sim_matrix = (
                    (1 - config_data['BEHAVIORAL_SIM_WEIGHT']) * preference_sim_matrix +
                    config_data['BEHAVIORAL_SIM_WEIGHT'] * behavioral_sim_matrix
                )
            else:
                final_sim_matrix = preference_sim_matrix

            for u_idx, user1_id in enumerate(cluster_user_list):
                user_score_pairs = sorted(list(zip(cluster_user_list, final_sim_matrix[u_idx])), key=lambda x: x[1], reverse=True)
                final_similarity_scores[user1_id] = user_score_pairs

            LiveLogger.update_progress(current=i + 1)

        LiveLogger.log(f"✅ Finished processing all {num_clusters} clusters.")

        # --- 5. Lưu Tạo phẩm Cuối cùng ---
        output_path = static_config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else static_config.PEARSON_SIMILARITY_PATH
        LiveLogger.log(f"💾 Saving final combined similarity score artifact to '{output_path.name}'...")
        save_pickle(final_similarity_scores, output_path)
        LiveLogger.log(f"   - Artifact saved successfully.")

        return True

    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the similarity pipeline: {e}")
        return False

# --- Khối để thực thi độc lập ---
# THAY ĐỔI: Cập nhật khối chạy standalone để mô phỏng việc truyền config
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

    from utils.feedbacks_data_loader import fetch_and_prepare_data
    from config.config import get_pipeline_config

    print("--- Running Similarity Pipeline in Standalone Mode ---")
    try:
        print("Step 0a: Fetching data for standalone test...")
        source_df = fetch_and_prepare_data(force_fetch=False)

        print("Step 0b: Loading pipeline configuration...")
        test_config = get_pipeline_config()
        print(f"   - Loaded config with USE_BEHAVIORAL_DATA = {test_config['USE_BEHAVIORAL_DATA']}")

        if source_df is not None and not source_df.empty:
            print("\nRunning PEARSON model for testing...")
            run_similarity_pipeline(source_df, model_type='pearson', config_data=test_config)

            print("\nRunning MUTIFACTOR model for testing...")
            run_similarity_pipeline(source_df, model_type='mutifactor', config_data=test_config)

            print("\n✅ Similarity pipeline completed in standalone mode.")
        else:
            print("❌ Could not load or prepare data. Halting standalone test.")

    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")

    print("--- Standalone run finished ---")
