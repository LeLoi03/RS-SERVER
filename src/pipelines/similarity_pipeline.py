import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import config.config as config  # Import centralized configuration
from src.utils.file_handlers import load_pickle, save_pickle # Import utilities

# --- Similarity Calculation Functions ---

def _calculate_pearson_sim(u1_ratings, u2_ratings):
    """Calculates Pearson Correlation and normalizes to [0, 1]."""
    common_mask = (u1_ratings != 0) & (u2_ratings != 0)
    if np.sum(common_mask) < 2: return 0.0
    u1_common, u2_common = u1_ratings[common_mask], u2_ratings[common_mask]
    mean1, mean2 = u1_common.mean(), u2_common.mean()
    cov = np.sum((u1_common - mean1) * (u2_common - mean2))
    std1, std2 = np.sqrt(np.sum((u1_common - mean1)**2)), np.sqrt(np.sum((u2_common - mean2)**2))
    if std1 == 0 or std2 == 0: return 0.0
    pearson_sim = cov / (std1 * std2)
    return (np.nan_to_num(pearson_sim) + 1) / 2

def _calculate_mutifactor_sim(u1_ratings, u2_ratings, u1_ts, u2_ts):
    """Calculates the full, multi-component similarity score."""
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
    return 1 / (1 + term_c + term_s + term_d + term_t)

# --- Helper Functions ---

def _create_matrices(df, user_map, item_map, create_timestamp_matrix=False):
    """Creates user-item rating and optionally timestamp matrices."""
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

def _calculate_hybrid_sims(user_idx, influencer_indices, rating_matrix, user_embeddings, user_id, influencer_ids, model_type, timestamp_matrix=None):
    """Calculates the hybrid similarity vector for a single user against all influencers."""
    if model_type == 'mutifactor':
        rating_sims = [_calculate_mutifactor_sim(rating_matrix[user_idx], rating_matrix[inf_idx], timestamp_matrix[user_idx], timestamp_matrix[inf_idx]) for inf_idx in influencer_indices]
    else: # Default to Pearson
        rating_sims = [_calculate_pearson_sim(rating_matrix[user_idx], rating_matrix[inf_idx]) for inf_idx in influencer_indices]

    user_emb = user_embeddings.get(user_id)
    review_sims = np.zeros(len(influencer_ids))
    if user_emb is not None:
        influencer_embs = np.array([user_embeddings.get(inf_id, np.zeros(config.EMBEDDING_DIM)) for inf_id in influencer_ids])
        review_sims = cosine_similarity([user_emb], influencer_embs)[0]
        
    return (np.array(rating_sims) + review_sims) / 2.0

# --- Main Pipeline Function ---

def run_similarity_pipeline(model_type: str):
    """
    Calculates user-user similarities using the specified model type.
    
    Args:
        model_type (str): The model to use, either 'pearson' or 'mutifactor'.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        print("\n" + "="*60)
        print(f"--- 🚀 Running Step 3: Similarity Pipeline ({model_type.upper()}) ---")
        print("="*60)

        # --- 1. Load Prerequisite Artifacts and Data ---
        print("📂 Loading prerequisite artifacts (clustering, embeddings) and source data...")
        clustering_artifacts = load_pickle(config.CLUSTERING_ARTIFACT_PATH)
        user_embeddings = load_pickle(config.EMBEDDINGS_ARTIFACT_PATH)
        df = pd.read_csv(config.SOURCE_DATASET_PATH)

        if not all([clustering_artifacts, user_embeddings, not df.empty]):
            print("❌ Error: One or more required inputs are missing or empty.")
            return False

        clusters = clustering_artifacts['clusters']
        user_map = clustering_artifacts['user_map']
        item_map = clustering_artifacts['item_map']
        
        # --- 2. Prepare Data and Identify Influencers ---
        print("🛠️  Preparing data matrices and identifying influencers...")
        use_timestamps = (model_type == 'mutifactor')
        rating_matrix, timestamp_matrix = _create_matrices(df, user_map, item_map, create_timestamp_matrix=use_timestamps)
        
        user_activity = df['user_id'].value_counts()
        influencer_ids = user_activity.head(config.NUM_INFLUENCERS).index.tolist()
        influencer_indices = [user_map[uid] for uid in influencer_ids]
        print(f"   - Identified {len(influencer_ids)} influencers.")

        # --- 3. Calculate Similarities within each Cluster ---
        print("🧠 Calculating final similarity scores within each cluster...")
        final_similarity_scores = {}

        for cluster_id, user_ids_in_cluster in tqdm(clusters.items(), desc="Processing Clusters"):
            if len(user_ids_in_cluster) < 2:
                continue

            user_influencer_sims = {
                user_id: _calculate_hybrid_sims(user_map[user_id], influencer_indices, rating_matrix, user_embeddings, user_id, influencer_ids, model_type, timestamp_matrix)
                for user_id in user_ids_in_cluster
            }

            cluster_user_list = list(user_influencer_sims.keys())
            influencer_vectors = np.array([user_influencer_sims[uid] for uid in cluster_user_list])
            final_sim_matrix = cosine_similarity(influencer_vectors)
            
            for i, user1_id in enumerate(cluster_user_list):
                user_score_pairs = sorted(list(zip(cluster_user_list, final_sim_matrix[i])), key=lambda x: x[1], reverse=True)
                final_similarity_scores[user1_id] = user_score_pairs

        # --- 4. Save the Final Artifact ---
        output_path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
        print(f"💾 Saving similarity scores artifact to '{output_path}'...")
        save_pickle(final_similarity_scores, output_path)
        
        print("\n--- ✅ Step 3: Similarity Pipeline Completed Successfully ---\n")
        return True

    except Exception as e:
        print(f"❌ An unexpected error occurred in the similarity pipeline: {e}")
        return False

if __name__ == '__main__':
    # This allows the script to be run standalone for testing/debugging
    # It will run both versions for comparison
    print("Running PEARSON model for testing...")
    run_similarity_pipeline(model_type='pearson')
    
    print("\nRunning MUTIFACTOR model for testing...")
    run_similarity_pipeline(model_type='mutifactor')