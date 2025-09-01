# inspect_combined_similarity.py

# ==============================================================================
# --- BOILERPLATE TO MAKE SCRIPT RUNNABLE FROM ANYWHERE ---
import sys
import os
import random


# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root
project_root = os.path.dirname(script_dir)
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==============================================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# Set a seed for reproducibility of the behavioral data simulation
random.seed(42)
np.random.seed(42)


# --- CONFIGURATION ---
ARTIFACTS_DIR = 'artifacts/'
DATA_DIR = 'data/'

# --- PATHS (relative to project root) ---
CLUSTERING_PATH = os.path.join(project_root, ARTIFACTS_DIR, "user_clustering.pkl")
EMBEDDINGS_PATH = os.path.join(project_root, ARTIFACTS_DIR, "user_embeddings.pkl")
DATASET_PATH = os.path.join(project_root, DATA_DIR, "conference_reviews_dataset.csv")
FINAL_SIMILARITY_PATH = os.path.join(project_root, ARTIFACTS_DIR, "similarity_mutifactor.pkl")

# --- PARAMETERS FOR INSPECTION ---
CLUSTER_ID_TO_INSPECT = 3 # A cluster with a good number of members
USER_TO_INSPECT = None    # Will be set automatically
TOP_N = 10
NUM_INFLUENCERS_FOR_INSPECTION = 50 # Should match the main pipeline config

# --- HELPER FUNCTIONS (Copied and adapted from pipeline for standalone execution) ---

def _calculate_pearson_sim_robust(u1_ratings, u2_ratings):
    """Calculates Pearson Correlation robustly, handling constant inputs."""
    common_mask = (u1_ratings != 0) & (u2_ratings != 0)
    if np.sum(common_mask) < 2: return 0.0
    
    u1_common, u2_common = u1_ratings[common_mask], u2_ratings[common_mask]
    
    if np.std(u1_common) == 0 or np.std(u2_common) == 0: return 0.0
    
    corr, _ = pearsonr(u1_common, u2_common)
    return (np.nan_to_num(corr) + 1) / 2

def _create_rating_matrix(df, user_map, item_map):
    """Creates a user-item rating matrix."""
    num_users, num_items = len(user_map), len(item_map)
    rating_matrix = np.zeros((num_users, num_items))
    for _, row in df.iterrows():
        uidx, iidx = user_map.get(row['user_id']), item_map.get(row['conference_key'])
        if uidx is not None and iidx is not None:
            rating_matrix[uidx, iidx] = row['rating']
    return rating_matrix

# data/inspect_combined_similarity.py

def _calculate_preference_sim_matrix(df, user_map, user_embeddings, users_in_cluster):
    """
    Simulates the 'influencer vector' method to calculate preference similarity.
    This provides a more accurate comparison than simple Pearson.
    """
    rating_matrix = _create_rating_matrix(df, user_map, item_map)
    
    # Identify influencers based on activity
    user_activity = df['user_id'].value_counts()
    influencer_ids = user_activity.head(NUM_INFLUENCERS_FOR_INSPECTION).index.tolist()
    influencer_indices = [user_map[uid] for uid in influencer_ids]
    
    user_influencer_sims = {}
    
    # --- FIX IS HERE ---
    # Get the embedding dimension from a sample user to handle cases where a user has no embedding
    sample_emb = next(iter(user_embeddings.values()), [])
    embedding_dim = len(sample_emb)

    for user_id in users_in_cluster:
        user_idx = user_map[user_id]
        
        # Part 1: Rating-based similarity (using robust Pearson)
        rating_sims = [_calculate_pearson_sim_robust(rating_matrix[user_idx], rating_matrix[inf_idx]) for inf_idx in influencer_indices]
        
        # Part 2: Content-based similarity
        user_emb_list = user_embeddings.get(user_id)
        review_sims = np.zeros(len(influencer_ids))

        if user_emb_list: # Check if the list is not None and not empty
            # Convert user embedding to NumPy array
            user_emb_np = np.array(user_emb_list).reshape(1, -1)
            
            # Get influencer embeddings, converting them to NumPy arrays and handling missing ones
            influencer_embs = []
            for inf_id in influencer_ids:
                inf_emb_list = user_embeddings.get(inf_id)
                if inf_emb_list:
                    influencer_embs.append(np.array(inf_emb_list))
                else:
                    influencer_embs.append(np.zeros(embedding_dim))
            
            review_sims = cosine_similarity(user_emb_np, np.array(influencer_embs))[0]
            
        # Part 3: Combine
        user_influencer_sims[user_id] = (np.array(rating_sims) + review_sims) / 2.0
        
    influencer_vectors = np.array([user_influencer_sims[uid] for uid in users_in_cluster])
    return cosine_similarity(influencer_vectors)

def _calculate_behavioral_sim_matrix(user_map, item_map, users_in_cluster):
    """Calculates the behavioral similarity matrix for a specific cluster."""
    from utils.behavioral_data_loader import fetch_behavioral_data
    from src.pipelines.similarity_pipeline import _create_behavioral_rating_matrix
    
    all_user_ids = list(user_map.keys())
    all_item_ids = list(item_map.keys())
    
    behavioral_df = fetch_behavioral_data(all_user_ids, all_item_ids)
    behavioral_rating_matrix = _create_behavioral_rating_matrix(behavioral_df, user_map, item_map)
    
    cluster_user_indices = [user_map[uid] for uid in users_in_cluster]
    cluster_behavioral_slice = behavioral_rating_matrix[cluster_user_indices, :]
    
    # Calculate Pearson similarity on the behavioral matrix slice
    num_users = cluster_behavioral_slice.shape[0]
    sim_matrix = np.identity(num_users)
    for i in range(num_users):
        for j in range(i + 1, num_users):
            sim = _calculate_pearson_sim_robust(cluster_behavioral_slice[i], cluster_behavioral_slice[j])
            sim_matrix[i, j] = sim_matrix[j, i] = sim
    return sim_matrix

# --- Main Inspection Logic ---

def main():
    global USER_TO_INSPECT, item_map
    print("="*80)
    print("--- Advanced Similarity Impact Inspector ---")
    print("This script compares preference-only, behavioral-only, and combined similarities.")
    print("="*80)

    # --- 1. Load all necessary data ---
    print("📂 Loading data and artifacts...")
    try:
        df = pd.read_csv(DATASET_PATH)
        with open(CLUSTERING_PATH, 'rb') as f: clustering_data = pickle.load(f)
        with open(EMBEDDINGS_PATH, 'rb') as f: user_embeddings = pickle.load(f)
        with open(FINAL_SIMILARITY_PATH, 'rb') as f: final_sim_scores = pickle.load(f)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not load a required file. {e}")
        return

    user_map = clustering_data['user_map']
    item_map = clustering_data['item_map']
    clusters = clustering_data['clusters']
    
    users_in_cluster = clusters.get(CLUSTER_ID_TO_INSPECT)
    if not users_in_cluster or len(users_in_cluster) < 2:
        print(f"❌ Error: Cluster ID {CLUSTER_ID_TO_INSPECT} not found or has fewer than 2 members.")
        return
        
    USER_TO_INSPECT = users_in_cluster[0]
    print(f"🎯 Inspecting user '{USER_TO_INSPECT}' within Cluster {CLUSTER_ID_TO_INSPECT} ({len(users_in_cluster)} members).")

    # --- 2. Re-calculate Preference-Only Similarity ---
    print("\n🧠 1/3: Re-calculating PREFERENCE-ONLY similarity (using influencer method)...")
    preference_sim_matrix = _calculate_preference_sim_matrix(df, user_map, user_embeddings, users_in_cluster)
    
    # --- 3. Re-calculate Behavioral-Only Similarity ---
    print("\n🧠 2/3: Re-calculating BEHAVIORAL-ONLY similarity...")
    behavioral_sim_matrix = _calculate_behavioral_sim_matrix(user_map, item_map, users_in_cluster)

    # --- 4. Extract and Format Results ---
    print("\n🧠 3/3: Formatting and comparing results...")
    target_user_cluster_idx = users_in_cluster.index(USER_TO_INSPECT)
    
    df_comp = pd.DataFrame({
        "Neighbor": [uid for uid in users_in_cluster if uid != USER_TO_INSPECT],
        "Pref_Score": [s for i, s in enumerate(preference_sim_matrix[target_user_cluster_idx]) if i != target_user_cluster_idx],
        "Behav_Score": [s for i, s in enumerate(behavioral_sim_matrix[target_user_cluster_idx]) if i != target_user_cluster_idx]
    })
    
    final_scores_for_user = dict(final_sim_scores.get(USER_TO_INSPECT, []))
    df_comp['Combined_Score'] = df_comp['Neighbor'].map(final_scores_for_user).fillna(0)
    
    # --- 5. Display Comparison Tables ---
    df_pref_top = df_comp.sort_values(by="Pref_Score", ascending=False).head(TOP_N)
    df_behav_top = df_comp.sort_values(by="Behav_Score", ascending=False).head(TOP_N)
    df_final_top = df_comp.sort_values(by="Combined_Score", ascending=False).head(TOP_N)

    print(f"\n--- Top {TOP_N} Neighbors for '{USER_TO_INSPECT}' (PREFERENCE-ONLY) ---")
    print(df_pref_top[['Neighbor', 'Pref_Score']].to_markdown(index=False))
    
    print(f"\n--- Top {TOP_N} Neighbors for '{USER_TO_INSPECT}' (BEHAVIORAL-ONLY) ---")
    print(df_behav_top[['Neighbor', 'Behav_Score']].to_markdown(index=False))
    
    print(f"\n--- Top {TOP_N} Neighbors for '{USER_TO_INSPECT}' (FINAL COMBINED from Pipeline) ---")
    print(df_final_top[['Neighbor', 'Combined_Score', 'Pref_Score', 'Behav_Score']].to_markdown(index=False))

    # --- 6. Analyze the Impact ---
    pref_neighbors = set(df_pref_top['Neighbor'])
    final_neighbors = set(df_final_top['Neighbor'])
    
    promoted = final_neighbors - pref_neighbors
    demoted = pref_neighbors - final_neighbors
    
    print("\n--- Summary of Impact ---")
    if not promoted and not demoted:
        print(f"The Top {TOP_N} neighbor lists are identical. No significant change observed.")
    else:
        print(f"Adding behavioral data changed the ranking, promoting {len(promoted)} new users into the Top {TOP_N}.")
        if promoted:
            print("\nUsers PROMOTED into Top 10 by behavioral data:")
            for user in sorted(list(promoted)):
                user_row = df_comp[df_comp['Neighbor'] == user].iloc[0]
                print(f"  - {user:<15} (Behavioral Score: {user_row['Behav_Score']:.3f}, Preference Score: {user_row['Pref_Score']:.3f})")
        if demoted:
            print("\nUsers DEMOTED out of Top 10:")
            for user in sorted(list(demoted)):
                user_row = df_comp[df_comp['Neighbor'] == user].iloc[0]
                print(f"  - {user:<15} (Behavioral Score: {user_row['Behav_Score']:.3f}, Preference Score: {user_row['Pref_Score']:.3f})")
            
    print("\n" + "="*80)

if __name__ == "__main__":
    main()