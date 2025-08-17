import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from config import config  # Import centralized configuration
from src.utils.file_handlers import save_pickle # Import the utility for saving files

def run_clustering_pipeline():
    """
    Performs the user clustering step of the pipeline.
    This function reads the source data, groups users into clusters based on their
    rating behavior using K-Means with ICR initialization, and saves the results.
    
    Returns:
        bool: True if the pipeline step completes successfully, False otherwise.
    """
    try:
        print("\n" + "="*60)
        print("--- 🚀 Running Step 1: User Clustering Pipeline ---")
        print("="*60)

        # --- 1. Load Source Data ---
        print(f"📂 Loading source data from '{config.SOURCE_DATASET_PATH}'...")
        df = pd.read_csv(config.SOURCE_DATASET_PATH)
        if df.empty:
            print("❌ Error: Source data file is empty.")
            return False
        print(f"   - Loaded {len(df)} reviews.")

        # --- 2. Create User-Item Matrix and Mappings ---
        print("🛠️  Creating user-item rating matrix and mappings...")
        users = df['user_id'].unique()
        items = df['conference_key'].unique()
        
        user_map = {uid: i for i, uid in enumerate(users)}
        item_map = {iid: i for i, iid in enumerate(items)}
        rev_user_map = {i: uid for uid, i in user_map.items()}
        rev_item_map = {i: iid for iid, i in item_map.items()}

        rating_matrix = np.zeros((len(users), len(items)))
        for _, row in df.iterrows():
            user_idx = user_map.get(row['user_id'])
            item_idx = item_map.get(row['conference_key'])
            if user_idx is not None and item_idx is not None:
                rating_matrix[user_idx, item_idx] = row['rating']
        print(f"   - Matrix created with shape: {rating_matrix.shape}")

        # --- 3. Apply K-Means with ICR Initialization ---
        print("🧠 Applying K-Means clustering with ICR initialization...")
        
        # a) Select ICR centroids
        num_ratings_per_user = np.sum(rating_matrix != 0, axis=1)
        centroid_indices = np.argsort(num_ratings_per_user)[-config.NUM_CLUSTERS:]
        initial_centroids = rating_matrix[centroid_indices]
        print(f"   - Selected {len(initial_centroids)} initial centroids based on user activity.")
        
        # b) Fill missing values for clustering algorithm
        user_means = np.true_divide(rating_matrix.sum(1), (rating_matrix != 0).sum(1))
        user_means[np.isnan(user_means)] = 0 # Handle users with no ratings
        filled_matrix = np.where(rating_matrix == 0, user_means[:, np.newaxis], rating_matrix)
        
        # c) Run K-Means
        kmeans = KMeans(
            n_clusters=config.NUM_CLUSTERS,
            init=initial_centroids,
            n_init=1, # Only one initialization since we provide the centroids
            random_state=42
        )
        labels = kmeans.fit_predict(filled_matrix)
        print(f"   - Successfully assigned {len(users)} users to {config.NUM_CLUSTERS} clusters.")

        # --- 4. Format and Save Artifacts ---
        print(f"💾 Formatting and saving clustering artifacts to '{config.CLUSTERING_ARTIFACT_PATH}'...")
        
        clusters = {i: [] for i in range(config.NUM_CLUSTERS)}
        for user_idx, cluster_label in enumerate(labels):
            original_user_id = rev_user_map[user_idx]
            clusters[cluster_label].append(original_user_id)

        # The artifact contains everything needed by downstream steps
        clustering_artifacts = {
            "clusters": clusters,
            "user_map": user_map,
            "item_map": item_map,
            "rev_user_map": rev_user_map,
            "rev_item_map": rev_item_map
        }
        
        save_pickle(clustering_artifacts, config.CLUSTERING_ARTIFACT_PATH)
        
        # --- 5. Display Cluster Statistics ---
        print("\n--- Clustering Statistics ---")
        cluster_stats = pd.DataFrame({
            "Cluster ID": list(clusters.keys()),
            "Number of Users": [len(users) for users in clusters.values()]
        }).sort_values(by="Number of Users", ascending=False)
        print(cluster_stats.to_markdown(index=False))
        
        print("\n--- ✅ Step 1: Clustering Pipeline Completed Successfully ---\n")
        return True

    except FileNotFoundError:
        print(f"❌ Error: Source data file not found at '{config.SOURCE_DATASET_PATH}'.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred in the clustering pipeline: {e}")
        return False

if __name__ == '__main__':
    # This allows the script to be run standalone for testing/debugging
    run_clustering_pipeline()