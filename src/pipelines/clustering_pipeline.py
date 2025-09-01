"""
================================================================================
Pipeline Step 1: User Clustering
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from config import config
from src.utils.file_handlers import save_pickle
from src.utils.live_logger import LiveLogger

# THAY ĐỔI: Hàm bây giờ nhận một DataFrame làm tham số
def run_clustering_pipeline(df: pd.DataFrame) -> bool:
    """
    Performs the user clustering step of the pipeline.
    This function receives a DataFrame, groups users into clusters based on their
    rating behavior using K-Means with ICR initialization, and saves the results.
    
    Args:
        df (pd.DataFrame): The source DataFrame containing user feedback data.
        
    Returns:
        bool: True if the pipeline step completes successfully, False otherwise.
    """
    try:
        # --- 1. Process Input Data ---
        # THAY ĐỔI: Không còn đọc file từ đĩa nữa.
        LiveLogger.log("✅ Source data received as DataFrame.")
        if df.empty:
            LiveLogger.log("❌ Error: Source DataFrame is empty.")
            return False
        LiveLogger.log(f"   - Processing {len(df)} reviews.")

        # --- 2. Create User-Item Matrix and Mappings ---
        # Logic này giữ nguyên vì nó hoạt động trên DataFrame đầu vào
        LiveLogger.log("🛠️  Creating user-item rating matrix and mappings...")
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
        LiveLogger.log(f"   - Matrix created with shape: {rating_matrix.shape} (Users x Items)")

        # --- 3. Apply K-Means with ICR Initialization ---
        # Logic này giữ nguyên
        LiveLogger.log("🧠 Applying K-Means clustering with ICR initialization...")
        
        num_ratings_per_user = np.sum(rating_matrix != 0, axis=1)
        # Đảm bảo không lấy nhiều centroid hơn số lượng user
        num_centroids = min(config.NUM_CLUSTERS, len(users))
        if num_centroids < config.NUM_CLUSTERS:
            LiveLogger.log(f"   - Warning: Number of users ({len(users)}) is less than NUM_CLUSTERS ({config.NUM_CLUSTERS}). Using {num_centroids} clusters instead.")
        
        centroid_indices = np.argsort(num_ratings_per_user)[-num_centroids:]
        initial_centroids = rating_matrix[centroid_indices]
        LiveLogger.log(f"   - Selected {len(initial_centroids)} initial centroids based on user activity (ICR).")
        
        user_means = np.true_divide(rating_matrix.sum(1), (rating_matrix != 0).sum(1))
        user_means[np.isnan(user_means)] = 0
        filled_matrix = np.where(rating_matrix == 0, user_means[:, np.newaxis], rating_matrix)
        
        kmeans = KMeans(
            n_clusters=num_centroids,
            init=initial_centroids,
            n_init=1,
            random_state=42
        )
        labels = kmeans.fit_predict(filled_matrix)
        LiveLogger.log(f"   - Successfully assigned {len(users)} users to {num_centroids} clusters.")

        # --- 4. Format and Save Artifacts ---
        # Logic này giữ nguyên
        LiveLogger.log(f"💾 Formatting and saving clustering artifacts to '{config.CLUSTERING_ARTIFACT_PATH.name}'...")
        
        clusters = {i: [] for i in range(num_centroids)}
        for user_idx, cluster_label in enumerate(labels):
            original_user_id = rev_user_map[user_idx]
            clusters[cluster_label].append(original_user_id)

        clustering_artifacts = {
            "clusters": clusters,
            "user_map": user_map,
            "item_map": item_map,
            "rev_user_map": rev_user_map,
            "rev_item_map": rev_item_map
        }
        
        save_pickle(clustering_artifacts, config.CLUSTERING_ARTIFACT_PATH)
        LiveLogger.log(f"   - Artifact saved successfully.")
        
        # --- 5. Log Cluster Statistics for Verification ---
        # Logic này giữ nguyên
        LiveLogger.log("\n--- Clustering Statistics ---")
        cluster_stats = pd.DataFrame({
            "Cluster ID": list(clusters.keys()),
            "Number of Users": [len(users) for users in clusters.values()]
        }).sort_values(by="Number of Users", ascending=False)
        
        stats_string = cluster_stats.to_markdown(index=False)
        LiveLogger.log(stats_string)
        
        return True

    # THAY ĐỔI: Loại bỏ FileNotFoundError vì không còn áp dụng
    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the clustering pipeline: {e}")
        return False

# THAY ĐỔI: Cập nhật khối chạy standalone
if __name__ == '__main__':
    # This block allows the script to be run directly for isolated testing.
    # It now simulates the orchestrator by first fetching data.
    class ConsoleLogger:
        @staticmethod
        def log(message): print(message)
        @staticmethod
        def start_progress(description, total): print(f"--- Starting Progress: {description} ---")
        @staticmethod
        def update_progress(current): pass

    import sys
    sys.modules['src.utils.live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    from src.utils.live_logger import LiveLogger
    
    # Import data loader để chạy thử nghiệm
    from utils.feedbacks_data_loader import fetch_and_prepare_data
    
    print("--- Running Clustering Pipeline in Standalone Mode ---")
    try:
        print("Step 0: Fetching data for standalone test...")
        # Lấy dữ liệu bằng cách gọi data loader, không dùng force_fetch để có thể test cache
        source_df = fetch_and_prepare_data(force_fetch=False) 
        
        if source_df is not None and not source_df.empty:
            print("Step 1: Running clustering logic...")
            success = run_clustering_pipeline(source_df)
            if success:
                print("\n✅ Clustering pipeline completed successfully in standalone mode.")
            else:
                print("\n❌ Clustering pipeline failed in standalone mode.")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")
            
    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")
        
    print("--- Standalone run finished ---")