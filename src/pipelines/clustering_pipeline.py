"""
================================================================================
Pipeline Step 1: User Clustering
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any
# Không cần import config.config nữa vì config sẽ được truyền vào
from src.utils.file_handlers import save_pickle, load_pickle
from src.utils.live_logger import LiveLogger
from config import config as static_config # Import để lấy đường dẫn artifact

# THAY ĐỔI: Hàm bây giờ nhận DataFrame và một dict cấu hình
def run_clustering_pipeline(df: pd.DataFrame, config_data: Dict[str, Any]) -> bool:
    """
    Thực hiện bước phân cụm người dùng của pipeline.
    Hàm này nhận một DataFrame, nhóm người dùng thành các cụm dựa trên
    hành vi đánh giá của họ bằng K-Means với khởi tạo ICR, và lưu kết quả.

    Args:
        df (pd.DataFrame): DataFrame nguồn chứa dữ liệu phản hồi của người dùng.
        config_data (Dict[str, Any]): Một dictionary chứa các tham số cấu hình
                                      cho lần chạy pipeline này (ví dụ: NUM_CLUSTERS).

    Returns:
        bool: True nếu bước pipeline hoàn thành thành công, False nếu ngược lại.
    """
    try:
        # --- 1. Process Input Data ---
        LiveLogger.log("✅ Source data received as DataFrame.")
        if df.empty:
            LiveLogger.log("❌ Error: Source DataFrame is empty.")
            return False
        LiveLogger.log(f"   - Processing {len(df)} reviews.")

        # --- 2. Create User-Item Matrix and Mappings ---
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
        LiveLogger.log("🧠 Applying K-Means clustering with ICR initialization...")

        num_ratings_per_user = np.sum(rating_matrix != 0, axis=1)

        # THAY ĐỔI: Sử dụng giá trị NUM_CLUSTERS từ dict config_data được truyền vào
        num_clusters_from_config = config_data['NUM_CLUSTERS']

        # Đảm bảo không lấy nhiều centroid hơn số lượng user
        num_centroids = min(num_clusters_from_config, len(users))
        if num_centroids < num_clusters_from_config:
            LiveLogger.log(f"   - Warning: Number of users ({len(users)}) is less than NUM_CLUSTERS ({num_clusters_from_config}). Using {num_centroids} clusters instead.")

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
        # Sử dụng đường dẫn tĩnh từ static_config
        LiveLogger.log(f"💾 Formatting and saving clustering artifacts to '{static_config.CLUSTERING_ARTIFACT_PATH.name}'...")

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

        save_pickle(clustering_artifacts, static_config.CLUSTERING_ARTIFACT_PATH)
        LiveLogger.log(f"   - Artifact saved successfully.")

        # --- 5. Log Cluster Statistics for Verification ---
        LiveLogger.log("\n--- Clustering Statistics ---")
        cluster_stats = pd.DataFrame({
            "Cluster ID": list(clusters.keys()),
            "Number of Users": [len(users) for users in clusters.values()]
        }).sort_values(by="Number of Users", ascending=False)

        stats_string = cluster_stats.to_markdown(index=False)
        LiveLogger.log(stats_string)

        return True

    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the clustering pipeline: {e}")
        return False

# THAY ĐỔI: Cập nhật khối chạy standalone để mô phỏng việc truyền config
if __name__ == '__main__':
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

    # Import data loader và config loader để chạy thử nghiệm
    from utils.feedbacks_data_loader import fetch_and_prepare_data
    from config.config import get_pipeline_config

    print("--- Running Clustering Pipeline in Standalone Mode ---")
    try:
        print("Step 0a: Fetching data for standalone test...")
        source_df = fetch_and_prepare_data(force_fetch=False)

        print("Step 0b: Loading pipeline configuration...")
        # Mô phỏng orchestrator: tải config trước khi chạy
        test_config = get_pipeline_config()
        print(f"   - Loaded config with NUM_CLUSTERS = {test_config['NUM_CLUSTERS']}")

        if source_df is not None and not source_df.empty:
            print("\nStep 1: Running clustering logic...")
            # Truyền cả DataFrame và config vào hàm
            success = run_clustering_pipeline(source_df, test_config)
            if success:
                print("\n✅ Clustering pipeline completed successfully in standalone mode.")
            else:
                print("\n❌ Clustering pipeline failed in standalone mode.")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")

    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")

    print("--- Standalone run finished ---")
