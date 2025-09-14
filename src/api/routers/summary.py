# api/routers/summary.py

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import numpy as np
import config.config as config
from src.utils.file_handlers import load_pickle
from src.api.models import ArtifactSummary, UserDetailsResponse, ClusterInfo, NeighborInfo, PredictionInfo

router = APIRouter()

# --- Helper functions for summary generation ---
def get_clustering_summary():
    artifact = load_pickle(config.CLUSTERING_ARTIFACT_PATH)
    if not artifact: return None
    clusters = artifact.get("clusters", {})
    cluster_stats = {f"Cluster {k}": len(v) for k, v in clusters.items()}
    return {
        "total_clusters": len(clusters),
        "total_users_clustered": sum(len(v) for v in clusters.values()),
        "users_per_cluster": sorted(cluster_stats.items(), key=lambda item: item[1], reverse=True),
        "total_unique_items": len(artifact.get("item_map", {}))
    }

def get_embedding_summary():
    artifact = load_pickle(config.EMBEDDINGS_ARTIFACT_PATH)
    if not artifact: return None
    return {
        "total_users_with_embedding": len(artifact),
        "embedding_dimension": len(next(iter(artifact.values()))) if artifact else 0
    }

def get_similarity_summary(model_type: str):
    path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
    artifact = load_pickle(path)
    if not artifact: return None
    sample_user_id = next(iter(artifact.keys())) if artifact else None
    sample_similarities = artifact.get(sample_user_id, [])

    # --- THAY ĐỔI: Làm tròn điểm số để dễ đọc hơn ---
    formatted_neighbors = [
        (uid, round(score, 4)) for uid, score in sample_similarities[:5]
    ]

    return {
        "total_users_with_scores": len(artifact),
        "sample_user_id": sample_user_id,
        "sample_top_5_neighbors": formatted_neighbors # Trả về dữ liệu đã được làm tròn
    }


def get_prediction_summary(model_type: str):
    path = config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else config.PEARSON_PREDICTIONS_PATH
    artifact = load_pickle(path)
    if not artifact: return None
    matrix = artifact.get("prediction_matrix")
    if matrix is None: return {"error": "Prediction matrix not found in artifact"}
    return {
        "matrix_shape": matrix.shape,
        "total_predictions": matrix.size,
        "average_predicted_rating": round(np.mean(matrix), 2),
        "min_predicted_rating": round(np.min(matrix), 2),
        "max_predicted_rating": round(np.max(matrix), 2),
    }

@router.get("/artifact-summary/{model_type}", response_model=List[ArtifactSummary])
def get_artifact_summaries(model_type: str):
    if model_type not in ['mutifactor', 'pearson']:
        raise HTTPException(status_code=400, detail="Invalid model_type. Must be 'mutifactor' or 'pearson'.")

    summaries = []
    artifact_processors = {
        "Clustering": (config.CLUSTERING_ARTIFACT_PATH, get_clustering_summary),
        "Embeddings": (config.EMBEDDINGS_ARTIFACT_PATH, get_embedding_summary),
        "Similarity": (None, lambda: get_similarity_summary(model_type)),
        "Predictions": (None, lambda: get_prediction_summary(model_type)),
    }

    for name, (path, processor_func) in artifact_processors.items():
        summary = ArtifactSummary(artifact_name=name, exists=False)
        try:
            current_path = path
            if name == "Similarity":
                current_path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
            elif name == "Predictions":
                current_path = config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else config.PEARSON_PREDICTIONS_PATH

            if current_path and current_path.exists():
                summary.exists = True
                summary.data_summary = processor_func()
            else:
                summary.data_summary = {"message": "Artifact file not found."}
        except Exception as e:
            summary.error = f"Failed to process artifact: {str(e)}"
        summaries.append(summary)
    return summaries




# --- THÊM MỚI: Endpoint để lấy chi tiết thông tin của một user ---

@router.get("/user-details/{model_type}/{user_id}", response_model=UserDetailsResponse)
def get_user_details(model_type: str, user_id: str):
    """
    Cung cấp thông tin chi tiết về một user cụ thể từ các artifact đã tính toán.
    """
    if model_type not in ['mutifactor', 'pearson']:
        raise HTTPException(status_code=400, detail="Invalid model_type.")

    try:
        # --- 1. Tải các artifact cần thiết ---
        clustering_artifact = load_pickle(config.CLUSTERING_ARTIFACT_PATH)
        embedding_artifact = load_pickle(config.EMBEDDINGS_ARTIFACT_PATH)

        sim_path = config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else config.PEARSON_SIMILARITY_PATH
        pred_path = config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else config.PEARSON_PREDICTIONS_PATH

        similarity_artifact = load_pickle(sim_path)
        prediction_artifact = load_pickle(pred_path)

        # --- 2. Kiểm tra sự tồn tại của user ---
        user_map = clustering_artifact.get("user_map") if clustering_artifact else None
        if not user_map or user_id not in user_map:
            return UserDetailsResponse(user_id=user_id, model_type=model_type, is_found=False, has_embedding=False)

        response = UserDetailsResponse(user_id=user_id, model_type=model_type, is_found=True, has_embedding=False)

        # --- 3. Trích xuất thông tin Clustering ---
        clusters = clustering_artifact.get("clusters", {}) if clustering_artifact else {}
        for cluster_id, users_in_cluster in clusters.items():
            if user_id in users_in_cluster:
                response.cluster_info = ClusterInfo(cluster_id=cluster_id, cluster_size=len(users_in_cluster))
                break

        # --- 4. Trích xuất thông tin Embedding ---
        if embedding_artifact and user_id in embedding_artifact:
            response.has_embedding = True

        # --- 5. Trích xuất thông tin Similarity ---
        if similarity_artifact and user_id in similarity_artifact:
            neighbors = similarity_artifact[user_id][1:11] # Lấy 10 hàng xóm gần nhất (bỏ qua user đầu tiên là chính nó)
            response.similarity_info = [
                NeighborInfo(user_id=uid, similarity_score=round(score, 4))
                for uid, score in neighbors
            ]

        # --- 6. Trích xuất thông tin Prediction ---
        if prediction_artifact:
            user_idx = user_map.get(user_id)
            pred_matrix = prediction_artifact.get("prediction_matrix")
            rev_item_map = clustering_artifact.get("rev_item_map", {}) if clustering_artifact else {}

            user_predictions = pred_matrix[user_idx, :]
            sorted_indices = np.argsort(user_predictions)

            # Lấy 10 gợi ý cao nhất
            top_indices = sorted_indices[-10:][::-1]
            response.top_predictions = [
                PredictionInfo(
                    conference_id=rev_item_map.get(i, "N/A"),
                    predicted_rating=round(float(user_predictions[i]), 4)
                ) for i in top_indices
            ]

            # Lấy 10 gợi ý thấp nhất
            bottom_indices = sorted_indices[:10]
            response.bottom_predictions = [
                PredictionInfo(
                    conference_id=rev_item_map.get(i, "N/A"),
                    predicted_rating=round(float(user_predictions[i]), 4)
                ) for i in bottom_indices
            ]

        return response

    except Exception as e:
        # Trả về lỗi nếu có vấn đề khi đọc file artifact
        raise HTTPException(status_code=500, detail=f"Error processing artifacts: {str(e)}")
