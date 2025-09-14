# src/pipeline_orchestrator.py

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import config.config as static_config
from config.config import get_pipeline_config
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.embedding_pipeline import run_embedding_pipeline
from src.pipelines.similarity_pipeline import run_similarity_pipeline
from src.pipelines.prediction_pipeline import run_prediction_pipeline
from src.utils.file_handlers import load_pickle # Thêm import này
from src.utils.live_logger import LiveLogger
from src.utils.feedbacks_data_loader import fetch_and_prepare_data

CONFIG_SNAPSHOT_PATH = static_config.ARTIFACTS_DIR / "pipeline_config_snapshot.json"

def _load_previous_config() -> Dict[str, Any]:
    if CONFIG_SNAPSHOT_PATH.exists():
        try:
            with open(CONFIG_SNAPSHOT_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def _save_config_snapshot(config_dict: Dict[str, Any]):
    with open(CONFIG_SNAPSHOT_PATH, 'w') as f:
        json.dump(config_dict, f, indent=4)

def _check_config_changes(current_config: Dict, previous_config: Dict) -> Dict[str, bool]:
    if not previous_config:
        return {'clustering': True, 'embedding': True, 'similarity': True, 'prediction': True}
    changes = {
        'clustering': current_config.get('NUM_CLUSTERS') != previous_config.get('NUM_CLUSTERS'),
        'embedding': current_config.get('EMBEDDING_BATCH_SIZE') != previous_config.get('EMBEDDING_BATCH_SIZE'),
        'similarity': (
            current_config.get('NUM_INFLUENCERS') != previous_config.get('NUM_INFLUENCERS') or
            current_config.get('MUTIFACTOR_ALPHA') != previous_config.get('MUTIFACTOR_ALPHA') or
            current_config.get('USE_BEHAVIORAL_DATA') != previous_config.get('USE_BEHAVIORAL_DATA') or
            current_config.get('INCLUDE_SEARCH_BEHAVIOR') != previous_config.get('INCLUDE_SEARCH_BEHAVIOR') or
            current_config.get('BEHAVIORAL_WEIGHTS') != previous_config.get('BEHAVIORAL_WEIGHTS') or
            current_config.get('BEHAVIORAL_SIM_WEIGHT') != previous_config.get('BEHAVIORAL_SIM_WEIGHT')
        ),
        'prediction': current_config.get('NUM_NEIGHBORS') != previous_config.get('NUM_NEIGHBORS'),
    }
    return changes

def run_full_pipeline(model_type: str = 'mutifactor', force_rerun: bool = False, steps_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
    try:
        LiveLogger.log("⚙️  Loading the latest pipeline configuration...")
        current_pipeline_config = get_pipeline_config()
        previous_pipeline_config = _load_previous_config()
        config_changes = _check_config_changes(current_pipeline_config, previous_pipeline_config)

        LiveLogger.log("\n--- Step 0: Data Loading ---")
        should_force_fetch = force_rerun or (not steps_to_run)
        source_df = fetch_and_prepare_data(force_fetch=should_force_fetch)
        if source_df is None or source_df.empty:
            raise Exception("Source data could not be loaded or is empty.")

        if steps_to_run is None:
            steps_to_run = []

        similarity_path = static_config.MUTIFACTOR_SIMILARITY_PATH if model_type == 'mutifactor' else static_config.PEARSON_SIMILARITY_PATH
        prediction_path = static_config.MUTIFACTOR_PREDICTIONS_PATH if model_type == 'mutifactor' else static_config.PEARSON_PREDICTIONS_PATH

        # --- 1. Lên kế hoạch xóa Artifact ---
        artifacts_to_delete = set()

        # A. Xóa do force_rerun
        if force_rerun:
            LiveLogger.log("🔥 Full force rerun enabled. Deleting all artifacts...")
            artifacts_to_delete.update([
                static_config.CLUSTERING_ARTIFACT_PATH, static_config.EMBEDDINGS_ARTIFACT_PATH,
                similarity_path, prediction_path, CONFIG_SNAPSHOT_PATH
            ])
        else:
            # B. Xóa do yêu cầu chạy lại bước cụ thể (trừ embedding)
            if 'clustering' in steps_to_run: artifacts_to_delete.add(static_config.CLUSTERING_ARTIFACT_PATH)
            if 'similarity' in steps_to_run: artifacts_to_delete.add(similarity_path)
            if 'prediction' in steps_to_run: artifacts_to_delete.add(prediction_path)

            # C. Xóa do thay đổi cấu hình
            if config_changes['clustering']: artifacts_to_delete.add(static_config.CLUSTERING_ARTIFACT_PATH)
            if config_changes['embedding']: artifacts_to_delete.add(static_config.EMBEDDINGS_ARTIFACT_PATH)
            if config_changes['similarity']: artifacts_to_delete.add(similarity_path)
            if config_changes['prediction']: artifacts_to_delete.add(prediction_path)

            # D. Xóa do thay đổi chiều dữ liệu (user/item mới)
            clustering_artifact = load_pickle(static_config.CLUSTERING_ARTIFACT_PATH)
            if clustering_artifact:
                old_users, old_items = set(clustering_artifact['user_map'].keys()), set(clustering_artifact['item_map'].keys())
                new_users, new_items = set(source_df['user_id'].unique()), set(source_df['conference_key'].unique())
                if old_users != new_users or old_items != new_items:
                    LiveLogger.log("💡 Data dimension change detected (new users/items). Invalidating clustering.")
                    artifacts_to_delete.add(static_config.CLUSTERING_ARTIFACT_PATH)

        # --- 2. Xử lý các phụ thuộc (Cascading Deletion) ---
        if static_config.CLUSTERING_ARTIFACT_PATH in artifacts_to_delete:
            LiveLogger.log("   - Dependency: Clustering change requires invalidating all downstream artifacts.")
            artifacts_to_delete.update([static_config.EMBEDDINGS_ARTIFACT_PATH, similarity_path, prediction_path])

        if static_config.EMBEDDINGS_ARTIFACT_PATH in artifacts_to_delete:
            LiveLogger.log("   - Dependency: Embedding change requires invalidating downstream artifacts.")
            artifacts_to_delete.update([similarity_path, prediction_path])

        if similarity_path in artifacts_to_delete:
            LiveLogger.log("   - Dependency: Similarity change requires invalidating Prediction artifact.")
            artifacts_to_delete.add(prediction_path)

        for path in artifacts_to_delete:
            if path.exists():
                path.unlink()
                LiveLogger.log(f"   - Deleted stale artifact: {path.name}")

        # --- 3. Thực thi Pipeline ---
        # Step 1: Clustering
        LiveLogger.log("\n--- Step 1: Clustering ---")
        if not static_config.CLUSTERING_ARTIFACT_PATH.exists():
            if not run_clustering_pipeline(source_df, current_pipeline_config):
                raise Exception("Clustering pipeline failed.")
        else:
            LiveLogger.log("⏩ Clustering artifact is up-to-date. Skipping.")

        # Step 2: Embedding (và phát hiện thay đổi dữ liệu review/behavior)
        LiveLogger.log("\n--- Step 2: Embedding (Data Change Detection) ---")
        embedding_result = run_embedding_pipeline(source_df, current_pipeline_config)

        if embedding_result["status"] == "error":
            raise Exception(f"Embedding pipeline failed critically: {embedding_result.get('message', 'Unknown error')}")
        if embedding_result["status"] == "incomplete":
            LiveLogger.request_admin_action(action_type="EMBEDDING_INCOMPLETE", message="Embedding process was incomplete.")
            return {"status": "paused", "reason": "EMBEDDING_INCOMPLETE"}

        # Logic quyết định muộn: Chỉ xóa các bước sau KHI biết embedding có thay đổi
        embeddings_added = embedding_result.get("embeddings_added", 0)
        if embeddings_added > 0:
            LiveLogger.log(f"✅ Data content change detected: {embeddings_added} new embeddings were added.")
            LiveLogger.log("   - Invalidating downstream artifacts (similarity, prediction) to re-compute.")
            if similarity_path.exists():
                similarity_path.unlink()
                LiveLogger.log(f"   - Deleted stale artifact: {similarity_path.name}")
            if prediction_path.exists():
                prediction_path.unlink()
                LiveLogger.log(f"   - Deleted stale artifact: {prediction_path.name}")
        else:
            LiveLogger.log("✅ No new data content detected that requires embedding.")

        # Step 3: Similarity Calculation
        LiveLogger.log(f"\n--- Step 3: Similarity ({model_type.upper()}) ---")
        if not similarity_path.exists():
            if not run_similarity_pipeline(source_df, model_type=model_type, config_data=current_pipeline_config):
                raise Exception("Similarity pipeline failed.")
        else:
            LiveLogger.log("⏩ Similarity artifact is up-to-date. Skipping.")

        # Step 4: Prediction
        LiveLogger.log(f"\n--- Step 4: Prediction ({model_type.upper()}) ---")
        if not prediction_path.exists():
            if not run_prediction_pipeline(source_df, model_type=model_type, config_data=current_pipeline_config):
                raise Exception("Prediction pipeline failed.")
        else:
            LiveLogger.log("⏩ Prediction artifact is up-to-date. Skipping.")

        _save_config_snapshot(current_pipeline_config)
        LiveLogger.log("\n✅ Saved current configuration as a snapshot for the next run.")
        LiveLogger.log("\n🎉 Pipeline completed successfully!")
        return {"status": "success", "artifact_path": str(prediction_path)}

    except Exception as e:
        LiveLogger.log(f"\n❌ ORCHESTRATOR ERROR: Pipeline execution halted. Reason: {e}")
        return {"status": "error", "error_message": str(e)}

# Khối chạy độc lập (giữ nguyên không đổi)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full offline pipeline for the recommendation system.")
    parser.add_argument("--model", type=str, choices=['pearson', 'mutifactor'], default='mutifactor', help="The similarity model to use.")
    parser.add_argument("--force", action='store_true', help="If set, fetches fresh data, deletes all artifacts and reruns the entire pipeline.")
    parser.add_argument("--steps", nargs='+', help="List of specific steps to run (e.g., --steps embedding similarity).")
    args = parser.parse_args()

    class ConsoleLogger:
        @staticmethod
        def log(message): print(message)
        @staticmethod
        def start_run(model_type): print(f"--- Starting standalone run for model: {model_type} ---")
        @staticmethod
        def end_run(status, result): print(f"--- Standalone run finished with status: {status} ---")
        @staticmethod
        def request_admin_action(action_type, message): print(f"--- Admin Action Requested: {action_type} - {message} ---")

    import sys
    if 'src.utils.live_logger' not in sys.modules or getattr(sys.modules['src.utils.live_logger'], 'LiveLogger', None) != ConsoleLogger:
        sys.modules['src.utils.live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})

    from src.utils.live_logger import LiveLogger

    LiveLogger.start_run(args.model)
    result = run_full_pipeline(model_type=args.model, force_rerun=args.force, steps_to_run=args.steps)
    LiveLogger.end_run(result['status'], result)
