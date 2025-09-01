# src/pipeline_orchestrator.py

import argparse
from pathlib import Path
import config.config as config
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.embedding_pipeline import run_embedding_pipeline
from src.pipelines.similarity_pipeline import run_similarity_pipeline
from src.pipelines.prediction_pipeline import run_prediction_pipeline
from src.utils.live_logger import LiveLogger
from src.utils.feedbacks_data_loader import fetch_and_prepare_data
from typing import Dict, Any, List, Optional

def run_full_pipeline(model_type: str = 'mutifactor', force_rerun: bool = False, steps_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Orchestrates the recommendation model build pipeline.
    This function manages the control flow and dependencies, while the actual
    logging is handled by the LiveLogger and the individual pipeline scripts.

    Args:
        model_type (str): The similarity model to use ('pearson' or 'mutifactor').
        force_rerun (bool): If True, fetches fresh data, deletes all artifacts, and reruns everything.
        steps_to_run (List[str]): A list of specific steps to run.

    Returns:
        Dict[str, Any]: A dictionary containing the final status and artifact path.
    """
    try:
        # --- Step 0: Data Loading ---
        # Tách biệt logic tải dữ liệu ra khỏi các bước tính toán.
        # force_rerun sẽ quyết định có fetch lại dữ liệu từ API hay không.
        LiveLogger.log("\n--- Step 0: Data Loading ---")
        # force_fetch=True khi người dùng yêu cầu force_rerun toàn bộ pipeline.
        source_df = fetch_and_prepare_data(force_fetch=force_rerun)
        if source_df is None or source_df.empty:
            raise Exception("Source data could not be loaded or is empty.")
        
        # --- Logic hiện có được giữ nguyên và điều chỉnh ---
        if steps_to_run is None:
            steps_to_run = []

        if model_type == 'pearson':
            similarity_path = config.PEARSON_SIMILARITY_PATH
            prediction_path = config.PEARSON_PREDICTIONS_PATH
        else: # mutifactor
            similarity_path = config.MUTIFACTOR_SIMILARITY_PATH
            prediction_path = config.MUTIFACTOR_PREDICTIONS_PATH

        # --- 1. Determine which artifacts to delete based on run options ---
        artifacts_to_delete = set()
        if force_rerun:
            LiveLogger.log("🔥 Full force rerun enabled. Deleting all artifacts...")
            artifacts_to_delete.update([
                config.CLUSTERING_ARTIFACT_PATH, config.EMBEDDINGS_ARTIFACT_PATH,
                config.PEARSON_SIMILARITY_PATH, config.MUTIFACTOR_SIMILARITY_PATH,
                config.PEARSON_PREDICTIONS_PATH, config.MUTIFACTOR_PREDICTIONS_PATH
            ])
        else:
            if 'clustering' in steps_to_run: artifacts_to_delete.add(config.CLUSTERING_ARTIFACT_PATH)
            if 'similarity' in steps_to_run: artifacts_to_delete.add(similarity_path)
            if 'prediction' in steps_to_run: artifacts_to_delete.add(prediction_path)

        # Handle dependencies
        if config.CLUSTERING_ARTIFACT_PATH in artifacts_to_delete:
            LiveLogger.log("   - Dependency: Clustering change requires invalidating downstream artifacts.")
            artifacts_to_delete.update([config.EMBEDDINGS_ARTIFACT_PATH, similarity_path, prediction_path])
        
        if similarity_path in artifacts_to_delete:
            LiveLogger.log("   - Dependency: Similarity change requires invalidating Prediction artifact.")
            artifacts_to_delete.add(prediction_path)

        for path in artifacts_to_delete:
            if path.exists():
                path.unlink()
                LiveLogger.log(f"   - Deleted stale artifact for rerun: {path.name}")

        # --- 2. Pipeline Execution ---
        # Mỗi bước pipeline giờ đây sẽ nhận DataFrame làm đầu vào.

        # Step 1: Clustering
        LiveLogger.log("\n--- Step 1: Clustering ---")
        if not config.CLUSTERING_ARTIFACT_PATH.exists():
            if not run_clustering_pipeline(source_df): 
                raise Exception("Clustering pipeline failed.")
        else:
            LiveLogger.log("⏩ Clustering artifact exists. Skipping.")

        # Step 2: Embeddings
        LiveLogger.log("\n--- Step 2: Embedding ---")
        embedding_mod_time_before = config.EMBEDDINGS_ARTIFACT_PATH.stat().st_mtime if config.EMBEDDINGS_ARTIFACT_PATH.exists() else -1
        
        if 'embedding' in steps_to_run or not config.EMBEDDINGS_ARTIFACT_PATH.exists():
            embedding_result = run_embedding_pipeline(source_df)
            
            if embedding_result["status"] == "error":
                raise Exception(f"Embedding pipeline failed critically: {embedding_result.get('message', 'Unknown error')}")
            
            if embedding_result["status"] == "incomplete":
                LiveLogger.request_admin_action(
                    action_type="EMBEDDING_INCOMPLETE",
                    message="Embedding process was incomplete due to API errors. Please choose how to proceed."
                )
                return {"status": "paused", "reason": "EMBEDDING_INCOMPLETE"}
        else:
            LiveLogger.log("⏩ Embedding artifact seems complete. Skipping.")

        embedding_mod_time_after = config.EMBEDDINGS_ARTIFACT_PATH.stat().st_mtime if config.EMBEDDINGS_ARTIFACT_PATH.exists() else -1

        if embedding_mod_time_after > embedding_mod_time_before:
            LiveLogger.log("🔄 Embedding artifact was updated. Invalidating downstream artifacts.")
            if similarity_path.exists():
                similarity_path.unlink()
                LiveLogger.log(f"   - Deleted stale artifact: {similarity_path.name}")
            if prediction_path.exists():
                prediction_path.unlink()
                LiveLogger.log(f"   - Deleted stale artifact: {prediction_path.name}")

        # Step 3: Similarity Calculation
        LiveLogger.log(f"\n--- Step 3: Similarity ({model_type.upper()}) ---")
        if not similarity_path.exists():
            if not run_similarity_pipeline(source_df, model_type=model_type): 
                raise Exception("Similarity pipeline failed.")
        else:
            LiveLogger.log("⏩ Similarity artifact exists. Skipping.")

        # Step 4: Prediction
        LiveLogger.log(f"\n--- Step 4: Prediction ({model_type.upper()}) ---")
        if not prediction_path.exists():
            if not run_prediction_pipeline(source_df, model_type=model_type): 
                raise Exception("Prediction pipeline failed.")
        else:
            LiveLogger.log("⏩ Prediction artifact exists. Skipping.")

        LiveLogger.log("\n🎉 Pipeline completed successfully!")
        return {"status": "success", "artifact_path": str(prediction_path)}

    except Exception as e:
        LiveLogger.log(f"\n❌ ORCHESTRATOR ERROR: Pipeline execution halted. Reason: {e}")
        return {"status": "error", "error_message": str(e)}

    
# This block allows the script to still be run from the command line for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full offline pipeline for the recommendation system.")
    parser.add_argument("--model", type=str, choices=['pearson', 'mutifactor'], default='mutifactor', help="The similarity model to use.")
    parser.add_argument("--force", action='store_true', help="If set, fetches fresh data, deletes all artifacts and reruns the entire pipeline.")
    parser.add_argument("--steps", nargs='+', help="List of specific steps to run (e.g., --steps embedding similarity).")
    args = parser.parse_args()
    
    # For standalone execution, we need a console logger
    class ConsoleLogger:
        @staticmethod
        def log(message): print(message)
        @staticmethod
        def start_run(model_type): print(f"--- Starting standalone run for model: {model_type} ---")
        @staticmethod
        def end_run(status, result): print(f"--- Standalone run finished with status: {status} ---")
        # Add dummy methods for new features to avoid errors in standalone mode
        @staticmethod
        def request_admin_action(action_type, message): print(f"--- Admin Action Requested: {action_type} - {message} ---")

    # Temporarily replace the LiveLogger with our console version
    import sys
    # Check if the mock is already in place to avoid re-import issues
    if 'src.utils.live_logger' not in sys.modules or getattr(sys.modules['src.utils.live_logger'], 'LiveLogger', None) != ConsoleLogger:
        sys.modules['src.utils.live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    
    from src.utils.live_logger import LiveLogger
    
    LiveLogger.start_run(args.model)
    result = run_full_pipeline(model_type=args.model, force_rerun=args.force, steps_to_run=args.steps)
    LiveLogger.end_run(result['status'], result)