import argparse
import config.config as config
from src.pipelines.clustering_pipeline import run_clustering_pipeline
from src.pipelines.embedding_pipeline import run_embedding_pipeline
from src.pipelines.similarity_pipeline import run_similarity_pipeline
from src.pipelines.prediction_pipeline import run_prediction_pipeline

def main(model_type: str, force_rerun: bool):
    """
    Orchestrates the recommendation model build pipeline with dependency checking.
    """
    print("="*60)
    print(f"🚀 Starting Recommendation System Build Pipeline")
    print(f"   Model Type: {model_type.upper()}")
    print(f"   Force Rerun: {force_rerun}")
    print("="*60)

    if model_type == 'pearson':
        similarity_path = config.PEARSON_SIMILARITY_PATH
        prediction_path = config.PEARSON_PREDICTIONS_PATH
    else: # mutifactor
        similarity_path = config.MUTIFACTOR_SIMILARITY_PATH
        prediction_path = config.MUTIFACTOR_PREDICTIONS_PATH

    if force_rerun:
        print("🔥 Force rerun enabled. Deleting existing artifacts...")
        artifacts_to_delete = [
            config.CLUSTERING_ARTIFACT_PATH, config.EMBEDDINGS_ARTIFACT_PATH,
            config.PEARSON_SIMILARITY_PATH, config.MUTIFACTOR_SIMILARITY_PATH,
            config.PEARSON_PREDICTIONS_PATH, config.MUTIFACTOR_PREDICTIONS_PATH
        ]
        for path in artifacts_to_delete:
            if path.exists():
                path.unlink()
                print(f"   - Deleted {path.name}")

    # --- Pipeline Execution with Dependency Checking ---
    
    # Step 1: Clustering
    if not config.CLUSTERING_ARTIFACT_PATH.exists():
        if not run_clustering_pipeline():
            print("❌ Clustering pipeline failed. Aborting.")
            return
    else:
        print("⏩ Step 1: Clustering artifacts already exist. Skipping.")

    # Step 2: Embeddings
    # Get modification time of embedding file BEFORE running the pipeline
    embedding_mod_time_before = config.EMBEDDINGS_ARTIFACT_PATH.stat().st_mtime if config.EMBEDDINGS_ARTIFACT_PATH.exists() else -1
    
    print("--- Delegating to Step 2: Embedding Pipeline (will check for completeness internally) ---")
    if not run_embedding_pipeline():
        print("❌ Embedding pipeline failed. Aborting.")
        return
        
    # Get modification time AFTER running the pipeline
    embedding_mod_time_after = config.EMBEDDINGS_ARTIFACT_PATH.stat().st_mtime if config.EMBEDDINGS_ARTIFACT_PATH.exists() else -1

    # --- NEW: Dependency Check Logic ---
    # If the embedding file was created or modified, downstream artifacts are stale.
    if embedding_mod_time_after > embedding_mod_time_before:
        print("🔄 Embedding artifact was updated. Invalidating downstream artifacts.")
        if similarity_path.exists():
            similarity_path.unlink()
            print(f"   - Deleted stale similarity file: {similarity_path.name}")
        if prediction_path.exists():
            prediction_path.unlink()
            print(f"   - Deleted stale prediction file: {prediction_path.name}")

    # Step 3: Similarity Calculation
    if not similarity_path.exists():
        if not run_similarity_pipeline(model_type=model_type):
            print("❌ Similarity pipeline failed. Aborting.")
            return
    else:
        print(f"⏩ Step 3: {model_type.capitalize()} similarity artifact already exist and is up-to-date. Skipping.")

    # Step 4: Prediction
    if not prediction_path.exists():
        if not run_prediction_pipeline(model_type=model_type):
            print("❌ Prediction pipeline failed. Aborting.")
            return
    else:
        print(f"⏩ Step 4: {model_type.capitalize()} prediction artifact already exist and is up-to-date. Skipping.")

    print("\n" + "="*60)
    print("🎉🎉🎉")
    print("✅  SUCCESS: Recommendation system build pipeline completed successfully!")
    print(f"   Final prediction artifact for the '{model_type.upper()}' model is ready at:")
    print(f"   ➡️   {prediction_path}")
    print("🎉🎉🎉")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full offline pipeline for the recommendation system.")
    parser.add_argument("--model", type=str, choices=['pearson', 'mutifactor'], default='mutifactor', help="The similarity model to use.")
    parser.add_argument("--force", action='store_true', help="If set, deletes all existing artifacts and reruns the entire pipeline.")
    args = parser.parse_args()
    
    main(model_type=args.model, force_rerun=args.force)