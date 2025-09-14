"""
================================================================================
Pipeline Step 2: Review Embedding
================================================================================
"""

import pandas as pd
from typing import Dict, Any

from config import config as static_config
from src.utils.file_handlers import load_pickle, save_pickle
from src.utils.live_logger import LiveLogger
from src.services.gemini_service import GeminiService

def run_embedding_pipeline(df: pd.DataFrame, config_data: Dict[str, Any]) -> Dict:
    """
    Tạo và lưu các vector embedding cho review của người dùng.
    Pipeline này có thể phục hồi (upsert): nó chỉ xử lý những người dùng chưa có
    embedding.

    Args:
        df (pd.DataFrame): DataFrame nguồn chứa dữ liệu phản hồi của người dùng.
        config_data (Dict[str, Any]): Dictionary chứa các tham số cấu hình.

    Returns:
        dict: Một dictionary với "status", "message" (tùy chọn), và
              "embeddings_added" (số lượng embedding mới được tạo).
    """
    try:
        # --- 1. Tải Embedding hiện có ---
        LiveLogger.log(f"📂 Checking for existing embeddings at '{static_config.EMBEDDINGS_ARTIFACT_PATH}'...")
        user_embeddings = load_pickle(static_config.EMBEDDINGS_ARTIFACT_PATH)
        if user_embeddings is None:
            user_embeddings = {}
            LiveLogger.log("   - No existing artifact found. Starting fresh.")

        # --- 2. Xác định người dùng còn thiếu ---
        LiveLogger.log("✅ Source data received as DataFrame. Processing user reviews...")
        if df.empty:
            LiveLogger.log("❌ Error: Source DataFrame is empty.")
            return {"status": "error", "message": "Input DataFrame is empty.", "embeddings_added": 0}

        user_docs = df.groupby('user_id')['review'].apply(lambda reviews: ' '.join(reviews)).to_dict()

        existing_user_ids = set(user_embeddings.keys())
        all_user_ids = set(user_docs.keys())
        missing_user_ids = sorted(list(all_user_ids - existing_user_ids))

        if not missing_user_ids:
            LiveLogger.log("✅ All user embeddings are already present. Nothing to do.")
            return {"status": "success", "embeddings_added": 0}

        LiveLogger.log(f"   - Found {len(all_user_ids)} total users in source data.")
        LiveLogger.log(f"   - Found {len(existing_user_ids)} existing embeddings.")
        LiveLogger.log(f"   - Found {len(missing_user_ids)} users requiring new embeddings.")

        # --- 3. Tạo Embedding cho người dùng còn thiếu ---
        LiveLogger.log("🧠 Initializing Gemini Service and generating new embeddings...")
        gemini_service = GeminiService(
            env_prefix=static_config.GEMINI_ENV_PREFIX,
            model=static_config.EMBEDDING_MODEL,
            task_type=static_config.EMBEDDING_TASK_TYPE,
            dim=static_config.EMBEDDING_DIM
        )

        docs_to_embed = [user_docs[uid] for uid in missing_user_ids]
        new_embeddings = {}
        has_failures = False
        batch_size = config_data['EMBEDDING_BATCH_SIZE']
        num_batches = (len(docs_to_embed) + batch_size - 1) // batch_size
        LiveLogger.start_progress(description=f"Embedding {len(missing_user_ids)} users", total=num_batches)

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch_ids = missing_user_ids[start_index:end_index]
            batch_docs = docs_to_embed[start_index:end_index]

            try:
                embeddings_result = gemini_service.embed_content(batch_docs, batch_index=i)
                if embeddings_result:
                    for user_id, emb in zip(batch_ids, embeddings_result):
                        new_embeddings[user_id] = emb
                else:
                    has_failures = True
            except gemini_service.google_exceptions.ResourceExhausted:
                has_failures = True
                LiveLogger.log(f"   - ❌ Batch #{i} failed after retries due to error.")

            LiveLogger.update_progress(current=i + 1)

        LiveLogger.log(f"✅ Finished embedding process.")

        num_new_embeddings = len(new_embeddings)

        if num_new_embeddings != len(missing_user_ids):
             LiveLogger.log(f"⚠️ Warning: Could not generate embeddings for all missing users. "
                           f"Successfully generated {num_new_embeddings} out of {len(missing_user_ids)}.")

        # --- 4. Cập nhật và Lưu Artifact ---
        if num_new_embeddings > 0:
            LiveLogger.log(f"💾 Updating and saving embeddings artifact to '{static_config.EMBEDDINGS_ARTIFACT_PATH.name}'...")
            user_embeddings.update(new_embeddings)
            save_pickle(user_embeddings, static_config.EMBEDDINGS_ARTIFACT_PATH)
            LiveLogger.log(f"   - Total embeddings now in file: {len(user_embeddings)}")

        # --- 5. Trả về trạng thái cuối cùng ---
        if has_failures:
            return {"status": "incomplete", "embeddings_added": num_new_embeddings}
        else:
            return {"status": "success", "embeddings_added": num_new_embeddings}

    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the embedding pipeline: {e}")
        return {"status": "error", "message": str(e), "embeddings_added": 0}

# Khối chạy độc lập (giữ nguyên không đổi)
if __name__ == '__main__':
    class ConsoleLogger:
        @staticmethod
        def log(message): print(message)
        @staticmethod
        def start_progress(description, total): print(f"--- Starting Progress: {description} (Total: {total}) ---")
        @staticmethod
        def update_progress(current): print(f"--- Progress: {current} ---")
        @staticmethod
        def request_admin_action(action_type, message): print(f"--- Admin Action Requested: {action_type} - {message} ---")

    import sys
    sys.modules['src/utils/live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    from src.utils.live_logger import LiveLogger

    from utils.feedbacks_data_loader import fetch_and_prepare_data
    from config.config import get_pipeline_config

    print("--- Running Embedding Pipeline in Standalone Mode ---")
    try:
        print("Step 0a: Fetching data for standalone test...")
        source_df = fetch_and_prepare_data(force_fetch=False)

        print("Step 0b: Loading pipeline configuration...")
        test_config = get_pipeline_config()
        print(f"   - Loaded config with EMBEDDING_BATCH_SIZE = {test_config['EMBEDDING_BATCH_SIZE']}")

        if source_df is not None and not source_df.empty:
            print("\nStep 1: Running embedding logic...")
            result = run_embedding_pipeline(source_df, test_config)
            print(f"\n✅ Embedding pipeline completed with status: {result['status']}")
            print(f"   - Embeddings added: {result.get('embeddings_added', 'N/A')}")
            if result.get('message'):
                print(f"   - Message: {result['message']}")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")

    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")

    print("--- Standalone run finished ---")
