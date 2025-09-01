import pandas as pd
import time
from typing import Dict

import config.config as config
from src.utils.file_handlers import load_pickle, save_pickle
from src.utils.live_logger import LiveLogger
from src.services.gemini_service import GeminiService 

# --- Main Pipeline Function ---
# THAY ĐỔI: Hàm bây giờ nhận một DataFrame làm tham số
def run_embedding_pipeline(df: pd.DataFrame) -> Dict:
    """
    Generates and saves vector embeddings for user reviews.
    This pipeline is recoverable: it only processes users who don't have an
    embedding yet. It receives a DataFrame instead of reading from a file.
    
    Args:
        df (pd.DataFrame): The source DataFrame containing user feedback data.

    Returns:
        dict: A dictionary with "status" ('success', 'incomplete', 'error') 
              and an optional "message".
    """
    try:
        # --- 1. Load Existing Embeddings ---
        # Logic này giữ nguyên, nó độc lập với nguồn dữ liệu
        LiveLogger.log(f"📂 Checking for existing embeddings at '{config.EMBEDDINGS_ARTIFACT_PATH}'...")
        user_embeddings = load_pickle(config.EMBEDDINGS_ARTIFACT_PATH)
        if user_embeddings is None:
            user_embeddings = {}
            LiveLogger.log("   - No existing artifact found. Starting fresh.")

        # --- 2. Identify Missing Users from DataFrame ---
        # THAY ĐỔI: Không đọc file, xử lý trực tiếp từ DataFrame đầu vào
        LiveLogger.log("✅ Source data received as DataFrame. Processing user reviews...")
        if df.empty:
            LiveLogger.log("❌ Error: Source DataFrame is empty.")
            return {"status": "error", "message": "Input DataFrame is empty."}
        
        # Tổng hợp tất cả review của mỗi user thành một document duy nhất
        user_docs = df.groupby('user_id')['review'].apply(lambda reviews: ' '.join(reviews)).to_dict()
        
        existing_user_ids = set(user_embeddings.keys())
        all_user_ids = set(user_docs.keys())
        missing_user_ids = sorted(list(all_user_ids - existing_user_ids))

        if not missing_user_ids:
            LiveLogger.log("✅ All user embeddings are already present. Nothing to do.")
            return {"status": "success"}
            
        LiveLogger.log(f"   - Found {len(all_user_ids)} total users in source data.")
        LiveLogger.log(f"   - Found {len(existing_user_ids)} existing embeddings.")
        LiveLogger.log(f"   - Found {len(missing_user_ids)} users requiring new embeddings.")

        # --- 3. Generate Embeddings for Missing Users ---
        # Toàn bộ logic này giữ nguyên, vì nó hoạt động dựa trên `missing_user_ids`
        LiveLogger.log("🧠 Initializing Gemini Service and generating new embeddings...")
        gemini_service = GeminiService(
            env_prefix=config.GEMINI_ENV_PREFIX, 
            model=config.EMBEDDING_MODEL, 
            task_type=config.EMBEDDING_TASK_TYPE,
            dim=config.EMBEDDING_DIM
        )
        
        docs_to_embed = [user_docs[uid] for uid in missing_user_ids]
        new_embeddings = {}
        has_failures = False

        num_batches = (len(docs_to_embed) + config.EMBEDDING_BATCH_SIZE - 1) // config.EMBEDDING_BATCH_SIZE
        LiveLogger.start_progress(description=f"Embedding {len(missing_user_ids)} users", total=num_batches)

        for i in range(num_batches):
            start_index = i * config.EMBEDDING_BATCH_SIZE
            end_index = start_index + config.EMBEDDING_BATCH_SIZE
            
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
                LiveLogger.log(f"   - ❌ Batch #{i} failed after retries due to (simulated) error.")

            LiveLogger.update_progress(current=i + 1)

        LiveLogger.log(f"✅ Finished embedding process.")

        if len(new_embeddings) != len(missing_user_ids):
             LiveLogger.log(f"⚠️ Warning: Could not generate embeddings for all missing users. "
                           f"Successfully generated {len(new_embeddings)} out of {len(missing_user_ids)}.")

        # --- 4. Update and Save Artifacts ---
        # Logic này giữ nguyên
        LiveLogger.log(f"💾 Updating and saving embeddings artifact to '{config.EMBEDDINGS_ARTIFACT_PATH.name}'...")
        user_embeddings.update(new_embeddings)
        save_pickle(user_embeddings, config.EMBEDDINGS_ARTIFACT_PATH)
        LiveLogger.log(f"   - Total embeddings now in file: {len(user_embeddings)}")
        
        # --- 5. Return final status based on failures ---
        # Logic này giữ nguyên
        if has_failures:
            return {"status": "incomplete"}
        else:
            return {"status": "success"}

    except Exception as e:
        LiveLogger.log(f"❌ An unexpected error occurred in the embedding pipeline: {e}")
        return {"status": "error", "message": str(e)}

# THAY ĐỔI: Cập nhật khối chạy standalone
if __name__ == '__main__':
    # This block allows the script to be run directly for isolated testing.
    # It now simulates the orchestrator by first fetching data.
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
    sys.modules['src.utils.live_logger'] = type('LiveLoggerMock', (), {'LiveLogger': ConsoleLogger})
    from src.utils.live_logger import LiveLogger
    
    # Import data loader để chạy thử nghiệm
    from utils.feedbacks_data_loader import fetch_and_prepare_data
    
    print("--- Running Embedding Pipeline in Standalone Mode ---")
    try:
        print("Step 0: Fetching data for standalone test...")
        # Lấy dữ liệu bằng cách gọi data loader
        source_df = fetch_and_prepare_data(force_fetch=False)
        
        if source_df is not None and not source_df.empty:
            print("Step 1: Running embedding logic...")
            result = run_embedding_pipeline(source_df)
            print(f"\n✅ Embedding pipeline completed with status: {result['status']}")
            if result.get('message'):
                print(f"   - Message: {result['message']}")
        else:
            print("❌ Failed to fetch or prepare data. Halting standalone test.")
            
    except Exception as e:
        print(f"An error occurred during standalone execution: {e}")
        
    print("--- Standalone run finished ---")