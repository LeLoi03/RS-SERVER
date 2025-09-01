# src/utils/behavioral_data_loader.py

import pandas as pd
from pymongo import MongoClient
import config.config as config
from src.utils.live_logger import LiveLogger

def fetch_behavioral_data() -> pd.DataFrame:
    """
    Fetches and processes user behavioral data from MongoDB.

    Connects to the MongoDB database, queries two collections for interaction logs,
    transforms the data into a unified format, and returns it as a DataFrame
    aggregated by interaction count.

    Returns:
        pd.DataFrame: A DataFrame with columns ['user_id', 'conference_key', 'behavior_type', 'count'].
                      Returns an empty DataFrame if connection fails or no data is found.
    """
    LiveLogger.log(" MONGODB: Connecting to database to fetch behavioral data...")
    if not config.MONGO_URI:
        LiveLogger.log("❌ MONGO_URI is not configured in .env file. Skipping behavioral data.")
        return pd.DataFrame(columns=['user_id', 'conference_key', 'behavior_type', 'count'])

    client = None
    try:
        client = MongoClient(config.MONGO_URI)
        db = client['fit-confhub-local'] # Tên database

        # --- 1. Lấy dữ liệu từ UserConferenceLogs ---
        LiveLogger.log("   - Querying 'UserConferenceLogs' collection...")
        conf_logs_collection = db['UserConferenceLogs']
        conf_logs_cursor = conf_logs_collection.find(
            {},  # Lấy tất cả document
            {'_id': 0, 'userId': 1, 'conferenceId': 1, 'action': 1} # Chỉ lấy các trường cần thiết
        )
        df_conf = pd.DataFrame(list(conf_logs_cursor))
        LiveLogger.log(f"     - Found {len(df_conf)} logs.")

        # --- 2. Lấy dữ liệu từ UserInteractLogs (chỉ action 'search') ---
        LiveLogger.log("   - Querying 'UserInteractLogs' collection for 'search' actions...")
        interact_logs_collection = db['UserInteractLogs']
        interact_logs_cursor = interact_logs_collection.find(
            {'action': 'search'},
            {'_id': 0, 'userId': 1, 'action': 1, 'content': 1}
        )
        df_interact = pd.DataFrame(list(interact_logs_cursor))
        LiveLogger.log(f"     - Found {len(df_interact)} search logs.")

        # --- 3. Xử lý và Hợp nhất Dữ liệu ---
        LiveLogger.log("   - Processing and unifying data...")
        
        # Xử lý df_conf
        if not df_conf.empty:
            df_conf = df_conf.rename(columns={
                'userId': 'user_id',
                'conferenceId': 'conference_key',
                'action': 'behavior_type'
            })
            # Chuẩn hóa giá trị 'view detail' -> 'view_detail' để khớp với config
            df_conf['behavior_type'] = df_conf['behavior_type'].str.replace(' ', '_', regex=False)
        
        # Xử lý df_interact
        df_interact_processed = pd.DataFrame()
        if not df_interact.empty:
            # Trích xuất conference_key từ trường 'content.source'
            df_interact['conference_key'] = df_interact['content'].apply(
                lambda x: x.get('source') if isinstance(x, dict) else None
            )
            # Chỉ giữ lại các log có conference_key (loại bỏ các search theo keyword)
            df_interact.dropna(subset=['conference_key'], inplace=True)
            
            df_interact_processed = df_interact.rename(columns={
                'userId': 'user_id',
                'action': 'behavior_type'
            })[['user_id', 'conference_key', 'behavior_type']]

        # Hợp nhất hai DataFrame
        combined_df = pd.concat([df_conf, df_interact_processed], ignore_index=True)

        if combined_df.empty:
            LiveLogger.log("   - No valid behavioral interactions found after processing.")
            return pd.DataFrame(columns=['user_id', 'conference_key', 'behavior_type', 'count'])

        # --- 4. Tổng hợp để lấy 'count' ---
        LiveLogger.log("   - Aggregating interaction counts...")
        behavioral_counts = combined_df.groupby(['user_id', 'conference_key', 'behavior_type']).size().reset_index(name='count')
        
        LiveLogger.log(f"✅ Successfully processed {len(behavioral_counts)} unique user-item-behavior interactions from database.")
        return behavioral_counts

    except Exception as e:
        LiveLogger.log(f"❌ An error occurred while fetching behavioral data from MongoDB: {e}")
        return pd.DataFrame(columns=['user_id', 'conference_key', 'behavior_type', 'count'])
    finally:
        if client:
            client.close()
            LiveLogger.log("   - MongoDB connection closed.")