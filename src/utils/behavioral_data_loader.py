# src/utils/behavioral_data_loader.py

import json
import pandas as pd
from pymongo import MongoClient
from typing import Dict, Any
# THAY ĐỔI: Import config tĩnh để lấy MONGO_URI
from config import config as static_config
from src.utils.live_logger import LiveLogger

# THAY ĐỔI: Hàm nhận thêm tham số config_data
def fetch_behavioral_data(config_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu hành vi người dùng từ MongoDB.

    Dựa vào `config_data['INCLUDE_SEARCH_BEHAVIOR']`, hàm sẽ quyết định có
    tải và xử lý dữ liệu từ collection 'UserInteractLogs' (hành vi search) hay không.

    Args:
        config_data (Dict[str, Any]): Dictionary cấu hình của pipeline.

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu hành vi đã được xử lý.
    """
    LiveLogger.log(" MONGODB: Connecting to the database to load behavioral data...")
    if not static_config.MONGO_URI:
        LiveLogger.log("❌ MONGO_URI is not configured in the .env file. Skipping behavioral data.")
        return pd.DataFrame(columns=['user_id', 'conference_key', 'behavior_type', 'count'])

    client = None
    try:
        client = MongoClient(static_config.MONGO_URI)
        db = client['fit-confhub-local']

        # --- 1. Tải dữ liệu từ UserConferenceLogs (luôn chạy) ---
        LiveLogger.log("   - Querying 'UserConferenceLogs' collection...")
        conf_logs_collection = db['UserConferenceLogs']
        conf_logs_cursor = conf_logs_collection.find(
            {},
            {'_id': 0, 'userId': 1, 'conferenceId': 1, 'action': 1}
        )
        df_conf = pd.DataFrame(list(conf_logs_cursor))
        LiveLogger.log(f"     - Found {len(df_conf)} conference logs.")

        # Khởi tạo DataFrame cho dữ liệu search
        df_interact_processed = pd.DataFrame()

        # --- 2. Tải dữ liệu từ UserInteractLogs (có điều kiện) ---
        # THAY ĐỔI: Chỉ chạy khối này nếu config cho phép
        if config_data.get('INCLUDE_SEARCH_BEHAVIOR', True):
            LiveLogger.log("   - Querying 'UserInteractLogs' collection for 'search' actions (as enabled by config)...")
            interact_logs_collection = db['UserInteractLogs']
            interact_logs_cursor = interact_logs_collection.find(
                {'action': 'search'},
                {'_id': 0, 'userId': 1, 'action': 1, 'content': 1}
            )
            df_interact = pd.DataFrame(list(interact_logs_cursor))
            LiveLogger.log(f"     - Found {len(df_interact)} search logs.")

            # Xử lý DataFrame log tương tác (chỉ khi nó không rỗng)
            if not df_interact.empty:
                def extract_conferences(content_str):
                    try:
                        content_json = json.loads(content_str)
                        return content_json.get('conferences')
                    except (json.JSONDecodeError, TypeError):
                        return None

                df_interact['conference_list'] = df_interact['content'].apply(extract_conferences)
                df_interact.dropna(subset=['conference_list'], inplace=True)
                df_interact = df_interact.explode('conference_list')
                df_interact.rename(columns={'conference_list': 'conference_key'}, inplace=True)
                df_interact.dropna(subset=['conference_key'], inplace=True)
                df_interact_processed = df_interact.rename(columns={
                    'userId': 'user_id',
                    'action': 'behavior_type'
                })[['user_id', 'conference_key', 'behavior_type']]
        else:
            LiveLogger.log("   - Skipping 'search' actions as disabled by 'INCLUDE_SEARCH_BEHAVIOR' config.")

        # --- 3. Xử lý và Thống nhất Dữ liệu ---
        LiveLogger.log("   - Processing and unifying data...")
        if not df_conf.empty:
            df_conf = df_conf.rename(columns={
                'userId': 'user_id',
                'conferenceId': 'conference_key',
                'action': 'behavior_type'
            })
            df_conf['behavior_type'] = df_conf['behavior_type'].str.replace(' ', '_', regex=False)

        # Gộp hai DataFrame. Nếu search bị bỏ qua, df_interact_processed sẽ rỗng.
        combined_df = pd.concat([df_conf, df_interact_processed], ignore_index=True)

        if combined_df.empty:
            LiveLogger.log("   - No valid behavioral interactions found after processing.")
            return pd.DataFrame(columns=['user_id', 'conference_key', 'behavior_type', 'count'])

        # --- 4. Tổng hợp để lấy 'count' ---
        LiveLogger.log("   - Aggregating interaction counts...")
        behavioral_counts = combined_df.groupby(['user_id', 'conference_key', 'behavior_type']).size().reset_index(name='count')

        LiveLogger.log(f"✅ Successfully processed {len(behavioral_counts)} unique user-item-behavior interactions from the database.")
        return behavioral_counts

    except Exception as e:
        LiveLogger.log(f"❌ An error occurred while loading behavioral data from MongoDB: {e}")
        return pd.DataFrame(columns=['user_id', 'conference_key', 'behavior_type', 'count'])
    finally:
        if client:
            client.close()
            LiveLogger.log("   - MongoDB connection closed.")
