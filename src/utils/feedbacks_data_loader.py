# src/utils/data_loader.py

import pandas as pd
import requests
import io
from datetime import datetime

import config.config as config
from src.utils.live_logger import LiveLogger

def fetch_and_prepare_data(force_fetch: bool = False) -> pd.DataFrame:
    """
    Fetches feedback data from the API, processes it, and caches it locally.
    
    Args:
        force_fetch (bool): If True, always fetches from the API, ignoring the cache.
        
    Returns:
        pd.DataFrame: A DataFrame with the required columns for the pipeline.
    """
    # 1. Logic Caching: Nếu không ép buộc và file cache tồn tại, dùng cache
    if not force_fetch and config.CACHED_DATASET_PATH.exists():
        LiveLogger.log(f"✅ Using cached data from '{config.CACHED_DATASET_PATH.name}'.")
        return pd.read_csv(config.CACHED_DATASET_PATH)

    # 2. Gọi API để lấy dữ liệu mới
    LiveLogger.log(f"🔄 Fetching fresh data from API: {config.FEEDBACKS_API_URL}")
    if not config.FEEDBACKS_API_URL:
        raise ValueError("FEEDBACKS_API_URL is not configured in .env file.")

    headers = {
        "Accept": "*/*",
        # Thêm header Authorization nếu API của bạn yêu cầu
        "Authorization": f"Bearer {config.FEEDBACKS_API_KEY}"
    }

    try:
        response = requests.get(config.FEEDBACKS_API_URL, headers=headers, timeout=60) # 60s timeout
        response.raise_for_status() # Báo lỗi nếu status code là 4xx hoặc 5xx
    except requests.RequestException as e:
        LiveLogger.log(f"❌ API call failed: {e}")
        raise ConnectionError(f"Could not fetch data from API. Error: {e}")

    # 3. Xử lý và chuyển đổi dữ liệu
    LiveLogger.log("🛠️  Processing and transforming API data...")
    
    # Đọc nội dung CSV từ response vào DataFrame
    csv_data = io.StringIO(response.text)
    df = pd.read_csv(csv_data)

    # Ánh xạ tên cột từ API sang tên cột mà pipeline sử dụng
    column_mapping = {
        'creatorId': 'user_id',
        'conferenceId': 'conference_key',
        'description': 'review',
        'star': 'rating',
        'createdAt': 'timestamp'
    }
    df = df.rename(columns=column_mapping)

    # Chuyển đổi cột 'timestamp' từ chuỗi sang Unix timestamp (số nguyên)
    # Định dạng của API: 'Sun Aug 31 2025 03:22:41 GMT+0000 (Coordinated Universal Time)'
    # Chúng ta cần loại bỏ phần trong ngoặc đơn để pandas có thể hiểu
    df['timestamp'] = df['timestamp'].str.replace(r' \(.*\)', '', regex=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %Y %H:%M:%S GMT%z')
    df['timestamp'] = (df['timestamp'].astype('int64') // 10**9).astype('int32')

    # Chỉ giữ lại các cột cần thiết cho pipeline
    required_columns = ['user_id', 'conference_key', 'review', 'rating', 'timestamp']
    df = df[required_columns]
    
    # 4. Lưu vào cache để sử dụng cho các lần chạy sau
    LiveLogger.log(f"💾 Caching processed data to '{config.CACHED_DATASET_PATH.name}'...")
    config.CACHED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CACHED_DATASET_PATH, index=False)
    
    return df