# config/config.py

from pathlib import Path
import os
from dotenv import load_dotenv # <--- THÊM DÒNG NÀY
load_dotenv() # <--- THÊM DÒNG NÀY

# --- DIRECTORY CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True) # Create artifacts dir if it doesn't exist

# --- DATA FILE PATHS ---
SOURCE_DATASET_PATH = DATA_DIR / "conference_reviews_dataset.csv"

# --- ARTIFACT FILE PATHS ---
CLUSTERING_ARTIFACT_PATH = ARTIFACTS_DIR / "user_clustering.pkl"
EMBEDDINGS_ARTIFACT_PATH = ARTIFACTS_DIR / "user_embeddings.pkl"

# Model-specific artifact paths
PEARSON_SIMILARITY_PATH = ARTIFACTS_DIR / "similarity_pearson.pkl"
PEARSON_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions_pearson.pkl"

MUTIFACTOR_SIMILARITY_PATH = ARTIFACTS_DIR / "similarity_mutifactor.pkl"
MUTIFACTOR_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions_mutifactor.pkl"

# --- MODEL PARAMETERS ---
# Clustering
NUM_CLUSTERS = 5

# Similarity
NUM_INFLUENCERS = 50
MUTIFACTOR_ALPHA = 0.00005 # Temporal decay factor

# Prediction
NUM_NEIGHBORS = 25

# --- GEMINI SERVICE CONFIGURATION ---
GEMINI_ENV_PREFIX = "GEMINI_API_KEY_"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 50


# --- BEHAVIORAL MODEL PARAMETERS ---
USE_BEHAVIORAL_DATA = True
BEHAVIORAL_WEIGHTS = {
    'search': 0.1,
    'view_detail': 0.2,
    'click': 0.2,
    'add_to_calendar': 0.3,
    'follow': 0.3,
    'blacklist': -0.3,
}

# The weight to give the behavioral similarity score when combining
# with the preference-based (rating/review) similarity score.
# 0.0 = 100% preference, 1.0 = 100% behavioral.
BEHAVIORAL_SIM_WEIGHT = 0.9 # e.g., 40% behavioral, 60% preference

# --- DEBUGGING & SIMULATION ---
# Set to an integer to simulate a Gemini API failure on that specific batch index (0-based).
# Set to None to disable simulation.
SIMULATE_EMBEDDING_ERROR_ON_BATCH = None # Sẽ gây lỗi ở batch thứ 3 (index 2)

# --- THÊM MỚI: SCHEDULER CONFIGURATION ---
# Công tắc chính để bật/tắt việc chạy pipeline tự động hàng đêm.
# Đặt thành False trong môi trường phát triển để tránh các lần chạy không mong muốn.
SCHEDULER_ENABLED = True

# Thời gian trong ngày để chạy tác vụ pipeline đã lên lịch (theo giờ địa phương của server).
# Sử dụng định dạng 24 giờ.
SCHEDULER_RUN_HOUR = 1   # ví dụ: 1 cho 1:00 sáng
SCHEDULER_RUN_MINUTE = 0 # ví dụ: 0 cho đúng giờ


# --- THAY ĐỔI: Chuyển từ file tĩnh sang cấu hình API và file cache ---
# SOURCE_DATASET_PATH = DATA_DIR / "conference_reviews_dataset.csv" # <-- XÓA DÒNG NÀY
CACHED_DATASET_PATH = DATA_DIR / "conference_reviews_cached.csv" # <-- THÊM DÒNG NÀY

# --- THÊM MỚI: Cấu hình API để lấy dữ liệu ---
FEEDBACKS_API_URL = os.getenv("FEEDBACKS_API_URL")
FEEDBACKS_API_KEY = os.getenv("FEEDBACKS_API_KEY")

MONGO_URI = os.getenv("MONGO_URI")
