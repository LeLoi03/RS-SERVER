# config/config.py

from pathlib import Path
import os
from dotenv import load_dotenv
import json

# Tải biến môi trường từ tệp .env
load_dotenv()

# --- CẤU HÌNH THƯ MỤC (TĨNH) ---
# Thư mục gốc của dự án (thư mục chứa thư mục 'config')
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
# Đảm bảo thư mục artifacts tồn tại
ARTIFACTS_DIR.mkdir(exist_ok=True)

# --- ĐƯỜNG DẪN TẠO PHẨM TĨNH ---
# Các đường dẫn này không thay đổi trong quá trình chạy ứng dụng
CLUSTERING_ARTIFACT_PATH = ARTIFACTS_DIR / "user_clustering.pkl"
EMBEDDINGS_ARTIFACT_PATH = ARTIFACTS_DIR / "user_embeddings.pkl"
PEARSON_SIMILARITY_PATH = ARTIFACTS_DIR / "similarity_pearson.pkl"
PEARSON_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions_pearson.pkl"
MUTIFACTOR_SIMILARITY_PATH = ARTIFACTS_DIR / "similarity_mutifactor.pkl"
MUTIFACTOR_PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions_mutifactor.pkl"
CACHED_DATASET_PATH = DATA_DIR / "conference_reviews_cached.csv"
SCHEDULER_CONFIG_PATH = ARTIFACTS_DIR / "scheduler_config.json"

# --- CẤU HÌNH PIPELINE ĐỘNG ---
# Cấu hình này có thể được thay đổi "nóng" thông qua API mà không cần khởi động lại server.
PIPELINE_CONFIG_PATH = BASE_DIR / "config" / "pipeline_config.json"

def get_pipeline_config() -> dict:
    """
    Tải và trả về cấu hình pipeline MỚI NHẤT từ tệp JSON.

    QUAN TRỌNG: Hàm này đọc trực tiếp từ tệp mỗi khi được gọi. Điều này đảm bảo
    rằng bất kỳ thay đổi nào được thực hiện thông qua API /config sẽ được áp dụng
    ngay lập tức cho lần chạy pipeline tiếp theo.

    Returns:
        dict: Một dictionary chứa các tham số cấu hình của pipeline.
    """
    try:
        # Thêm encoding='utf-8' để đảm bảo tính tương thích
        with open(PIPELINE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"CẢNH BÁO: Không thể tải cấu hình từ {PIPELINE_CONFIG_PATH}. Lỗi: {e}. Sử dụng giá trị mặc định.")
        # Fallback về giá trị mặc định nếu tệp bị thiếu hoặc lỗi
        return {
            "NUM_CLUSTERS": 5,
            "NUM_INFLUENCERS": 50,
            "MUTIFACTOR_ALPHA": 0.00005,
            "NUM_NEIGHBORS": 25,
            "EMBEDDING_BATCH_SIZE": 50,
            "USE_BEHAVIORAL_DATA": True,
            "INCLUDE_SEARCH_BEHAVIOR": True,
            "BEHAVIORAL_WEIGHTS": {
                "search": 0.1, "view_detail": 0.2, "click": 0.2,
                "add_to_calendar": 0.3, "follow": 0.3, "blacklist": -0.3
            },
            "BEHAVIORAL_SIM_WEIGHT": 0.5,
            "SCHEDULER_ENABLED": True
        }

# ==============================================================================
# THAY ĐỔI QUAN TRỌNG NHẤT
# ==============================================================================
# Dòng code dưới đây đã bị XÓA:
# PIPELINE_CONFIG = get_pipeline_config()
#
# Lý do: Việc gán vào một biến toàn cục như thế này chỉ xảy ra MỘT LẦN khi
# ứng dụng khởi động. Điều này khiến cho mọi thay đổi vào file JSON sau đó
# không được cập nhật vào biến này.
#
# Thay vào đó, các module khác trong ứng dụng sẽ phải gọi trực tiếp hàm
# `config.get_pipeline_config()` mỗi khi cần truy cập cấu hình để luôn
# lấy được phiên bản mới nhất.
# ==============================================================================


# --- CẤU HÌNH NHẠY CẢM & TĨNH (Từ biến môi trường) ---
# Các giá trị này yêu cầu khởi động lại server để thay đổi
GEMINI_ENV_PREFIX = "GEMINI_API_KEY_"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_TASK_TYPE = "SEMANTIC_SIMILARITY"
EMBEDDING_DIM = 768
FEEDBACKS_API_URL = os.getenv("FEEDBACKS_API_URL")
FEEDBACKS_API_KEY = os.getenv("FEEDBACKS_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# --- DEBUGGING (TĨNH) ---
SIMULATE_EMBEDDING_ERROR_ON_BATCH = None
