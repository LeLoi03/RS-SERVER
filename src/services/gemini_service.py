import numpy as np
import os
import time
from itertools import cycle
from typing import List
from dotenv import load_dotenv
from src.utils.live_logger import LiveLogger # --- IMPORT THE LIVE LOGGER ---
import config.config as config

# Load environment variables from .env file
load_dotenv()

class KeyManager:
    """
    Quản lý, xoay vòng và áp dụng thời gian chờ (cooldown) cho các key API.
    """
    def __init__(self, env_prefix: str, cooldown_seconds: int = 90):
        """
        Khởi tạo KeyManager.

        Args:
            env_prefix (str): Tiền tố của các biến môi trường chứa API key.
            cooldown_seconds (int): Số giây mỗi key phải "nghỉ" trước khi được sử dụng lại.
        """
        self.keys = [v for k, v in os.environ.items() if k.startswith(env_prefix)]
        if not self.keys:
            raise ValueError(f"❌ No API keys found with prefix '{env_prefix}' in .env file.")
        
        self._key_cycler = cycle(self.keys)
        self.cooldown_seconds = cooldown_seconds
        
        # Sử dụng time.monotonic() vì nó phù hợp để đo khoảng thời gian trôi qua
        # và không bị ảnh hưởng bởi thay đổi giờ hệ thống.
        # Khởi tạo tất cả các key với timestamp 0.0 để chúng sẵn sàng ngay lập tức.
        self.key_last_used = {key: 0.0 for key in self.keys}
        
        LiveLogger.log(f"🔑 KeyManager initialized with {len(self.keys)} API keys and a {self.cooldown_seconds}s per-key cooldown.")

    def get_next_key(self) -> str:
        """
        Lấy key tiếp theo trong vòng lặp. Nếu key đó đang trong thời gian cooldown,
        hàm sẽ đợi cho đến khi hết cooldown rồi mới trả về key.
        """
        api_key = next(self._key_cycler)
        
        last_used_time = self.key_last_used[api_key]
        current_time = time.monotonic()
        
        elapsed_time = current_time - last_used_time
        
        if elapsed_time < self.cooldown_seconds:
            wait_time = self.cooldown_seconds - elapsed_time
            LiveLogger.log(f"   - ⏳ Key ...{api_key[-4:]} is on cooldown. Waiting for {wait_time:.1f} seconds...")
            time.sleep(wait_time)
        
        # Cập nhật thời gian sử dụng của key này là "ngay bây giờ"
        self.key_last_used[api_key] = time.monotonic()
        
        return api_key

# ==============================================================================
# --- KẾT THÚC PHẦN THAY ĐỔI ---
# ==============================================================================


class GeminiService:
    """A service class for handling Gemini API embedding operations."""
    def __init__(self, env_prefix: str, model: str, task_type: str, dim: int):
        try:
            from google import genai
            from google.genai import types
            from google.api_core import exceptions as google_exceptions
            self.genai = genai
            self.types = types
            self.google_exceptions = google_exceptions
        except ImportError:
            raise ImportError("❌ Please install google-genai: pip install google-genai")
        # --- THAY ĐỔI NHỎ Ở ĐÂY ---
        # Truyền vào thời gian cooldown, có thể lấy từ config nếu muốn
        self.key_manager = KeyManager(env_prefix, cooldown_seconds=90) 
        self.model = model
        self.task_type = task_type
        self.output_dim = dim

    @staticmethod
    def _normalize_embedding(embedding: list) -> list:
        np_embedding = np.array(embedding)
        norm = np.linalg.norm(np_embedding)
        return (np_embedding / norm).tolist() if norm != 0 else embedding

    def embed_content(self, batch_docs: List[str], batch_index: int) -> List[List[float]] | None: # <--- THÊM batch_index
        """
        Embeds a batch of documents, with an option to simulate failure for testing.
        """
        # --- START: CODE GIẢ LẬP LỖI ---
        # Kiểm tra xem cờ giả lập có được bật và có khớp với batch hiện tại không
        if config.SIMULATE_EMBEDDING_ERROR_ON_BATCH is not None and \
           batch_index == config.SIMULATE_EMBEDDING_ERROR_ON_BATCH:
            
            LiveLogger.log(f"   - 🛑 SIMULATING API FAILURE on batch #{batch_index} as configured.")
            # Ném ra một lỗi mà logic retry có thể bắt được, để giả lập lỗi 429 thật
            # Sau vài lần retry, nó sẽ thất bại hoàn toàn.
            raise self.google_exceptions.ResourceExhausted("Simulated 429 Resource Exhausted Error")
        # --- END: CODE GIẢ LẬP LỖI ---

        retries, max_retries, backoff_delay = 0, 3, 60
        while retries < max_retries:
            # --- LOGIC MỚI ĐƯỢC ÁP DỤNG NGẦM Ở ĐÂY ---
            # Lệnh này giờ sẽ tự động đợi nếu cần thiết
            api_key = self.key_manager.get_next_key() 
            client = self.genai.Client(api_key=api_key)
            try:
                result = client.models.embed_content(
                    model=self.model,
                    contents=batch_docs,
                    config=self.types.EmbedContentConfig(task_type=self.task_type, output_dimensionality=self.output_dim)
                )
                return [self._normalize_embedding(e.values) for e in result.embeddings]
            except (self.google_exceptions.InternalServerError, self.google_exceptions.ResourceExhausted, self.google_exceptions.PermissionDenied) as e:
                retries += 1
                # Nếu đây là lỗi giả lập, log sẽ hiển thị nó
                LiveLogger.log(f"   - ⚠️ Gemini API error (key ...{api_key[-4:]}): {type(e).__name__}: {e}. Retrying in {backoff_delay}s... ({retries}/{max_retries})")
                time.sleep(backoff_delay)
                backoff_delay *= 2
            except Exception as e:
                LiveLogger.log(f"   - ❌ Unexpected error during embedding with key ...{api_key[-4:]}: {e}")
                break
        LiveLogger.log(f"   - ❌ Skipping batch after failing all retry attempts.")
        return None
