# src/api/routers/config.py

from fastapi import APIRouter, HTTPException, status
import json
from src.api.models import PipelineConfig
from src.api.lifespan import scheduler
import config.config as config

router = APIRouter()

# Define the path to the configuration file
PIPELINE_CONFIG_PATH = config.BASE_DIR / "config" / "pipeline_config.json"

@router.get("/config", response_model=PipelineConfig)
def get_pipeline_config():
    """
    Lấy cấu hình pipeline hiện tại từ tệp JSON.
    """
    try:
        with open(PIPELINE_CONFIG_PATH, 'r') as f:
            config_data = json.load(f)
        return config_data
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tệp cấu hình pipeline không được tìm thấy.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lỗi khi đọc tệp cấu hình pipeline.")

@router.post("/config", status_code=status.HTTP_200_OK)
def update_pipeline_config(new_config: PipelineConfig):
    """
    Cập nhật cấu hình pipeline và lưu vào tệp JSON.
    Đồng thời bật/tắt scheduler dựa trên cờ SCHEDULER_ENABLED.
    """
    try:
        # Chuyển đổi Pydantic model thành dict, sử dụng alias để giữ nguyên key format
        config_dict = new_config.dict(by_alias=True)

        # Ghi đè cấu hình mới vào tệp JSON
        with open(PIPELINE_CONFIG_PATH, 'w') as f:
            json.dump(config_dict, f, indent=4)

        # Xử lý việc bật/tắt scheduler
        if new_config.scheduler_enabled and not scheduler.running:
            scheduler.start(paused=False)
            message_suffix = "Scheduler đã được khởi động."
        elif not new_config.scheduler_enabled and scheduler.running:
            scheduler.shutdown()
            message_suffix = "Scheduler đã được tắt."
        else:
            scheduler_status = "đang chạy" if scheduler.running else "đã tắt"
            message_suffix = f"Trạng thái scheduler không đổi ({scheduler_status})."

        return {"message": f"Cập nhật cấu hình pipeline thành công. {message_suffix}"}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Không thể cập nhật cấu hình: {str(e)}")
