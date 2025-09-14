# src/api/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.asyncio import AsyncIOScheduler
# THAY ĐỔI: Import có chọn lọc hơn
from config import config as static_config
from config.config import get_pipeline_config
import json
from src.api.dependencies import load_model_data, model_data
from src.api.routers.pipeline import pipeline_runner_task
from src.api.models import PipelineRunRequest

# Khởi tạo scheduler để có thể import từ các module khác, sử dụng timezone UTC để nhất quán
scheduler = AsyncIOScheduler(timezone="UTC")

def scheduled_pipeline_run():
    """Hàm được gọi bởi scheduler để chạy pipeline hàng đêm."""
    from src.utils.live_logger import LiveLogger
    print("⏰ Scheduler triggered: Attempting to run nightly pipeline...")
    status = LiveLogger.get_status()
    if status["is_running"]:
        print("⏰ Scheduler skipped: A pipeline process is already running.")
        return

    request = PipelineRunRequest(
        similarity_model='mutifactor',
        force_rerun=True,
        steps_to_run=None
    )
    pipeline_runner_task(request)
    print("⏰ Scheduler job started: Full forced pipeline rerun initiated.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý các sự kiện khởi động và tắt của ứng dụng."""
    print("--- 🚀 API Server is starting up... ---")

    # THAY ĐỔI: Tải cấu hình động ngay khi khởi động
    pipeline_config = get_pipeline_config()

    # Tải model và kiểm tra, nếu thất bại thì kích hoạt pipeline chạy nền
    model_loaded_successfully = load_model_data()
    if not model_loaded_successfully:
        print("🔴 Model artifact not found. Triggering initial pipeline run in the background.")
        background_tasks = BackgroundTasks()
        initial_run_request = PipelineRunRequest(
            similarity_model='mutifactor',
            force_rerun=True,
            steps_to_run=None
        )
        background_tasks.add_task(pipeline_runner_task, initial_run_request)
        app.state.initial_pipeline_tasks = background_tasks

    # --- KHỞI TẠO SCHEDULER DỰA TRÊN CẤU HÌNH MỚI NHẤT ---
    # THAY ĐỔI: Sử dụng `pipeline_config` thay vì `config.PIPELINE_CONFIG`
    if pipeline_config['SCHEDULER_ENABLED']:
        print("📅 Initializing scheduler for nightly tasks...")

        # 1. Đọc cấu hình từ file JSON, nếu không có thì dùng giá trị mặc định
        run_hour = 1    # Giá trị mặc định
        run_minute = 0  # Giá trị mặc định

        try:
            # Sử dụng đường dẫn tĩnh từ static_config
            if static_config.SCHEDULER_CONFIG_PATH.exists():
                with open(static_config.SCHEDULER_CONFIG_PATH, 'r') as f:
                    saved_config = json.load(f)
                    run_hour = saved_config.get('hour', run_hour)
                    run_minute = saved_config.get('minute', run_minute)
                print(f"   - Loaded schedule from '{static_config.SCHEDULER_CONFIG_PATH.name}': {run_hour:02d}:{run_minute:02d}")
            else:
                # Nếu file không tồn tại, tạo nó với giá trị mặc định
                print(f"   - No schedule config file found. Creating with default time: {run_hour:02d}:{run_minute:02d}")
                with open(static_config.SCHEDULER_CONFIG_PATH, 'w') as f:
                    json.dump({'hour': run_hour, 'minute': run_minute}, f, indent=4)
        except Exception as e:
            print(f"   - ⚠️ Warning: Could not read/create scheduler config file. Using default. Error: {e}")

        # 2. Thêm job với thời gian đã được xác định
        scheduler.add_job(
            scheduled_pipeline_run, 'cron',
            hour=run_hour,
            minute=run_minute,
            id="nightly_pipeline_run",
            name="Nightly Full Pipeline Rerun",
            replace_existing=True
        )
        scheduler.start()
        run_time = f"{run_hour:02d}:{run_minute:02d}"
        print(f"✅ Scheduler started. Nightly job scheduled for {run_time} UTC.")
    else:
        print("⚪ Scheduler is disabled by configuration. Skipping.")

    print("="*50)
    yield # Server bắt đầu chạy tại đây
    print("\n" + "="*50)
    print("--- 🌙 Shutting down API Server... ---")

    # Không cần đọc lại config khi tắt, chỉ cần kiểm tra xem nó có đang chạy không
    if scheduler.running:
        print("📅 Shutting down scheduler...")
        scheduler.shutdown()

    model_data.clear()
    print("✅ Model data cleared. Server shutdown complete.")
