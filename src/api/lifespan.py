# api/lifespan.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks # <--- Thêm BackgroundTasks
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import config.config as config
from src.api.dependencies import load_model_data, model_data
from src.api.routers.pipeline import pipeline_runner_task
from src.api.models import PipelineRunRequest

scheduler = AsyncIOScheduler()

def scheduled_pipeline_run():
    # ... (hàm này giữ nguyên)
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
    """Manages application startup and shutdown events."""
    print("--- 🚀 API Server is starting up... ---")
    
    # --- BẮT ĐẦU THAY ĐỔI ---
    
    # 1. Tải model và kiểm tra kết quả
    model_loaded_successfully = load_model_data()
    
    # 2. Nếu tải thất bại, chạy pipeline trong nền
    if not model_loaded_successfully:
        print("🔴 Model artifact not found. Triggering initial pipeline run in the background.")
        # Tạo một đối tượng BackgroundTasks
        background_tasks = BackgroundTasks()
        
        # Tạo request cho một lần chạy full, force
        initial_run_request = PipelineRunRequest(
            similarity_model='mutifactor', # Model mặc định
            force_rerun=True,
            steps_to_run=None
        )
        
        # Thêm tác vụ vào background
        background_tasks.add_task(pipeline_runner_task, initial_run_request)
        
        # Gán background_tasks vào app để nó được thực thi sau khi lifespan kết thúc
        # Đây là một cách để chạy tác vụ nền từ lifespan
        app.state.initial_pipeline_tasks = background_tasks
    
    # --- KẾT THÚC THAY ĐỔI ---

    # Cấu hình scheduler (giữ nguyên)
    if config.SCHEDULER_ENABLED:
        print("📅 Initializing scheduler for nightly tasks...")
        scheduler.add_job(
            scheduled_pipeline_run, 'cron',
            hour=config.SCHEDULER_RUN_HOUR,
            minute=config.SCHEDULER_RUN_MINUTE,
            id="nightly_pipeline_run",
            name="Nightly Full Pipeline Rerun",
            replace_existing=True
        )
        scheduler.start()
        run_time = f"{config.SCHEDULER_RUN_HOUR:02d}:{config.SCHEDULER_RUN_MINUTE:02d}"
        print(f"✅ Scheduler started. Nightly job scheduled for {run_time}.")
    else:
        print("⚪ Scheduler is disabled by configuration. Skipping.")
    
    print("="*50)
    yield # Server bắt đầu chạy ở đây
    print("\n" + "="*50)
    print("--- 🌙 Shutting down API Server... ---")
    
    if config.SCHEDULER_ENABLED and scheduler.running:
        print("📅 Shutting down scheduler...")
        scheduler.shutdown()
    
    model_data.clear()
    print("✅ Model data cleared. Server shutdown complete.")