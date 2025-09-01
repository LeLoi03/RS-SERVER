# src/api/routers/scheduler.py

from fastapi import APIRouter, HTTPException
from apscheduler.triggers.cron import CronTrigger
import json  # <-- Thêm import
import config.config as config  # <-- Thêm import
from src.api.lifespan import scheduler
from src.api.models import SchedulerStatusResponse, SchedulerJob, UpdateSchedulerRequest

router = APIRouter()

@router.get("/scheduler/status", response_model=SchedulerStatusResponse)
def get_scheduler_status():
    """Lấy trạng thái hiện tại và các công việc của scheduler."""
    if not scheduler.running:
        return SchedulerStatusResponse(is_running=False, jobs=[])

    jobs_list = []
    for job in scheduler.get_jobs():
        trigger = job.trigger
        cron_expr = "N/A"
        if isinstance(trigger, CronTrigger):
            # Lấy giờ và phút từ các trường của trigger
            minute = trigger.fields[6].__str__()
            hour = trigger.fields[5].__str__()
            cron_expr = f"{minute} {hour} * * *"
        
        jobs_list.append(SchedulerJob(
            id=job.id,
            name=job.name,
            next_run_time=str(job.next_run_time),
            cron_trigger=cron_expr
        ))
    return SchedulerStatusResponse(is_running=True, jobs=jobs_list)


@router.post("/scheduler/update", status_code=200)
def update_scheduler_job(request: UpdateSchedulerRequest):
    """Cập nhật thời gian chạy của công việc pipeline và lưu cấu hình một cách bền vững."""
    job_id = "nightly_pipeline_run"
    
    if not scheduler.running:
        raise HTTPException(status_code=400, detail="Scheduler is not running.")
        
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")

    try:
        # 1. Cập nhật job đang chạy trong bộ nhớ
        scheduler.reschedule_job(job_id, trigger='cron', hour=request.hour, minute=request.minute)
        
        # 2. Ghi đè cấu hình mới vào file JSON để lưu trữ bền vững
        new_config = {'hour': request.hour, 'minute': request.minute}
        with open(config.SCHEDULER_CONFIG_PATH, 'w') as f:
            json.dump(new_config, f, indent=4)
        
        new_time = f"{request.hour:02d}:{request.minute:02d}"
        return {"message": f"Successfully rescheduled nightly pipeline to run at {new_time} daily. Configuration saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update schedule: {str(e)}")