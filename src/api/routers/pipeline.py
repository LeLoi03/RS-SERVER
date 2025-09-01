# api/routers/pipeline.py

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from src.api.models import PipelineRunRequest
from src.api.dependencies import load_model_data
from src.utils.live_logger import LiveLogger
from src.pipeline_orchestrator import run_full_pipeline

router = APIRouter()

def pipeline_runner_task(request: PipelineRunRequest):
    """The actual function that runs the pipeline in a background thread."""
    status = LiveLogger.get_status()
    if not status["is_running"] or not status["last_run"].get("admin_action_required"):
         LiveLogger.start_run(request.similarity_model) 
    
    result = run_full_pipeline(
        model_type=request.similarity_model,
        force_rerun=request.force_rerun,
        steps_to_run=request.steps_to_run
    )
    
    if result.get("status") != "paused":
        LiveLogger.end_run(result["status"], result)
        
        if result["status"] == "success":
            print("Pipeline finished successfully. Reloading model data...")
            load_model_data()
        else:
            print(f"Pipeline finished with error: {result.get('error_message')}")

@router.post("/run-pipeline", status_code=202)
def trigger_pipeline_run(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Triggers a model rebuild in the background."""
    status = LiveLogger.get_status()
    
    is_awaiting_action = status["last_run"].get("admin_action_required") is not None
    if status["is_running"] and not is_awaiting_action:
        raise HTTPException(status_code=409, detail="A pipeline process is already running.")
    
    background_tasks.add_task(pipeline_runner_task, request)
    
    if is_awaiting_action:
        return {"message": "Admin action received. Resuming pipeline execution..."}
    else:
        return {"message": "Pipeline execution started. Check /pipeline-status for progress."}
    
@router.get("/pipeline-status")
def get_pipeline_status() -> Dict[str, Any]:
    """Returns the current status of the model pipeline from the LiveLogger."""
    return LiveLogger.get_status()

@router.post("/reload-model", status_code=200)
def reload_model():
    """Manually reloads the model from the latest artifact file."""
    if load_model_data():
        return {"message": "Model reloaded successfully."}
    raise HTTPException(status_code=500, detail="Failed to reload model. Artifact may be missing or corrupted.")