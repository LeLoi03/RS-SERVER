# api/models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class PredictionRequest(BaseModel):
    user_id: str
    conference_ids: List[str]

class PipelineRunRequest(BaseModel):
    similarity_model: str = 'mutifactor'
    force_rerun: bool = False
    steps_to_run: Optional[List[str]] = None

class ArtifactSummary(BaseModel):
    artifact_name: str
    exists: bool
    data_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None




# --- THÊM MỚI: Models cho User Details ---
class ClusterInfo(BaseModel):
    cluster_id: int
    cluster_size: int

class NeighborInfo(BaseModel):
    user_id: str
    similarity_score: float

class PredictionInfo(BaseModel):
    conference_id: str
    predicted_rating: float

class UserDetailsResponse(BaseModel):
    user_id: str
    model_type: str
    is_found: bool
    cluster_info: Optional[ClusterInfo] = None
    has_embedding: bool
    similarity_info: Optional[List[NeighborInfo]] = None
    top_predictions: Optional[List[PredictionInfo]] = None
    bottom_predictions: Optional[List[PredictionInfo]] = None
    error: Optional[str] = None


# --- THÊM MỚI: Models cho Scheduler ---
class SchedulerJob(BaseModel):
    id: str
    name: str
    next_run_time: str
    cron_trigger: str

class SchedulerStatusResponse(BaseModel):
    is_running: bool
    jobs: list[SchedulerJob]

class UpdateSchedulerRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    minute: int = Field(..., ge=0, le=59, description="Minute of the hour (0-59)")