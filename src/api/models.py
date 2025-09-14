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

# --- Models for Pipeline Configuration API ---

class BehavioralWeights(BaseModel):
    search: float
    view_detail: float
    click: float
    add_to_calendar: float
    follow: float
    blacklist: float

class PipelineConfig(BaseModel):
    num_clusters: int = Field(..., alias='NUM_CLUSTERS')
    num_influencers: int = Field(..., alias='NUM_INFLUENCERS')
    mutifactor_alpha: float = Field(..., alias='MUTIFACTOR_ALPHA')
    num_neighbors: int = Field(..., alias='NUM_NEIGHBORS')
    embedding_batch_size: int = Field(..., alias='EMBEDDING_BATCH_SIZE')
    use_behavioral_data: bool = Field(..., alias='USE_BEHAVIORAL_DATA')
    behavioral_weights: BehavioralWeights = Field(..., alias='BEHAVIORAL_WEIGHTS')
    behavioral_sim_weight: float = Field(..., alias='BEHAVIORAL_SIM_WEIGHT')
    scheduler_enabled: bool = Field(..., alias='SCHEDULER_ENABLED')
    include_search_behavior: bool = Field(..., alias='INCLUDE_SEARCH_BEHAVIOR')

    class Config:
        allow_population_by_field_name = True
