# server.py

from fastapi import FastAPI, Request # <--- Thêm Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.lifespan import lifespan
from src.api.routers import predictions, pipeline, summary, scheduler # <-- Thêm scheduler

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Recommendation System API",
    description="An API to serve recommendations and manage the model pipeline.",
    version="1.5.0", # Version bump to reflect self-healing startup
    lifespan=lifespan
)

# --- 2. CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:1314",
    "http://localhost:8386"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THÊM MỚI: Middleware để chạy tác vụ nền từ lifespan ---
@app.middleware("http")
async def run_startup_background_tasks(request: Request, call_next):
    """
    This middleware checks if there are any background tasks scheduled during startup
    (via app.state) and runs them after the first response is sent.
    """
    response = await call_next(request)
    if hasattr(app.state, "initial_pipeline_tasks"):
        # Lấy các tác vụ và chạy chúng
        tasks = app.state.initial_pipeline_tasks
        response.background = tasks
        # Xóa khỏi state để nó chỉ chạy một lần
        delattr(app.state, "initial_pipeline_tasks")
    return response

# --- 3. Include Routers ---
app.include_router(predictions.router, tags=["Predictions"])
app.include_router(pipeline.router, tags=["Pipeline Management"])
app.include_router(summary.router, tags=["Artifact Summary"])
app.include_router(scheduler.router, tags=["Scheduler Management"]) # <-- Thêm dòng này

# --- 4. Optional: Root endpoint for basic check ---
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the Recommendation System API"}