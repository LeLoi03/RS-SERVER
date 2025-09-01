# src/utils/live_logger.py

import threading
from typing import Dict, Any, Optional # <--- Thêm Optional
import time

_pipeline_status: Dict[str, Any] = {
    "is_running": False,
    "last_run": {
        "timestamp": None,
        "status": "Not yet run",
        "logs": [],
        "similarity_model": None,
        "admin_action_required": None, # <--- Thêm trường mới
        "progress": {
            "is_active": False,
            "description": "",
            "percentage": 0,
            "current": 0,
            "total": 0
        }
    }
}
# A lock to ensure that updates to the shared state are thread-safe.
_lock = threading.Lock()

class LiveLogger:
    """
    A thread-safe logger that manages and updates a shared global status dictionary,
    including detailed logs and structured progress information for long-running tasks.
    """

    @staticmethod
    def get_status() -> Dict[str, Any]:
        """Safely gets a copy of the current pipeline status."""
        with _lock:
            return _pipeline_status.copy()

    @staticmethod
    def start_run(model_type: str):
        """Initializes the status for a new pipeline run."""
        with _lock:
            _pipeline_status["is_running"] = True
            _pipeline_status["last_run"] = {
                "timestamp": time.time(),
                "status": "running",
                "logs": ["🚀 Pipeline run initiated..."],
                "similarity_model": model_type,
                "admin_action_required": None, # <--- Reset khi bắt đầu run mới
                "progress": {
                    "is_active": False,
                    "description": "",
                    "percentage": 0,
                    "current": 0,
                    "total": 0
                }
            }

    @staticmethod
    def log(message: str):
        """
        Adds a standard log message.
        Also deactivates any active progress bar, as a new step is starting.
        """
        with _lock:
            if _pipeline_status["is_running"]:
                print(message)  # Also print to console for server-side debugging
                # Deactivate progress bar when a new standard log is added
                _pipeline_status["last_run"]["progress"]["is_active"] = False
                _pipeline_status["last_run"]["logs"].append(message)

    @staticmethod
    def end_run(final_status: str, result_dict: Dict[str, Any]):
        """Finalizes the status of a pipeline run."""
        with _lock:
            _pipeline_status["is_running"] = False
            _pipeline_status["last_run"]["status"] = final_status
            _pipeline_status["last_run"]["progress"]["is_active"] = False # Ensure progress is off

            if final_status == 'success':
                final_log = f"\n🎉 SUCCESS: Pipeline completed!"
                if result_dict.get('artifact_path'):
                    final_log += f"\n   -> Final artifact: {result_dict.get('artifact_path')}"
                _pipeline_status["last_run"]["logs"].append(final_log)
            else:
                final_log = f"\n❌ ERROR: Pipeline failed."
                if result_dict.get('error_message'):
                     final_log += f"\n   -> Reason: {result_dict.get('error_message')}"
                _pipeline_status["last_run"]["logs"].append(final_log)

    # --- Methods for Structured Progress Reporting ---

    @staticmethod
    def start_progress(description: str, total: int):
        """Activates and initializes the progress bar state."""
        with _lock:
            if _pipeline_status["is_running"]:
                print(f"--- Starting Progress: {description} (Total: {total}) ---")
                _pipeline_status["last_run"]["progress"] = {
                    "is_active": True,
                    "description": description,
                    "percentage": 0,
                    "current": 0,
                    "total": total
                }

    @staticmethod
    def update_progress(current: int):
        """Updates the progress bar's current value and percentage."""
        with _lock:
            progress_state = _pipeline_status.get("last_run", {}).get("progress", {})
            if _pipeline_status["is_running"] and progress_state.get("is_active"):
                progress_state["current"] = current
                if progress_state["total"] > 0:
                    progress_state["percentage"] = min(100, round((current / progress_state["total"]) * 100))
                else:
                    progress_state["percentage"] = 0 # Avoid division by zero
    
     # --- Thêm phương thức mới ---
    @staticmethod
    def request_admin_action(action_type: str, message: str):
        """
        Pauses the pipeline and sets a flag requiring admin intervention.
        """
        with _lock:
            if _pipeline_status["is_running"]:
                print(f"ADMIN ACTION REQUIRED: {message}")
                _pipeline_status["last_run"]["logs"].append(f"\n🛑 {message}")
                _pipeline_status["last_run"]["admin_action_required"] = action_type
                # Giữ is_running = True để FE biết pipeline đang ở trạng thái "active"
                # nhưng không thực sự chạy.