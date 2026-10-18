"""
backend/app/api/logs.py
=======================
Endpoints for Admin Panel: Request audit logs, system performance metrics, and GPU telemetry.
"""

from fastapi import APIRouter
from backend.app.db.audit import get_recent_logs, get_summary_stats
from backend.app.core.model_manager import get_model_manager

router = APIRouter(prefix="/api/admin", tags=["Admin & Telemetry"])


@router.get("/logs")
def get_audit_trail(limit: int = 50):
    """Returns recent requests audit log entries."""
    return get_recent_logs(limit=limit)


@router.get("/stats")
def get_dashboard_telemetry():
    """Returns aggregated stats, popular models, and live GPU memory usage."""
    db_stats = get_summary_stats()
    manager = get_model_manager()
    gpu_stats = manager.get_gpu_status()

    return {
        "summary": db_stats,
        "gpu": gpu_stats,
    }
