"""
backend/app/main.py
===================
FastAPI Main Application Entry Point.
Mounts REST API routers, handles CORS, and serves the Vanilla CSS frontend dashboard.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

# Routers
from backend.app.api.zones import router as zones_router
from backend.app.api.inference import router as inference_router
from backend.app.api.cloud_repair import router as cloud_repair_router
from backend.app.api.upload import router as upload_router
from backend.app.api.logs import router as logs_router
from backend.app.api.research import router as research_router

# Auto-Cleaner Utility
from scripts.cleanup_scratch_outputs import purge_old_scratch_files, DEFAULT_TARGET_DIR

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"


async def periodic_scratch_cleanup_worker():
    """
    Background worker that runs continuously while the web application is active.
    Automatically purges temporary scratch outputs older than 3 days (72 hours).
    Runs every 12 hours.
    """
    while True:
        try:
            # Check every 12 hours while web app is running
            await asyncio.sleep(43200)
            res = purge_old_scratch_files(max_age_days=3.0)
            if res.get("deleted_count", 0) > 0:
                print(
                    f"[*] [Auto-Cleaner] Purged {res['deleted_count']} scratch files "
                    f"(>3 days old). Freed {res['bytes_freed_mb']} MB."
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[!] [Auto-Cleaner] Periodic cleanup error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Run initial cleanup check & launch periodic cleaner task
    try:
        init_res = purge_old_scratch_files(max_age_days=3.0)
        print(
            f"[*] [Auto-Cleaner] Startup scratch cleanup complete. "
            f"Preserved: {init_res['preserved_count']}, Purged: {init_res['deleted_count']}"
        )
    except Exception as e:
        print(f"[!] [Auto-Cleaner] Startup cleanup check error: {e}")

    cleaner_task = asyncio.create_task(periodic_scratch_cleanup_worker())
    yield
    # 2. Shutdown: Gracefully cancel worker
    cleaner_task.cancel()
    try:
        await cleaner_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="GISxKMUTNB — True Miracle GeoAI Platform",
    description="Multi-Modal Satellite Oil Palm Detection & Radar-Guided Cloud-MAE Web Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware (Allows seamless local and cloudflare tunnel / ngrok requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(zones_router)
app.include_router(inference_router)
app.include_router(cloud_repair_router)
app.include_router(upload_router)
app.include_router(logs_router)
app.include_router(research_router)


# Mount Static Frontend Directories
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=FRONTEND_DIR / "js"), name="js")


@app.get("/", include_in_schema=False)
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "GISxKMUTNB API Running. Frontend is being assembled."}


@app.get("/admin", include_in_schema=False)
def serve_admin():
    admin_file = FRONTEND_DIR / "admin.html"
    if admin_file.exists():
        return FileResponse(admin_file)
    return {"message": "Admin view not found."}


@app.get("/cloudmae", include_in_schema=False)
@app.get("/cloud-mae", include_in_schema=False)
def serve_cloudmae():
    cloudmae_file = FRONTEND_DIR / "cloud_mae.html"
    if cloudmae_file.exists():
        return FileResponse(
            cloudmae_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Cloud-MAE Presentation view not found."}


@app.get("/import", include_in_schema=False)
@app.get("/upload", include_in_schema=False)
def serve_import():
    import_file = FRONTEND_DIR / "import.html"
    if import_file.exists():
        return FileResponse(
            import_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Data Import Studio view not found."}


@app.get("/research", include_in_schema=False)
@app.get("/benchmarks", include_in_schema=False)
def serve_research():
    research_file = FRONTEND_DIR / "research.html"
    if research_file.exists():
        return FileResponse(
            research_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {"message": "Research & Benchmarks view not found."}


@app.get("/ndvi", include_in_schema=False)
@app.get("/health", include_in_schema=False)
def serve_ndvi():
    return RedirectResponse("/?mode=health", status_code=307)


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "gis-kmutnb-backend"}


@app.get("/api/admin/scratch-storage")
def get_scratch_storage_stats():
    """Returns storage usage statistics for outputs/scratch_previews."""
    target = Path(DEFAULT_TARGET_DIR)
    target.mkdir(parents=True, exist_ok=True)
    files = [
        f for f in target.iterdir()
        if f.is_file() and f.name.lower() not in {"readme.md", ".gitkeep", ".gitignore"}
    ]
    total_size = sum(f.stat().st_size for f in files)
    return {
        "directory": str(target),
        "total_files": len(files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "retention_policy_days": 3.0,
    }


@app.post("/api/admin/cleanup-scratch")
def trigger_scratch_cleanup(days: float = 3.0, dry_run: bool = False):
    """Manually triggers scratch output cleanup for files older than specified days."""
    res = purge_old_scratch_files(max_age_days=days, dry_run=dry_run)
    return res

