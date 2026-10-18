"""
scripts/run_server.py
=====================
Starts the FastAPI Web Application & PyTorch GPU Engine.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print("==================================================================")
    print("  [*] STARTING GISxKMUTNB - TRUE MIRACLE GEOAI PLATFORM")
    print(f"  Local URL: http://localhost:{port}")
    print(f"  Admin URL: http://localhost:{port}/admin")
    print(f"  API Docs : http://localhost:{port}/docs")
    print("==================================================================")

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        workers=1,
        app_dir=str(ROOT_DIR),
    )


