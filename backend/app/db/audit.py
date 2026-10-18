"""
backend/app/db/audit.py
=======================
SQLite-based audit logging and telemetry store for the Web Application.
Tracks request timestamps, client IPs, model selections, GPU latencies, and VRAM.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_DIR = Path(__file__).resolve().parents[3] / "logs"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "webapp_audit.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                client_ip TEXT NOT NULL,
                action TEXT NOT NULL,
                zone_id INTEGER,
                model_id TEXT,
                cloud_coverage REAL,
                latency_ms REAL,
                vram_used_mb REAL,
                details TEXT
            )
        """)
        conn.commit()


def log_audit(
    client_ip: str,
    action: str,
    zone_id: Optional[int] = None,
    model_id: Optional[str] = None,
    cloud_coverage: Optional[float] = None,
    latency_ms: Optional[float] = None,
    vram_used_mb: Optional[float] = None,
    details: Optional[str] = None,
) -> int:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO audit_logs (
                timestamp, client_ip, action, zone_id, model_id,
                cloud_coverage, latency_ms, vram_used_mb, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_str,
                client_ip,
                action,
                zone_id,
                model_id,
                cloud_coverage,
                latency_ms,
                vram_used_mb,
                details,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_summary_stats() -> Dict[str, Any]:
    with get_connection() as conn:
        total_reqs = conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0]
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM audit_logs WHERE latency_ms IS NOT NULL"
        ).fetchone()[0] or 0.0

        model_counts = conn.execute(
            """
            SELECT model_id, COUNT(*) as cnt FROM audit_logs 
            WHERE model_id IS NOT NULL 
            GROUP BY model_id 
            ORDER BY cnt DESC
            """
        ).fetchall()

        recent_ips = conn.execute(
            "SELECT COUNT(DISTINCT client_ip) FROM audit_logs"
        ).fetchone()[0]

        return {
            "total_requests": total_reqs,
            "avg_latency_ms": round(float(avg_latency), 2),
            "unique_clients": recent_ips,
            "popular_models": [{"model_id": r["model_id"], "count": r["cnt"]} for r in model_counts],
        }


# Initialize on import
init_db()
