"""
scripts/cleanup_scratch_outputs.py
==================================
Purges temporary scratch and preview images older than N days (default: 3 days).
Can be run standalone from CLI or imported as a background task in FastAPI.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_DIR = ROOT_DIR / "outputs" / "scratch_previews"
PRESERVE_FILENAMES = {"readme.md", ".gitkeep", ".gitignore"}


def purge_old_scratch_files(
    target_dir: Path | str = DEFAULT_TARGET_DIR,
    max_age_days: float = 3.0,
    dry_run: bool = False,
    include_root_scratch: bool = True,
) -> Dict[str, Any]:
    """
    Scans the target directory and deletes image/scratch files older than `max_age_days`.
    
    Args:
        target_dir: Directory containing scratch outputs.
        max_age_days: Retention threshold in days (e.g. 3.0 = 72 hours).
        dry_run: If True, only reports without deleting files.
        include_root_scratch: Also checks project root for accidental scratch_*.png files.

    Returns:
        Dict containing stats of scanned, deleted, and preserved files.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    now = time.time()
    cutoff_seconds = max_age_days * 86400.0

    files_to_check: List[Path] = []

    # 1. Gather all files in target_dir
    for item in target_path.iterdir():
        if item.is_file() and item.name.lower() not in PRESERVE_FILENAMES:
            files_to_check.append(item)

    # 2. Also check for stray scratch_*.png in project root
    if include_root_scratch and target_path != ROOT_DIR:
        for stray in ROOT_DIR.glob("scratch_*.png"):
            if stray.is_file():
                files_to_check.append(stray)

    deleted_files: List[str] = []
    preserved_files: List[str] = []
    bytes_freed: int = 0

    for fpath in files_to_check:
        try:
            mtime = fpath.stat().st_mtime
            age_seconds = now - mtime
            age_days = age_seconds / 86400.0
            file_size = fpath.stat().st_size

            if age_seconds >= cutoff_seconds:
                if not dry_run:
                    fpath.unlink()
                deleted_files.append(f"{fpath.name} (age: {age_days:.2f}d, size: {file_size/1024:.1f}KB)")
                bytes_freed += file_size
            else:
                preserved_files.append(f"{fpath.name} (age: {age_days:.2f}d)")
        except Exception as e:
            print(f"[!] Warning: Failed processing {fpath.name}: {e}")

    summary = {
        "status": "success",
        "dry_run": dry_run,
        "max_age_days": max_age_days,
        "total_scanned": len(files_to_check),
        "deleted_count": len(deleted_files),
        "preserved_count": len(preserved_files),
        "bytes_freed": bytes_freed,
        "bytes_freed_mb": round(bytes_freed / (1024 * 1024), 2),
        "deleted_files": deleted_files,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Purge scratch preview files older than N days.")
    parser.add_argument(
        "--days",
        type=float,
        default=3.0,
        help="Maximum file age in days before deletion (default: 3.0 = 72 hours)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DEFAULT_TARGET_DIR),
        help=f"Target directory to clean (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate cleanup without deleting any files",
    )

    args = parser.parse_args()

    print("==================================================================")
    print("  🧹 GISxKMUTNB Scratch & Output Storage Auto-Cleaner")
    print(f"  Target Dir : {args.dir}")
    print(f"  Max Age    : {args.days} days ({args.days * 24:.0f} hours)")
    print(f"  Dry Run    : {args.dry_run}")
    print("==================================================================")

    res = purge_old_scratch_files(
        target_dir=args.dir,
        max_age_days=args.days,
        dry_run=args.dry_run,
    )

    print(f"[*] Total Files Scanned : {res['total_scanned']}")
    print(f"[*] Files Preserved     : {res['preserved_count']} (within {args.days} days)")
    print(f"[*] Files Purged        : {res['deleted_count']} (older than {args.days} days)")
    print(f"[*] Disk Space Freed    : {res['bytes_freed_mb']} MB")

    if res["deleted_files"]:
        print("\nPurged Files List:")
        for df in res["deleted_files"]:
            print(f"  - {df}")
    print("==================================================================")


if __name__ == "__main__":
    main()
