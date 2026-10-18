#!/usr/bin/env python3
"""
scripts/download_weights.py
===========================
Automated weight downloader for GISxKMUTNB Pretrained Models.
Downloads model checkpoints directly from official GitHub Releases,
verifies cryptographic SHA-256 integrity, and stores them in local checkpoints/ or cache.
"""

import os
import sys
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lib2_vitpalm.hub import (
    MODEL_REGISTRY,
    verify_checkpoint,
    calculate_sha256,
)


class DownloadProgressBar(tqdm):
    """tqdm-based progress hook for urllib.request."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_single_model(model_id: str, dest: Path, force: bool = False, custom_url: str = None) -> bool:
    if model_id not in MODEL_REGISTRY:
        print(f"[-] Unknown model '{model_id}'. Available: {list(MODEL_REGISTRY.keys())}")
        return False

    meta = MODEL_REGISTRY[model_id]
    url = custom_url or meta.get("release_url")

    if not url:
        print(f"[-] Model '{model_id}' does not have a public release URL configured.")
        return False

    # Check if already present and valid
    if dest.exists() and not force:
        print(f"[*] Found existing checkpoint at: {dest}")
        if meta.get("sha256"):
            if verify_checkpoint(dest, meta["sha256"]):
                print(f"[+] Checksum verified (SHA-256 matches official registry). No download needed.")
                return True
            else:
                print(f"[!] Checksum mismatch for existing file. Re-downloading...")
        else:
            print("[+] Checkpoint exists. Skipping download (use --force to overwrite).")
            return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] Downloading '{model_id}'...")
    print(f"    Source URL : {url}")
    print(f"    Destination: {dest.resolve()}")

    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
            urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)
        print(f"\n[+] Successfully downloaded to: {dest.resolve()}")

        # Verify checksum
        if meta.get("sha256"):
            print("[*] Verifying cryptographic integrity (SHA-256)...")
            computed = calculate_sha256(dest)
            if computed == meta["sha256"].lower():
                print(f"[+] INTEGRITY VERIFIED: {computed}")
                return True
            else:
                print(f"[-] CHECKSUM MISMATCH!\n    Expected: {meta['sha256']}\n    Computed: {computed}")
                return False
        return True

    except urllib.error.HTTPError as e:
        print(f"\n[-] HTTP Error {e.code}: {e.reason}")
        if e.code == 404:
            print(
                "\n[💡 NOTE]: The GitHub Release tag or asset has not been uploaded to GitHub yet.\n"
                f"    Target URL: {url}\n"
                "    Please ensure you have created Release 'v1.0.0' on GitHub and uploaded 'cloud_mae_best.pth'.\n"
                f"    Alternatively, copy your trained checkpoint directly to:\n"
                f"    -> {dest.resolve()}"
            )
        return False
    except Exception as e:
        print(f"\n[-] Failed to download '{model_id}': {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained model checkpoints for GISxKMUTNB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cloud_mae_500e",
        help="Model ID to download ('cloud_mae_500e', 'all', etc.)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Custom destination directory or file path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if local checkpoint exists and is verified",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Override release URL",
    )

    args = parser.parse_args()

    models_to_download = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    all_success = True
    for mid in models_to_download:
        if args.dest:
            p = Path(args.dest)
            dest = p if p.suffix else p / Path(MODEL_REGISTRY[mid]["default_rel_path"]).name
        else:
            dest = ROOT_DIR / MODEL_REGISTRY[mid]["default_rel_path"]

        ok = download_single_model(mid, dest, force=args.force, custom_url=args.url)
        if not ok:
            all_success = False

    if all_success:
        print("\n[+] All checkpoint operations completed successfully.")
    else:
        print("\n[!] One or more checkpoints could not be completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
