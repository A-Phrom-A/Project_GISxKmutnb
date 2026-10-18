"""
lib2_vitpalm.cli
================
Command-Line Interface for the GISxKMUTNB Deep Learning Library.
Provides commands for managing pretrained weights, checking model registries,
and inspecting system runtime capabilities.
"""

import sys
import argparse
from pathlib import Path

# Ensure package root is in sys.path when executed directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from lib2_vitpalm import __version__
from lib2_vitpalm.hub import (
    MODEL_REGISTRY,
    get_checkpoint_path,
    verify_checkpoint,
    calculate_sha256,
)


def cmd_info(args):
    """Prints environment, hardware, and library status."""
    print(f"=== GISxKMUTNB Library Info (v{__version__}) ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    cache_dir = Path.home() / ".cache" / "gis_kmutnb" / "checkpoints"
    print(f"Default Weights Cache: {cache_dir.resolve()}")


def cmd_list_models(args):
    """Lists registered models in the hub and their local status."""
    print("=== Registered Pretrained Models ===")
    for mid, info in MODEL_REGISTRY.items():
        print(f"\nModel ID: {mid}")
        print(f"  Description : {info['description']}")
        print(f"  Default Path: {info['default_rel_path']}")
        if "release_url" in info:
            print(f"  Release URL : {info['release_url']}")
        
        try:
            path = get_checkpoint_path(mid)
            valid = verify_checkpoint(path, info.get("sha256"))
            print(f"  Status      : FOUND ({path})")
            print(f"  Integrity   : {'VALID (SHA-256 match)' if valid else 'CHECKSUM MISMATCH'}")
        except FileNotFoundError:
            print("  Status      : NOT FOUND (Run 'gis-kmutnb download-weights --model " + mid + "')")


def cmd_verify_weights(args):
    """Verifies SHA256 integrity of model checkpoints."""
    mid = args.model
    if mid not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{mid}'. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    info = MODEL_REGISTRY[mid]
    try:
        path = get_checkpoint_path(mid, custom_path=args.path)
        print(f"Checking '{mid}' at: {path}")
        if not info.get("sha256"):
            print("Notice: No reference SHA256 registered for this checkpoint.")
            return

        computed = calculate_sha256(path)
        expected = info["sha256"].lower()
        if computed == expected:
            print(f"SUCCESS: SHA256 verified! ({computed})")
        else:
            print(f"FAILED: Checksum mismatch!\nExpected: {expected}\nComputed: {computed}")
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_download_weights(args):
    """Downloads model weights from GitHub Releases."""
    # Delegate to download script logic
    import urllib.request
    from tqdm import tqdm

    mid = args.model
    if mid not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{mid}'. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    info = MODEL_REGISTRY[mid]
    url = args.url or info.get("release_url")
    if not url:
        print(f"Error: No download URL configured for '{mid}'.")
        sys.exit(1)

    # Determine destination
    dest = Path(args.dest) if args.dest else Path(info["default_rel_path"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {mid} from:\n  {url}\nDestination:\n  {dest.resolve()}")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
            urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)
        print(f"\nDownload completed: {dest}")

        # Verification
        if info.get("sha256"):
            valid = verify_checkpoint(dest, info["sha256"])
            if valid:
                print("Integrity verification: PASSED (SHA256 Match)")
            else:
                print("WARNING: Downloaded file checksum does not match official registry!")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="gis-kmutnb",
        description="GISxKMUTNB CLI: Deep Learning & GeoAI Management Utility",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: info
    p_info = subparsers.add_parser("info", help="Show system environment and GPU status")
    p_info.set_defaults(func=cmd_info)

    # Command: list-models
    p_list = subparsers.add_parser("list-models", help="List all registered models and checkpoint status")
    p_list.set_defaults(func=cmd_list_models)

    # Command: verify-weights
    p_verify = subparsers.add_parser("verify-weights", help="Verify SHA256 checksum of model weights")
    p_verify.add_argument("--model", type=str, default="cloud_mae_500e", help="Model ID to verify")
    p_verify.add_argument("--path", type=str, default=None, help="Custom checkpoint path to verify")
    p_verify.set_defaults(func=cmd_verify_weights)

    # Command: download-weights
    p_dl = subparsers.add_parser("download-weights", help="Download model weights from GitHub Releases")
    p_dl.add_argument("--model", type=str, default="cloud_mae_500e", help="Model ID to download")
    p_dl.add_argument("--dest", type=str, default=None, help="Custom destination file path")
    p_dl.add_argument("--url", type=str, default=None, help="Override release download URL")
    p_dl.set_defaults(func=cmd_download_weights)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
