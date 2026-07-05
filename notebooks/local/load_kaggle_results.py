# ============================================================
# 💻 LOCAL SCRIPT: load_kaggle_results.py
# โหลด model weights จาก Kaggle → save ลงเครื่อง (CPU only)
# ============================================================
# วิธีใช้:
#   1. Download ไฟล์จาก Kaggle Output panel:
#      - mae_pretrained.pth
#      - vitpalm_finetuned.pth  (หรือ vitpalm_best.pth)
#   2. วางไฟล์ไว้ที่ ~/Downloads/ หรือกำหนด path ใน DOWNLOADED_PATH
#   3. รัน script นี้: python notebooks/local/load_kaggle_results.py
# ============================================================

import shutil
import torch
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── เปลี่ยน path นี้ให้ตรงกับที่ download มา ──
DOWNLOADED_FILES = {
    "mae_pretrained.pth"   : Path.home() / "Downloads" / "mae_pretrained.pth",
    "vitpalm_finetuned.pth": Path.home() / "Downloads" / "vitpalm_finetuned.pth",
    "vitpalm_best.pth"     : Path.home() / "Downloads" / "vitpalm_best.pth",
}

SAVE_TARGETS = {
    "mae_pretrained.pth"   : CHECKPOINT_DIR / "pretrained" / "mae_pretrained.pth",
    "vitpalm_finetuned.pth": CHECKPOINT_DIR / "finetuned"  / "vitpalm_finetuned.pth",
    "vitpalm_best.pth"     : CHECKPOINT_DIR / "finetuned"  / "vitpalm_best.pth",
}


# ─── Copy + Verify ─────────────────────────────────────────────────────────────

def load_and_verify(src: Path, dst: Path) -> bool:
    """Copy weight file and verify it loads correctly (CPU)."""
    if not src.exists():
        print(f"  ⏭️  Skipping {src.name} — file not found at {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  📥 Copied: {src.name} → {dst.relative_to(PROJECT_ROOT)}")

    # Verify: load on CPU
    try:
        ckpt = torch.load(dst, map_location="cpu")
        keys = list(ckpt.keys())
        print(f"     ✅ Verified — keys: {keys}")

        if "config" in ckpt:
            cfg = ckpt["config"]
            print(f"     📋 Config: embed_dim={cfg.get('embed_dim')}, "
                  f"depth={cfg.get('depth')}, "
                  f"patch_size={cfg.get('patch_size')}")

        if "iou" in ckpt:
            print(f"     📊 Best IoU: {ckpt['iou']:.4f} | F1: {ckpt.get('f1', 'N/A')}")

    except Exception as e:
        print(f"     ❌ Failed to load: {e}")
        return False

    return True


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  GISxKMUTNB — Load Kaggle Results to Local Machine")
    print("=" * 60)
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}\n")

    results = {}
    for name, src in DOWNLOADED_FILES.items():
        dst = SAVE_TARGETS[name]
        results[name] = load_and_verify(src, dst)

    print("\n" + "=" * 60)
    print("  Summary:")
    for name, ok in results.items():
        status = "✅ Saved" if ok else "⏭️  Skipped"
        print(f"  {status} — {name}")

    saved = [n for n, ok in results.items() if ok]
    if saved:
        print(f"\n  🎉 {len(saved)} file(s) saved to checkpoints/")
        print("  Git LFS will track these files automatically.")
        print("\n  Next steps:")
        print("    git add checkpoints/")
        print("    git commit -m 'feat: add Kaggle trained weights'")
    else:
        print("\n  ⚠️  No files were saved. Check that download paths are correct.")
