# ============================================================
# 🔥 KAGGLE NOTEBOOK: 01 — MAE Self-Supervised Pretraining
# GISxKMUTNB | ViTPalm — Multi-Modal Satellite Palm Detection
# ============================================================
# วิธีใช้:
#   1. Upload โฟลเดอร์ lib1_fusion/ และ lib2_vitpalm/ ขึ้น Kaggle Dataset
#   2. Copy code นี้ไปวางใน Kaggle Notebook (GPU T4 x2 หรือ P100)
#   3. Run ทุก cell ตามลำดับ
#   4. Download output: checkpoints/pretrained/mae_pretrained.pth
# ============================================================

# %% [markdown]
# # 🛰️ MAE Self-Supervised Pretraining — ViTPalm
# **Input**: 12-band satellite tiles (12, 256, 256)
# **Task**: Masked Autoencoder — reconstruct masked patches from visible ones
# **Output**: Pretrained ViT encoder weights → `mae_pretrained.pth`

# %% [code] — Cell 1: Environment Setup
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Kaggle paths ──────────────────────────────────────────────
KAGGLE_INPUT   = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR     = KAGGLE_WORKING / "checkpoints" / "pretrained"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Add lib path (upload lib1_fusion/ + lib2_vitpalm/ as Kaggle Dataset) ──
LIB_PATH = KAGGLE_INPUT / "gisxkmutnb-libs"   # ← ชื่อ Kaggle Dataset ที่ upload ขึ้นไป
sys.path.insert(0, str(LIB_PATH))

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"GPU device      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device    : {DEVICE}")

# %% [code] — Cell 2: Config
CFG = {
    # Data
    "data_dir"     : str(KAGGLE_INPUT / "gisxkmutnb-tiles"),  # Kaggle Dataset ของ processed tiles
    "num_bands"    : 12,
    "tile_size"    : 256,

    # MAE
    "mask_ratio"   : 0.75,     # mask 75% of patches (MAE default)
    "patch_size"   : 16,       # ViT patch size (16×16 px)

    # ViT Backbone
    "embed_dim"    : 512,
    "depth"        : 6,        # number of transformer blocks
    "num_heads"    : 8,
    "mlp_ratio"    : 4.0,

    # Training
    "epochs"       : 100,
    "batch_size"   : 16,
    "lr"           : 1.5e-4,
    "weight_decay" : 0.05,
    "warmup_epochs": 10,

    # Checkpoint
    "save_every"   : 10,       # save checkpoint every N epochs
    "output_dir"   : str(OUTPUT_DIR),
}
print("Config loaded:", CFG)

# %% [code] — Cell 3: Dataset
class SatelliteTileDataset(Dataset):
    """
    Loads preprocessed satellite tiles from disk.
    Expected file format: .npy arrays of shape (12, 256, 256), float32, [0.0, 1.0]
    """

    def __init__(self, data_dir: str):
        self.files = sorted(Path(data_dir).glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
        print(f"Found {len(self.files)} tiles in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tile = np.load(self.files[idx])  # (12, 256, 256)
        return torch.from_numpy(tile).float()


# %% [code] — Cell 4: MAE Model (inline — no lib2 dependency for pretraining)
class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embedding dim."""

    def __init__(self, in_channels=12, patch_size=16, img_size=256, embed_dim=512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)                    # (B, D, h, w)
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return x


class MAEEncoder(nn.Module):
    """Lightweight ViT encoder for MAE pretraining."""

    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(
            in_channels=cfg["num_bands"],
            patch_size=cfg["patch_size"],
            img_size=cfg["tile_size"],
            embed_dim=cfg["embed_dim"],
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg["embed_dim"]))
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, cfg["embed_dim"]))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg["embed_dim"],
            nhead=cfg["num_heads"],
            dim_feedforward=int(cfg["embed_dim"] * cfg["mlp_ratio"]),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg["depth"])
        self.norm = nn.LayerNorm(cfg["embed_dim"])

    def forward(self, x, mask_indices=None):
        B = x.size(0)
        tokens = self.patch_embed(x)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)              # (B, N+1, D)
        tokens = tokens + self.pos_embed

        if mask_indices is not None:
            # Keep only visible (unmasked) tokens
            visible = []
            for b in range(B):
                keep = tokens[b, mask_indices[b]]
                visible.append(keep)
            tokens = torch.stack(visible)

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens                                          # (B, N_visible, D)


class MAEDecoder(nn.Module):
    """Simple linear decoder to reconstruct masked patches."""

    def __init__(self, embed_dim, patch_size, num_bands):
        super().__init__()
        self.pred = nn.Linear(embed_dim, patch_size * patch_size * num_bands)

    def forward(self, x):
        return self.pred(x)   # (B, N_masked, patch_size^2 * num_bands)


# %% [code] — Cell 5: MAE random masking helper
def random_masking(B, N, mask_ratio, device):
    """
    Returns:
        ids_keep   : (B, N_visible) — indices of visible patches
        ids_masked : (B, N_masked)  — indices of masked patches
    """
    N_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep    = ids_shuffle[:, :N_keep]
    ids_masked  = ids_shuffle[:, N_keep:]
    return ids_keep, ids_masked


# %% [code] — Cell 6: Training Loop
def train_mae(cfg, device):
    # Data
    dataset    = SatelliteTileDataset(cfg["data_dir"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)

    # Model
    encoder = MAEEncoder(cfg).to(device)
    N_patches = encoder.patch_embed.num_patches
    decoder = MAEDecoder(cfg["embed_dim"], cfg["patch_size"], cfg["num_bands"]).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    mask_ratio = cfg["mask_ratio"]

    print(f"\n🚀 Starting MAE pretraining for {cfg['epochs']} epochs...")
    print(f"   Tiles: {len(dataset)} | Batch: {cfg['batch_size']} | Mask ratio: {mask_ratio}")

    for epoch in range(1, cfg["epochs"] + 1):
        encoder.train(); decoder.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)    # (B, 12, 256, 256)
            B = batch.size(0)

            # Random masking
            ids_keep, ids_masked = random_masking(B, N_patches, mask_ratio, device)

            # Patch targets (ground truth for masked patches)
            patches = encoder.patch_embed.proj(batch)               # (B, D, h, w)
            patches = patches.flatten(2).transpose(1, 2)            # (B, N, D)

            # Encode visible patches
            visible_tokens = encoder(batch, mask_indices=ids_keep)  # (B, N_vis, D)

            # Decode → predict masked patches
            pred = decoder(visible_tokens[:, 1:, :])                # skip cls token

            # MSE loss on masked patch pixel values
            # Gather target pixels for masked patches
            target_list = []
            for b in range(B):
                target_list.append(patches[b, ids_masked[b]])
            targets = torch.stack(target_list)  # (B, N_masked, D)

            loss = nn.functional.mse_loss(pred, targets.detach())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch:3d}/{cfg['epochs']}] | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            ckpt_path = Path(cfg["output_dir"]) / f"mae_epoch_{epoch:03d}.pth"
            torch.save({
                "epoch"       : epoch,
                "encoder"     : encoder.state_dict(),
                "decoder"     : decoder.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "loss"        : avg_loss,
                "config"      : cfg,
            }, ckpt_path)
            print(f"   ✅ Checkpoint saved → {ckpt_path.name}")

    # Save final pretrained encoder
    final_path = Path(cfg["output_dir"]) / "mae_pretrained.pth"
    torch.save({"encoder": encoder.state_dict(), "config": cfg}, final_path)
    print(f"\n🎉 Pretraining complete! Final weights: {final_path}")
    return encoder


# %% [code] — Cell 7: Run Training
encoder = train_mae(CFG, DEVICE)

# %% [code] — Cell 8: Verify output files
import os
print("\n📦 Output files in", CFG["output_dir"])
for f in sorted(os.listdir(CFG["output_dir"])):
    size_mb = os.path.getsize(os.path.join(CFG["output_dir"], f)) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")

print("\n✅ Done! Download 'mae_pretrained.pth' from Kaggle Output panel.")
print("   Then run notebooks/local/load_kaggle_results.py to save locally.")
