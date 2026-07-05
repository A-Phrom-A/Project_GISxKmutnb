# ============================================================
# 🔥 KAGGLE NOTEBOOK: 02 — Fine-tuning ViTPalm (Palm Detection)
# GISxKMUTNB | ViTPalm — Multi-Modal Satellite Palm Detection
# ============================================================
# วิธีใช้:
#   1. ต้องรัน 01_mae_pretrain.py ก่อน และได้ mae_pretrained.pth มาแล้ว
#   2. Upload mae_pretrained.pth เป็น Kaggle Dataset ชื่อ "gisxkmutnb-weights"
#   3. Run notebook นี้บน Kaggle GPU
#   4. Download output: checkpoints/finetuned/vitpalm_finetuned.pth
# ============================================================

# %% [markdown]
# # 🌴 Fine-tuning ViTPalm — Palm Detection (Binary Segmentation)
# **Input**: 12-band satellite tiles (12, 256, 256) + palm mask labels (1, 256, 256)
# **Task**: Binary segmentation → Probability Map (H, W) ∈ [0.0, 1.0]
# **Loss**: BCE + Dice
# **Output**: Fine-tuned model → `vitpalm_finetuned.pth`

# %% [code] — Cell 1: Setup
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

KAGGLE_INPUT   = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR     = KAGGLE_WORKING / "checkpoints" / "finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIB_PATH = KAGGLE_INPUT / "gisxkmutnb-libs"
sys.path.insert(0, str(LIB_PATH))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# %% [code] — Cell 2: Config
CFG = {
    # Data
    "tile_dir"      : str(KAGGLE_INPUT / "gisxkmutnb-tiles" / "tiles"),      # (12, 256, 256) .npy
    "mask_dir"      : str(KAGGLE_INPUT / "gisxkmutnb-tiles" / "masks"),      # (256, 256) binary .npy
    "num_bands"     : 12,
    "tile_size"     : 256,
    "val_split"     : 0.15,

    # Pretrained weights
    "pretrained_ckpt": str(KAGGLE_INPUT / "gisxkmutnb-weights" / "mae_pretrained.pth"),

    # ViT (must match MAE config)
    "embed_dim"     : 512,
    "depth"         : 6,
    "num_heads"     : 8,
    "mlp_ratio"     : 4.0,
    "patch_size"    : 16,

    # Training
    "epochs"        : 50,
    "batch_size"    : 8,
    "lr"            : 1e-4,
    "weight_decay"  : 1e-4,
    "bce_weight"    : 0.5,
    "dice_weight"   : 0.5,

    # Checkpoint
    "save_every"    : 5,
    "output_dir"    : str(OUTPUT_DIR),
}

# %% [code] — Cell 3: Dataset with masks
class PalmTileDataset(Dataset):
    """
    Loads paired satellite tiles + palm mask labels.
    tile:  (12, 256, 256) float32 [0.0, 1.0]
    mask:  (256, 256)     float32 binary {0.0, 1.0}
    """

    def __init__(self, tile_dir, mask_dir):
        self.tiles = sorted(Path(tile_dir).glob("*.npy"))
        self.masks = sorted(Path(mask_dir).glob("*.npy"))
        assert len(self.tiles) == len(self.masks), \
            f"Tile/mask count mismatch: {len(self.tiles)} vs {len(self.masks)}"
        print(f"Dataset: {len(self.tiles)} tile-mask pairs")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = torch.from_numpy(np.load(self.tiles[idx])).float()   # (12, 256, 256)
        mask = torch.from_numpy(np.load(self.masks[idx])).float()   # (256, 256)
        return tile, mask


# %% [code] — Cell 4: Load MAE encoder (reuse from 01_mae_pretrain.py architecture)
# ── Copy PatchEmbed and MAEEncoder classes here ─────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, in_channels=12, patch_size=16, img_size=256, embed_dim=512):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.patch_embed = PatchEmbed(cfg["num_bands"], cfg["patch_size"],
                                      cfg["tile_size"], cfg["embed_dim"])
        N = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg["embed_dim"]))
        self.pos_embed  = nn.Parameter(torch.zeros(1, N + 1, cfg["embed_dim"]))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg["embed_dim"], nhead=cfg["num_heads"],
            dim_feedforward=int(cfg["embed_dim"] * cfg["mlp_ratio"]),
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg["depth"])
        self.norm = nn.LayerNorm(cfg["embed_dim"])
        self.num_patches = N

    def forward(self, x):
        B = x.size(0)
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 1:, :]   # (B, N, D) — exclude cls token


# %% [code] — Cell 5: Decoder + Palm Head (Upsample → Probability Map)
class SegDecoder(nn.Module):
    """
    Reshapes ViT token sequence back to spatial feature map,
    then upsamples to original resolution (H, W).
    """

    def __init__(self, embed_dim, patch_size, tile_size):
        super().__init__()
        self.h = tile_size // patch_size      # spatial grid height
        self.w = tile_size // patch_size      # spatial grid width
        self.embed_dim = embed_dim

        self.up = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),  # ×2
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),         # ×4
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),          # ×8
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),           # ×16
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),   # Palm Head — single channel
        )

    def forward(self, tokens):
        # tokens: (B, N, D) → reshape to (B, D, h, w)
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, self.h, self.w)  # (B, D, h, w)
        x = self.up(x)            # (B, 1, H, W)
        x = torch.sigmoid(x)      # Probability Map ∈ [0.0, 1.0]
        return x.squeeze(1)       # (B, H, W)


# %% [code] — Cell 6: Combined ViTPalm Model
class ViTPalmModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ViTEncoder(cfg)
        self.decoder = SegDecoder(cfg["embed_dim"], cfg["patch_size"], cfg["tile_size"])

    def forward(self, x):
        tokens = self.encoder(x)       # (B, N, D)
        prob_map = self.decoder(tokens) # (B, H, W)
        return prob_map


# %% [code] — Cell 7: Loss — BCE + Dice
def dice_loss(pred, target, smooth=1.0):
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred, target, bce_w=0.5, dice_w=0.5):
    bce  = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce_w * bce + dice_w * dice


# %% [code] — Cell 8: Metrics
def compute_iou(pred_prob, target, threshold=0.5):
    pred_bin = (pred_prob > threshold).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    return (intersection / (union + 1e-6)).item()


def compute_f1(pred_prob, target, threshold=0.5):
    pred_bin = (pred_prob > threshold).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    return (2 * precision * recall / (precision + recall + 1e-6)).item()


# %% [code] — Cell 9: Fine-tuning loop
def finetune(cfg, device):
    # Dataset
    full_dataset = PalmTileDataset(cfg["tile_dir"], cfg["mask_dir"])
    val_size  = int(len(full_dataset) * cfg["val_split"])
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = ViTPalmModel(cfg).to(device)

    # Load pretrained MAE encoder weights
    ckpt = torch.load(cfg["pretrained_ckpt"], map_location=device)
    model.encoder.load_state_dict(ckpt["encoder"], strict=False)
    print(f"✅ Loaded pretrained encoder from {cfg['pretrained_ckpt']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_iou = 0.0
    print(f"\n🚀 Fine-tuning for {cfg['epochs']} epochs | Train: {train_size} | Val: {val_size}")

    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for tiles, masks in train_loader:
            tiles, masks = tiles.to(device), masks.to(device)
            pred = model(tiles)
            loss = combined_loss(pred, masks, cfg["bce_weight"], cfg["dice_weight"])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_iou, val_f1 = 0.0, 0.0
        with torch.no_grad():
            for tiles, masks in val_loader:
                tiles, masks = tiles.to(device), masks.to(device)
                pred = model(tiles)
                val_iou += compute_iou(pred, masks)
                val_f1  += compute_f1(pred, masks)

        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        avg_iou  = val_iou / len(val_loader)
        avg_f1   = val_f1  / len(val_loader)

        print(f"Epoch [{epoch:3d}/{cfg['epochs']}] Loss: {avg_loss:.4f} | "
              f"IoU: {avg_iou:.4f} | F1: {avg_f1:.4f}")

        # Save best
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_path = Path(cfg["output_dir"]) / "vitpalm_best.pth"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "iou": avg_iou, "f1": avg_f1, "config": cfg}, best_path)
            print(f"   ⭐ New best IoU={avg_iou:.4f} → saved {best_path.name}")

        if epoch % cfg["save_every"] == 0:
            ckpt_path = Path(cfg["output_dir"]) / f"vitpalm_epoch_{epoch:03d}.pth"
            torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg}, ckpt_path)

    # Save final
    final_path = Path(cfg["output_dir"]) / "vitpalm_finetuned.pth"
    torch.save({"model": model.state_dict(), "config": cfg, "best_iou": best_iou}, final_path)
    print(f"\n🎉 Fine-tuning complete! Best IoU: {best_iou:.4f}")
    print(f"   Final weights: {final_path}")
    return model


# %% [code] — Cell 10: Run
model = finetune(CFG, DEVICE)

# %% [code] — Cell 11: Check outputs
import os
print("\n📦 Output files:")
for f in sorted(os.listdir(CFG["output_dir"])):
    size_mb = os.path.getsize(os.path.join(CFG["output_dir"], f)) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")
print("\n✅ Download 'vitpalm_finetuned.pth' from Kaggle Output → run load_kaggle_results.py")
