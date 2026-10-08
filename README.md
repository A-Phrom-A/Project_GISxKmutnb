# GISxKMUTNB — Multi-Modal Satellite Palm Detection

> Self-Supervised + Fine-Tuning ViT pipeline for oil palm detection  
> using Sentinel-1 (SAR), Sentinel-2 (Optical), and GEO-Satellite data

---

## Project Structure

```
Project_GISxKmutnb/
├── lib1_fusion/        # Library 1: Multi-Sensor Data Fusion (S1 + S2 + GEO-Sat)
├── lib2_vitpalm/       # Library 2: ViTPalm Model (Self-Supervised + Fine-Tuning)
├── poc/                # Proof of Concept end-to-end pipeline
├── data/
│   ├── raw/            # Raw satellite data (tracked via Git LFS)
│   └── processed/      # Fused tensors (12, H, W) from lib1_fusion (Git LFS)
├── notebooks/          # Exploration notebooks
├── tests/              # Unit tests
├── .gitignore
├── .gitattributes      # Git LFS config for large files
└── README.md
```

---

## Libraries

### `lib1_fusion` — Multi-Sensor Data Fusion
Converts S1 (SAR: VV, VH), S2 (Optical: B2,B3,B4,B8,B11,B12,RedEdge), and GEO-Sat (R,G,B)
into a stacked tensor of shape `(12, H, W)`.

**Fusion Strategies:**
- Early Fusion (Stack channels)
- Late Fusion (feature-level merge)
- Cross-Modal Attention

### `lib2_vitpalm` — ViTPalm Model
Vision Transformer that outputs a **probability map** `(H, W) ∈ [0.0, 1.0]`
indicating palm coverage probability per pixel.

**Supports:**
- Self-Supervised Pretraining (MAE / masked band prediction)
- Fine-Tuning with BCE + Dice Loss

---

## Proof of Concept (PoC)

| Step | Description |
|------|-------------|
| 1 | Data preparation via `lib1_fusion` |
| 2 | Data Augmentation (flip, rotate, spectral jitter) |
| 3 | Build ViTPalm model + loss via `lib2_vitpalm` |
| 4 | Train model |
| 5 | Evaluation (IoU, F1-Score, Precision, Recall) |

---

## Git Milestones

| Tag | Milestone |
|-----|-----------|
| `v0.1.0` | lib1_fusion — 3 fusion strategies complete |
| `v0.2.0` | lib2_vitpalm — backbone + palm head |
| `v0.3.0` | Self-Supervised pretraining complete |
| `v0.4.0` | Fine-tuning + PoC pipeline complete |
| `v1.0.0` | Evaluation passed + ready to publish |

---

## Requirements

- Python 3.12+
- PyTorch 2.2+ (CUDA 12.x)
- rasterio, albumentations, wandb
- See `requirements.txt` for full list
