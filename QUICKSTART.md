# 🚀 GISxKMUTNB: Quickstart Guide
Multi-Modal Satellite Oil Palm Detection & Radar-Guided Cloud-MAE Library  
Developed at **King Mongkut's University of Technology North Bangkok (KMUTNB)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Phrom-A/Project_GISxKmutnb/blob/main/notebooks/GISxKMUTNB_Tutorial_Colab.ipynb)
[![Interactive Tutorial](https://img.shields.io/badge/Notebook-Colab%20Tutorial-orange.svg)](notebooks/GISxKMUTNB_Tutorial_Colab.ipynb)

---

## 🎓 Google Colab Interactive Tutorial (สื่อการสอนออนไลน์)
สำหรับผู้ที่ต้องการทดลองรันระบบ AI บน Google Colab (ฟรี GPU) หรือนำไปใช้เป็นสื่อการสอนในห้องเรียน:
- **Notebook:** [`notebooks/GISxKMUTNB_Tutorial_Colab.ipynb`](notebooks/GISxKMUTNB_Tutorial_Colab.ipynb)
- **Demo Dataset (~108 MB):** [`colab_demo_data/`](colab_demo_data/) (พร้อมอัปโหลดขึ้น Google Drive ใช้งานได้ทันที)

---

## 📦 1. Installation

### Option A: Local Editable Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/A-Phrom-A/Project_GISxKmutnb.git
cd Project_GISxKmutnb

# Install package in development mode
pip install -e .
```

### Option B: With Web Dependencies
```bash
pip install -e ".[web]"
```

---

## 📥 2. Pretrained Weights Setup

Download the official 500-epoch Cloud-MAE pretrained weights (25.6 MB) via script or CLI:

```bash
# Using CLI
gis-kmutnb download-weights --model cloud_mae_500e

# Or using the standalone script
python scripts/download_weights.py --model cloud_mae_500e
```

Verify SHA-256 integrity:
```bash
gis-kmutnb verify-weights --model cloud_mae_500e
```

---

## 💡 3. Quickstart Examples

### Example 1: Radar-Guided Cloud Removal with Cloud-MAE (5 lines)
```python
import torch
from lib2_vitpalm import CloudMAEModel
from lib2_vitpalm.hub import load_cloud_mae_weights

# 1. Initialize Cloud-MAE and load weights
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CloudMAEModel(s1_channels=2, s2_channels=7).to(device).eval()
load_cloud_mae_weights(model, device=device)

# 2. Input S1 (SAR: VV, VH), S2 (Optical 7 bands), and Binary Cloud Mask
s1 = torch.randn(1, 2, 256, 256).to(device)         # SAR penetrates clouds
s2_cloudy = torch.randn(1, 7, 256, 256).to(device)  # Optical with clouds
cloud_mask = (torch.rand(1, 1, 256, 256) > 0.5).float().to(device)

# 3. Reconstruct clear optical imagery guided by SAR
with torch.no_grad():
    s2_reconstructed = model.reconstruct(s1, s2_cloudy, cloud_mask)

print("Reconstructed shape:", s2_reconstructed.shape)  # (1, 7, 256, 256)
```

---

### Example 2: Multi-Modal Palm Segmentation (Tri-Stream Late Fusion)
```python
import torch
from lib2_vitpalm import ViTLateFusionModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTLateFusionModel(s1_channels=2, s2_channels=7, gsat_channels=3).to(device).eval()

# Load benchmark trained checkpoint
ckpt = torch.load("checkpoints/benchmark_latefusion_cloudmae/vitpalm_best.pth", map_location=device, weights_only=False)
state_dict = ckpt["model"] if "model" in ckpt else ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state_dict)

# Inputs: S1 (2 ch), S2 (7 ch), High-Res GSat (3 ch)
s1   = torch.randn(1, 2, 256, 256).to(device)
s2   = torch.randn(1, 7, 256, 256).to(device)
gsat = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    logits = model(s1, s2, gsat)  # or model(torch.cat([s1, s2, gsat], dim=1))
    palm_prob = torch.sigmoid(logits)  # Values in [0, 1]

print("Palm probability map:", palm_prob.shape)  # (1, 256, 256)
```

---

### Example 3: Weight Transfer Protocol (Transfer Cloud-MAE to Downstream)
```python
import torch
from lib2_vitpalm import CloudMAEModel, ViTCrossAttnModel
from lib2_vitpalm.hub import load_cloud_mae_weights

# Load pretrained Cloud-MAE
mae = CloudMAEModel(s1_channels=2, s2_channels=7)
load_cloud_mae_weights(mae)

# Transfer pretrained radar and optical encoders to Cross-Attention model
downstream_model = ViTCrossAttnModel(s1_channels=2, s2_channels=7, gsat_channels=3)
mae.transfer_to_cross_attn(downstream_model)
print("Weight transfer successful! Ready for fine-tuning.")
```

---

## 🛠️ 4. Command-Line Interface (`gis-kmutnb`)

```bash
# Display system, CUDA GPU, and cache information
gis-kmutnb info

# List all models in the official registry and check local status
gis-kmutnb list-models

# Verify local weights integrity against SHA-256 digests
gis-kmutnb verify-weights --model cloud_mae_500e

# Download model weights from release
gis-kmutnb download-weights --model cloud_mae_500e
```

---

## 📂 5. Modules Reference

| Module | Description | Key Exports |
|---|---|---|
| `lib1_fusion` | Multi-modal raster processing & tile extraction | `S1Processor`, `S2Processor`, `GSatProcessor`, `FusionTileBuilder` |
| `lib2_vitpalm` | Deep learning models, training engine & model hub | `CloudMAEModel`, `ViTLateFusionModel`, `ViTCrossAttnModel`, `Trainer` |
| `lib3_geoai` | Geospatial I/O, polygon extraction & NDVI analytics | `GeoTIFFHandler`, `VectorExporter`, `NDVIAnalyzer` |

---
**License:** MIT  
**Affiliation:** Faculty of Applied Science / Faculty of Engineering, KMUTNB
