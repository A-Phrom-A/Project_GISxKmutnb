# GISxKMUTNB — Multi-Modal Satellite Palm Detection

> **Self-Supervised Cloud-MAE + Fine-Tuning ViT Pipeline for Oil Palm Detection**  
> Integrating Sentinel-1 (SAR), Sentinel-2 (Multispectral Optical), and GEO-Satellite imagery with zero Git LFS overhead.

---

## 📌 Project Overview

This research project develops a multi-sensor vision transformer framework specifically engineered for accurate tropical oil palm canopy segmentation under pervasive cloud and atmospheric interference.

### Key Highlights
- **Cloud-MAE Pretraining:** 500 epochs of self-supervised Masked Autoencoding on 12-channel multi-modal satellite tiles, achieving an **80.5% error reduction** (Validation Loss: `0.00410` $\to$ `0.00080`).
- **Master Benchmark (E1 to E6):** Demonstrates significant performance superiority over baseline architectures:
  - **Supervised ViT Baseline (EXP-01):** IoU: 70.18% | F1: 82.48%
  - **Cloud-MAE Pretrained + Early Fusion (EXP-02):** IoU: 74.00% | F1: 85.06%
  - **Cloud-MAE + Cross-Modal Attention (EXP-04):** IoU: 75.05% | F1: 85.75%
  - **Cloud-MAE + Robust Augmentation (EXP-05):** IoU: 75.33% | F1: 85.93%
  - **Full Production Pipeline (EXP-06):** **IoU: 75.64% | F1: 86.13% | 0% Cloud IoU: 81.39%**
- **Cloud Robustness Stress Test:** Cloud-MAE retains high accuracy even under heavy cloud degradation (50% cloud drop only 2.92% IoU vs. 5.17% for baseline).

---

## 📂 Project Structure

```
Project_GISxKmutnb/
├── lib1_fusion/        # Library 1: Multi-Sensor Data Fusion (S1 + S2 + GEO-Sat)
│   ├── fusion.py       # Early, Late, and Cross-Modal Attention Fusion
│   └── preprocessor.py # 12-band normalization and cloud-mask handling
├── lib2_vitpalm/       # Library 2: ViTPalm Model Architecture
│   ├── model/          # CloudMAEPretrainViT, ViTPalmClassifier, ViTPalmModel
│   └── train/          # Cloud-MAE pretrainer & fine-tuning engine
├── configs/            # Experiment YAML configurations (Cloud-MAE, Fusion, etc.)
├── docs/               # Architecture documents and master development tracker
├── results/            # Benchmark metrics (CSV) and publication figures (PNG)
│   ├── figures/        # Training curves & 5-panel reconstruction figures
│   └── metrics/        # master_benchmark_results.csv
├── tests/              # Unit & integration test suite (pytest)
├── .gitignore          # Clean config ignoring local large rasters and checkpoints
└── README.md
```

---

## 🛰️ Sensor Fusion & Channel Layout

Converts multi-sensor inputs into a normalized stacked tensor of shape `(12, H, W)`:
1. **Sentinel-1 (SAR, 2 bands):** VV, VH (penetrates cloud cover)
2. **Sentinel-2 (Optical, 7 bands):** Blue (B2), Green (B3), Red (B4), NIR (B8), SWIR-1 (B11), SWIR-2 (B12), RedEdge (B5)
3. **GEO-Satellite (Optical, 3 bands):** Red, Green, Blue

---

## 📊 Benchmark Results

| Experiment | Method / Architecture | IoU (%) | F1-Score (%) | Precision (%) | Recall (%) | Stress Test (50% Cloud) IoU |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **EXP-01** | Supervised ViT (Scratch) | 70.18 | 82.48 | 84.81 | 80.26 | 65.01 |
| **EXP-02** | Cloud-MAE Pretrained + Early Fusion | 74.00 | 85.06 | 87.89 | 82.40 | 70.52 |
| **EXP-03** | Supervised Late Fusion ViT | 71.74 | 83.54 | 86.66 | 80.64 | 66.83 |
| **EXP-04** | Cloud-MAE + Cross-Modal Attention | 75.05 | 85.75 | 88.66 | 83.02 | 71.98 |
| **EXP-05** | Cloud-MAE + Robust Cloud Augmentation | 75.33 | 85.93 | 88.92 | 83.14 | 72.19 |
| **EXP-06** | **Full Production Pipeline** | **75.64** | **86.13** | **89.17** | **83.29** | **72.72** |

*(Comprehensive metrics stored in `results/metrics/master_benchmark_results.csv`)*

---

## 🧪 Running Tests & Pipelines

### Environment Setup
```powershell
conda activate gis_kmutnb
pip install -r requirements.txt
```

### Run Unit Tests
```powershell
pytest tests/ -v
```

### Run Master Benchmark Evaluation
```powershell
python poc/run_full_pipeline.py
```

---

## 💾 Data & Checkpoints Storage Policy
To ensure this repository remains completely lightweight, open-source, and **100% free forever without Git LFS quotas or bandwidth costs**, large raw satellite rasters (`*.tif`) and PyTorch weights (`*.pth`) are ignored from Git and stored on local disk / secure cloud storage.

---

## 📄 License & Attribution
KMUTNB GIS Lab — Academic Research Project.
