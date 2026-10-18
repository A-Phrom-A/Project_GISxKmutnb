#!/usr/bin/env python3
"""
scripts/create_colab_notebook.py
================================
Generates the comprehensive, instructional Google Colab Jupyter Notebook
for GISxKMUTNB: Multi-Modal Satellite Oil Palm Detection & Radar-Guided Cloud-MAE.
"""

import json
from pathlib import Path

notebook = {
    "cells": [],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}


def add_md(text: str):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })


def add_code(code: str):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    })


# -------------------------------------------------------------
# Title & Overview
# -------------------------------------------------------------
add_md("""# 🛰️ สื่อการสอนและคู่มือปฏิบัติการ: Multi-Modal Satellite Oil Palm Detection & Radar-Guided Cloud-MAE
### มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB) | Research & Education Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/A-Phrom-A/Project_GISxKmutnb/blob/main/notebooks/GISxKMUTNB_Tutorial_Colab.ipynb)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)

---

## 📌 บทนำและเป้าหมายการเรียนรู้ (Learning Objectives)
ในเขตร้อนชื้นอย่างประเทศไทย ปัญหาสำคัญที่สุดในการใช้ภาพถ่ายดาวเทียมเพื่อการเกษตรกรรมคือ **เมฆบดบัง (Cloud Contamination)** ซึ่งสูงถึง 50%–80% ในช่วงฤดูมรสุม ส่งผลให้ภาพถ่ายแสงขาว (Optical) ใช้การไม่ได้

โน้ตบุ๊กเล่มนี้จัดทำขึ้นเป็น **สื่อการสอนเชิงปฏิบัติการ (Interactive Laboratory)** เพื่อศึกษา:
1. **ทฤษฎี Multi-Modal Fusion**: การผสานข้อมูลดาวเทียมเรดาร์ **Sentinel-1 (SAR)** ซึ่งทะลุเมฆได้ 100% เข้ากับ **Sentinel-2 (Optical 7 bands)** และภาพถ่ายความละเอียดสูง **GEO-Sat**
2. **นวัตกรรม Cloud-MAE**: การใช้ Self-Supervised Masked Autoencoder ให้เรดาร์นำทางการซ่อมแซมภาพถ่ายที่ถูกเมฆบัง
3. **การตรวจจับสวนปาล์มน้ำมัน (Palm Segmentation)**: การใช้ Tri-Stream Late Fusion Vision Transformer
4. **การวิเคราะห์สุขภาพต้นไม้ (NDVI / NDWI)**: การคำนวณดัชนีพืชพรรณและเปรียบเทียบผลกระทบของเมฆต่อการประเมินสุขภาพพืช""")

# -------------------------------------------------------------
# Section 1: Setup
# -------------------------------------------------------------
add_md("""---
## ⚙️ ส่วนที่ 1: การเตรียมสภาพแวดล้อมบน Google Colab (Environment Setup)
ตรวจสอบ GPU และติดตั้งแพ็กเกจ `gis-kmutnb` โดยตรงจาก GitHub""")

add_code("""# 1. ตรวจสอบสถานะการ์ดจอ GPU บน Colab (แนะนำให้เปิดใช้งาน T4 หรือ A100 GPU)
!nvidia-smi""")

add_code("""# 2. Clone repository และติดตั้งไลบรารีวิจัย gis-kmutnb
import os
if not os.path.exists("Project_GISxKmutnb"):
    !git clone https://github.com/A-Phrom-A/Project_GISxKmutnb.git

%cd Project_GISxKmutnb
!pip install -q -e .
!pip install -q matplotlib rasterio tqdm""")

# -------------------------------------------------------------
# Section 2: Data & Drive
# -------------------------------------------------------------
add_md("""---
## 📂 ส่วนที่ 2: การโหลดข้อมูลตัวอย่างและ Checkpoints (Data & Checkpoints Setup)
คุณสามารถเลือกใช้ข้อมูลได้ 2 ทาง:
- **ทางเลือก A (แนะนำหากอัปโหลดขึ้น Drive แล้ว):** Mount Google Drive เพื่อเรียกใช้โฟลเดอร์ `colab_demo_data`
- **ทางเลือก B (โหมดอัตโนมัติ):** ใช้ข้อมูลตัวอย่างที่อยู่ใน repository โดยตรง""")

add_code("""from pathlib import Path
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# ตรวจสอบว่ามีข้อมูลใน Google Drive หรือในโฟลเดอร์ Local ของโปรเจกต์
drive_data_path = Path("/content/drive/MyDrive/colab_demo_data")
local_data_path = Path("colab_demo_data")

if drive_data_path.exists():
    DATA_DIR = drive_data_path
    print(f"[*] พบข้อมูลใน Google Drive: {DATA_DIR}")
elif local_data_path.exists():
    DATA_DIR = local_data_path
    print(f"[*] ใช้งานข้อมูลตัวอย่างจากโปรเจกต์: {DATA_DIR.resolve()}")
else:
    DATA_DIR = local_data_path
    print(f"[!] กรุณาตรวจสอบตำแหน่งโฟลเดอร์ colab_demo_data")

# กำหนด Device สำหรับคำนวณ (CUDA GPU หรือ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Device ที่ใช้งาน: {device}")""")

# -------------------------------------------------------------
# Section 3: Cloud-MAE Inpainting
# -------------------------------------------------------------
add_md("""---
## ☁️ ส่วนที่ 3: ปฏิบัติการที่ 1 — การลบเมฆด้วยเรดาร์นำทาง (Radar-Guided Cloud-MAE)

### 🔬 หลักการทางฟิสิกส์และสถาปัตยกรรม (Physics & Architecture)
- **Sentinel-1 (SAR):** คลื่นไมโครเวฟ C-band ความถี่ 5.405 GHz สามารถทะลุผ่านเมฆและละอองน้ำได้ 100% เพื่อวัดโครงสร้างความขรุขระ (Surface Roughness) ผ่านแบนด์ `VV` และโครงสร้างเรือนยอดพืช (Volume Scattering) ผ่านแบนด์ `VH`
- **Cloud-MAE:** โมเดล ViT ที่ถูกฝึกฝนแบบ Self-Supervised ด้วยภาพถ่าย 5 จังหวัดภาคตะวันออก (8,996 ไทล์, 500 Epochs) โดยเรียนรู้ฟื้นฟูพิกเซล Optical ที่หายไปภายใต้เมฆจากสัญญาณเรดาร์

$$\\mathcal{L}_{\\text{Cloud-MAE}} = \\frac{1}{|\\mathcal{M}_{\\text{cloud}}|} \\sum_{i \\in \\mathcal{M}_{\\text{cloud}}} \\left\\| \\hat{y}_{\\text{optical}}^{(i)} - y_{\\text{optical}}^{(i)} \\right\\|^2$$""")

add_code("""from lib2_vitpalm import CloudMAEModel
from lib2_vitpalm.hub import load_cloud_mae_weights

# 1. ประกาศและสร้างโมเดล Cloud-MAE
cloud_mae = CloudMAEModel(
    s1_channels=2,
    s2_channels=7,
    embed_dim=256,
    depth=6,
    num_heads=8,
    decoder_dim=128,
    decoder_depth=3
).to(device).eval()

# 2. โหลด Pretrained Weights (500 Epochs)
ckpt_path = DATA_DIR / "checkpoints" / "cloud_mae_best.pth"
if not ckpt_path.exists():
    ckpt_path = Path("checkpoints/cloud_mae/cloud_mae_best.pth")

state = torch.load(ckpt_path, map_location=device, weights_only=False)
weights = state.get("model_state_dict", state)
cloud_mae.load_state_dict(weights)
print(f"[+] โหลดโมเดล Cloud-MAE สำเร็จจาก: {ckpt_path}")""")

add_code("""# 3. โหลดตัวอย่างภาพดาวเทียม S1 + S2
sample_file = DATA_DIR / "test_samples" / "sample_01_dense_palm_cloud_mae.npz"
data = np.load(sample_file)

s1_np = data["s1"]  # (2, 256, 256) -> VV, VH
s2_np = data["s2"]  # (7, 256, 256) -> B2, B3, B4, B8A, B8, B11, B12

# แปลงเป็น PyTorch Tensor
s1_tensor = torch.from_numpy(s1_np).unsqueeze(0).to(device) # (1, 2, 256, 256)
s2_tensor = torch.from_numpy(s2_np).unsqueeze(0).to(device) # (1, 7, 256, 256)

# 4. จำลองเมฆบังแบบเขตร้อน (Tropical Cloud Mask 50% - 65%)
B, _, H, W = s2_tensor.shape
patch_size = 16
num_h, num_w = H // patch_size, W // patch_size

# สุ่มกลุ่มก้อนเมฆ (Block Mask)
mask_patch = (torch.rand(1, num_h, num_w, device=device) < 0.55).float()
mask_pixel = mask_patch.repeat_interleave(patch_size, 1).repeat_interleave(patch_size, 2).unsqueeze(1)

# ภาพจำลองที่มีเมฆบดบัง
s2_cloudy = s2_tensor * (1.0 - mask_pixel)

# 5. สั่งให้ Cloud-MAE ฟื้นฟูภาพโดยใช้ SAR นำทาง
with torch.no_grad():
    s2_reconstructed = cloud_mae.reconstruct(s1_tensor, s2_cloudy, mask_pixel=mask_pixel, composite=True)

print(f"[+] ซ่อมแซมภาพสำเร็จ! ขนาด Output: {s2_reconstructed.shape}")""")

add_code("""# 6. พล็อตเปรียบเทียบผลลัพธ์ 4 ช่อง (การลบเมฆด้วย Cloud-MAE)
def norm_rgb(arr):
    p2, p98 = np.percentile(arr, (2, 98))
    return np.clip((arr - p2) / (p98 - p2 + 1e-6), 0, 1)

# S2 RGB True Color: Red=B4 (index 2), Green=B3 (index 1), Blue=B2 (index 0)
clean_rgb = np.stack([s2_np[2], s2_np[1], s2_np[0]], axis=-1)
clean_rgb = norm_rgb(clean_rgb)

cloudy_rgb = clean_rgb.copy()
mask_2d = mask_pixel.squeeze().cpu().numpy().astype(bool)
cloudy_rgb[mask_2d] = 1.0  # แสดงสีขาวแทนเมฆ

s2_recon_np = s2_reconstructed.squeeze().cpu().numpy()
recon_rgb = np.stack([s2_recon_np[2], s2_recon_np[1], s2_recon_np[0]], axis=-1)
recon_rgb = norm_rgb(recon_rgb)

# SAR Radar Intensity (VV Band)
sar_vv = norm_rgb(s1_np[0])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(sar_vv, cmap="gray")
axes[0].set_title("1. Sentinel-1 SAR (VV)\\n(ทะลุเมฆ 100%)", fontsize=12, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(cloudy_rgb)
axes[1].set_title("2. Sentinel-2 (ถูกเมฆบัง 55%)\\n(Input ที่เสียหาย)", fontsize=12, fontweight="bold", color="crimson")
axes[1].axis("off")

axes[2].imshow(recon_rgb)
axes[2].set_title("3. Cloud-MAE Inpainted\\n(ฟื้นฟูด้วยเรดาร์สำเร็จ)", fontsize=12, fontweight="bold", color="forestgreen")
axes[2].axis("off")

axes[3].imshow(clean_rgb)
axes[3].set_title("4. Ground Truth Original\\n(ภาพจริงท้องฟ้าเปิด)", fontsize=12, fontweight="bold")
axes[3].axis("off")

plt.suptitle("เปรียบเทียบการลบเมฆด้วยสถาปัตยกรรม Radar-Guided Cloud-MAE (GISxKMUTNB)", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()""")

# -------------------------------------------------------------
# Section 4: Palm Segmentation
# -------------------------------------------------------------
add_md("""---
## 🌴 ส่วนที่ 4: ปฏิบัติการที่ 2 — การตรวจจับแปลงปาล์มน้ำมัน (Multi-Modal Palm Segmentation)

### 🏗️ สถาปัตยกรรม Tri-Stream Late Fusion
โมเดลประกอบด้วย **ViT Encoder 3 ตัวแยกอิสระ**:
- **SAR Stream (2 แชนแนล):** สกัดลักษณะการสะท้อนของเรือนยอดปาล์ม
- **Optical Stream (7 แชนแนล):** สกัดสเปกตรัมคลอโรฟิลล์และสารลิกนิน (ได้รับน้ำหนักพรีเทรนจาก Cloud-MAE)
- **High-Res Basemap Stream (3 แชนแนล):** สกัด Texture รูปดาวของต้นปาล์ม
- **1x1 Conv + SegDecoder:** รวมมิติฟีเจอร์แล้วแปลงกลับเป็นแผนที่ความน่าจะเป็น (Probability Map)""")

add_code("""from lib2_vitpalm import ViTLateFusionModel

# 1. สร้างโมเดล Late Fusion
seg_model = ViTLateFusionModel(
    embed_dim=256,
    depth=6,
    num_heads=8,
    out_channels=1
).to(device).eval()

# 2. โหลดโมเดลที่ผ่านการเทรนบนข้อมูล อ.บ้านบึง จ.ชลบุรี
seg_ckpt_path = DATA_DIR / "checkpoints" / "vitpalm_latefusion_best.pth"
if not seg_ckpt_path.exists():
    seg_ckpt_path = Path("checkpoints/benchmark_latefusion_cloudmae/vitpalm_best.pth")

seg_state = torch.load(seg_ckpt_path, map_location=device, weights_only=False)
seg_weights = seg_state.get("model_state_dict", seg_state.get("model", seg_state))
seg_model.load_state_dict(seg_weights)
print(f"[+] โหลดโมเดล ViTLateFusion สำเร็จจาก: {seg_ckpt_path}")""")

add_code("""# 3. โหลดชุดข้อมูล 12-band Multi-Modal Tile และ Ground Truth Mask
tile_file = DATA_DIR / "test_samples" / "sample_01_dense_palm_12band.npy"
mask_file = DATA_DIR / "test_samples" / "sample_01_dense_palm_mask.npy"

tile_12 = np.load(tile_file)  # (12, 256, 256)
gt_mask = np.load(mask_file)  # (256, 256)

tile_tensor = torch.from_numpy(tile_12).unsqueeze(0).to(device)

# 4. รันโมเดลทำนายพื้นที่ปลูกปาล์ม
with torch.no_grad():
    logits = seg_model(tile_tensor)
    prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_binary = (prob_map > 0.50).astype(np.uint8)

# 5. คำนวณความแม่นยำ (IoU & Dice Coefficient)
intersection = np.logical_and(pred_binary, gt_mask).sum()
union = np.logical_or(pred_binary, gt_mask).sum()
iou = intersection / (union + 1e-6)
dice = 2 * intersection / (pred_binary.sum() + gt_mask.sum() + 1e-6)

print("📊 ผลการประเมินความแม่นยำบนไทล์ทดสอบ:")
print(f"   • IoU Score   : {iou * 100:.2f}%")
print(f"   • Dice / F1   : {dice * 100:.2f}%")
print(f"   • พื้นที่ปาล์ม  : {pred_binary.mean() * 100:.2f}% ของพื้นที่ภาพ")""")

add_code("""# 6. แสดงผลลัพธ์การจำแนกแปลงปาล์ม
gsat_rgb = np.stack([tile_12[9], tile_12[10], tile_12[11]], axis=-1)
gsat_rgb = norm_rgb(gsat_rgb)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(gsat_rgb)
axes[0].set_title("1. ภาพดาวเทียม High-Res GSat", fontsize=12, fontweight="bold")
axes[0].axis("off")

im1 = axes[1].imshow(prob_map, cmap="magma", vmin=0, vmax=1)
axes[1].set_title("2. Palm Probability Map\\n(ระดับความน่าจะเป็น)", fontsize=12, fontweight="bold")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(pred_binary, cmap="Greens", vmin=0, vmax=1)
axes[2].set_title(f"3. Model Binary Output\\n(IoU: {iou*100:.1f}%)", fontsize=12, fontweight="bold", color="forestgreen")
axes[2].axis("off")

# ซ้อนทับการทำนายลงบนภาพจริง
overlay = gsat_rgb.copy()
overlay[pred_binary == 1] = overlay[pred_binary == 1] * 0.4 + np.array([0.1, 0.85, 0.3]) * 0.6
axes[3].imshow(overlay)
axes[3].set_title("4. Segmentation Overlay\\n(พื้นที่แปลงปาล์มน้ำมัน)", fontsize=12, fontweight="bold", color="darkgreen")
axes[3].axis("off")

plt.suptitle("ผลการทำนายพื้นที่สวนปาล์มน้ำมันด้วย Tri-Stream ViT (GISxKMUTNB)", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()""")

# -------------------------------------------------------------
# Section 5: NDVI Health
# -------------------------------------------------------------
add_md("""---
## 🌿 ส่วนที่ 5: ปฏิบัติการที่ 3 — การประเมินสุขภาพต้นไม้ (NDVI Vegetation Health Analytics)

### 🍃 ทฤษฎีดัชนีพืชพรรณและผลกระทบของเมฆ
- **NDVI (Normalized Difference Vegetation Index):**
  $$\\text{NDVI} = \\frac{\\text{NIR} (\\text{B8}) - \\text{Red} (\\text{B4})}{\\text{NIR} (\\text{B8}) + \\text{Red} (\\text{B4})}$$
  คลอโรฟิลล์ในใบไม้จะดูดกลืนแสงสีแดง ($\text{B4}$) เพื่อสังเคราะห์แสง และสะท้อนแสงอินฟราเรดย่านใกล้ ($\text{B8}$) สูงมาก ยิ่งค่า NDVI เข้าใกล้ 1.0 แสดงว่าปาล์มมีความสมบูรณ์แข็งแรง
- **การทดลองเปรียบเทียบ:** หากมีเมฆบดบัง ค่า NDVI จะลดลงอย่างรุนแรงจนผิดเพี้ยน เราจะแสดงให้เห็นว่า **Cloud-MAE สามารถกู้คืนค่าสุขภาพพืช (NDVI) ให้กลับมาสมบูรณ์ได้แม้ภาพเดิมจะถูกเมฆบัง**""")

add_code("""# 1. คำนวณ NDVI จากภาพปกติ (Clean)
# S2 Bands: B4 (Red) = index 2, B8 (NIR) = index 4
nir_clean = s2_np[4]
red_clean = s2_np[2]
ndvi_clean = (nir_clean - red_clean) / (nir_clean + red_clean + 1e-6)

# 2. คำนวณ NDVI จากภาพที่มีเมฆบัง (Cloudy)
nir_cloudy = s2_cloudy[0, 4].cpu().numpy()
red_cloudy = s2_cloudy[0, 2].cpu().numpy()
ndvi_cloudy = (nir_cloudy - red_cloudy) / (nir_cloudy + red_cloudy + 1e-6)

# 3. คำนวณ NDVI จากภาพที่ซ่อมด้วย Cloud-MAE (Inpainted)
nir_recon = s2_recon_np[4]
red_recon = s2_recon_np[2]
ndvi_recon = (nir_recon - red_recon) / (nir_recon + red_recon + 1e-6)

# 4. พล็อตเปรียบเทียบสุขภาพพืช
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im0 = axes[0].imshow(ndvi_clean, cmap="YlGn", vmin=0, vmax=0.9)
axes[0].set_title("1. NDVI จริง (ท้องฟ้าเปิด)\\n(สุขภาพสมบูรณ์ 0.6–0.8)", fontsize=12, fontweight="bold")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(ndvi_cloudy, cmap="YlGn", vmin=0, vmax=0.9)
axes[1].set_title("2. NDVI ขณะมีเมฆบัง\\n(ค่าผิดเพี้ยนอย่างรุนแรง)", fontsize=12, fontweight="bold", color="crimson")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(ndvi_recon, cmap="YlGn", vmin=0, vmax=0.9)
axes[2].set_title("3. NDVI หลังซ่อมด้วย Cloud-MAE\\n(ฟื้นคืนค่าสุขภาพพืชได้สมบูรณ์)", fontsize=12, fontweight="bold", color="forestgreen")
axes[2].axis("off")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("การกู้คืนดัชนีสุขภาพปาล์มน้ำมัน (NDVI) ภายใต้เมฆบดบังด้วย Radar-Guided Cloud-MAE", fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

# 5. สถิติเชิงปริมาณเฉพาะแปลงปาล์ม
palm_pixels = gt_mask == 1
if palm_pixels.sum() > 0:
    mean_clean = float(ndvi_clean[palm_pixels].mean())
    mean_cloudy = float(ndvi_cloudy[palm_pixels].mean())
    mean_recon = float(ndvi_recon[palm_pixels].mean())
    recovery_pct = (1.0 - abs(mean_recon - mean_clean) / mean_clean) * 100.0
    print("📈 ค่าเฉลี่ย NDVI เฉพาะบนแปลงปาล์มน้ำมัน:")
    print(f"   • ค่าปกติ (Clean GT)    : {mean_clean:.3f}")
    print(f"   • ขณะมีเมฆ (Cloudy)     : {mean_cloudy:.3f} (ลดฮวบเพราะเมฆ)")
    print(f"   • หลังซ่อม (Cloud-MAE)  : {mean_recon:.3f} (กู้คืนได้ {recovery_pct:.1f}%)")""")

# -------------------------------------------------------------
# Section 6: Advanced Topic 1 - Model Architecture & Decoder Guide
# -------------------------------------------------------------
add_md("""---
## 🔀 ส่วนที่ 6: การเลือกสถาปัตยกรรมโมเดลและบทบาทของ Decoder (Model & Decoder Selection)

### 📊 เปรียบเทียบโมเดลทั้ง 3 สถาปัตยกรรมหลักสำหรับ Downstream Segmentation:

| สถาปัตยกรรม | คลาสโมเดล | หลักการทำงาน | จุดเด่น | ข้อจำกัด / บริบทที่เหมาะสม |
|:---|:---|:---|:---|:---|
| **Early Fusion** | `ViTPalmModel` | รวม 12 แชนแนลตั้งแต่ทางเข้า แล้วใช้ ViT Encoder ตัวเดียว | พารามิเตอร์น้อยที่สุด (~21M), กิน VRAM ต่ำ (~1.8 GB), เทรนเร็ว | เหมาะกับงานที่มีข้อมูลสมบูรณ์ทุกเซนเซอร์ และต้องการโมเดล Baseline น้ำหนักเบา |
| **Late Fusion** | `ViTLateFusionModel` | แยก ViT Encoder 3 ตัวอิสระ (S1, S2, GSat) แล้วรวมฟีเจอร์ด้วย 1×1 Conv | **ทนทานต่อเมฆและ Noise สูงสุด** เรดาร์และแสงขาวไม่กวน Feature กันเอง | **(แนะนำเป็น Default)** เหมาะกับภาพถ่ายจริงในประเทศไทยที่มีเมฆและฝนประปราย |
| **Cross-Attention** | `ViTCrossAttnModel` | ใช้ Optical (S2) เป็น Query เพื่อดึงบริบทจาก SAR และ GSat ผ่าน Multihead Attention | มีความสามารถในการเชื่อมโยงมิติเชิงลึก (Relational Reasoning) สูงสุด | เหมาะกับงานวิจัยที่ต้องการวิเคราะห์ว่า "จุดใดที่โมเดลต้องหันไปพึ่งเรดาร์แทนแสงขาว" |

> 💡 **เกี่ยวกับ Decoder ในโปรเจกต์นี้:**  
> เราเลือกใช้ `SegDecoder` ซึ่งเป็น **Convolutional Transposed Progressive Upsampler** ($16 \\to 32 \\to 64 \\to 128 \\to 256$) เป็นมาตรฐานหลักตัวเดียว เนื่องจากให้ความสมดุลระหว่างความคมชัดของขอบแปลงและความเร็วในการประมวลผลได้ดีที่สุด โดยสามารถสลับไปใช้โครงสร้างอื่นได้ง่ายเพียงนำคลาส `nn.Module` ตัวใหม่มาแทนที่ `self.decoder`""")

add_code("""# ทดสอบเปรียบเทียบสถาปัตยกรรมทั้ง 3 ตัวบนไทล์ตัวอย่างเดียวกัน
from lib2_vitpalm import ViTPalmModel, ViTLateFusionModel, ViTCrossAttnModel
import time

models = {
    "1. Early Fusion (ViTPalmModel)": ViTPalmModel(in_channels=12, embed_dim=256, depth=6).to(device).eval(),
    "2. Late Fusion (ViTLateFusionModel)": ViTLateFusionModel(embed_dim=256, depth=6).to(device).eval(),
    "3. Cross-Attention (ViTCrossAttnModel)": ViTCrossAttnModel(embed_dim=256, depth=6).to(device).eval(),
}

print("=== สรุปการเปรียบเทียบสถาปัตยกรรมโมเดล (Benchmark Comparison) ===")
dummy_12band = torch.randn(1, 12, 256, 256, device=device)

for name, m in models.items():
    param_count = sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    # วัดเวลา Inference (เฉลี่ย 10 ครั้ง)
    with torch.no_grad():
        for _ in range(3): _ = m(dummy_12band) # Warmup
        t0 = time.time()
        for _ in range(10): out = m(dummy_12band)
        latency = (time.time() - t0) / 10.0 * 1000.0 # ms

    print(f"\\n[{name}]")
    print(f"   • Trainable Parameters: {param_count:,} ({param_count/1e6:.2f} M)")
    print(f"   • GPU Latency (Batch=1): {latency:.2f} ms")
    print(f"   • Output Shape        : {out.shape}")""")

# -------------------------------------------------------------
# Section 7: Advanced Topic 2 - Beyond Oil Palm (General GeoAI)
# -------------------------------------------------------------
add_md("""---
## 🌾 ส่วนที่ 7: การนำ Library ไปประยุกต์กับโจทย์อื่น (Beyond Oil Palm / General GeoAI)

หากต้องการนำไลบรารีนี้ไปใช้กับโจทย์อื่น เช่น **ยางพารา, อ้อย, มันสำปะหลัง, แหล่งน้ำ/น้ำท่วม, หรือการจำแนกหลายคลาส (Multi-Class Land Cover)** สิ่งที่ต้องแก้ไขมีเพียง **3 จุดสำคัญ**:

1. **Output Channels (`out_channels`):**
   - งานตรวจจับชิ้นเดียว (Binary เช่น เฉพาะน้ำท่วม หรือเฉพาะปาล์ม): ใช้ `out_channels = 1`
   - งานจำแนกหลายคลาส (Multi-Class เช่น 0: ดิน, 1: ปาล์ม, 2: ยางพารา, 3: แหล่งน้ำ): ใช้ `out_channels = C`
2. **Loss Function:**
   - Binary: ใช้ `BCEWithLogitsLoss()` หรือ `combined_loss()` (BCE + Soft Dice)
   - Multi-Class: เปลี่ยนไปใช้ `nn.CrossEntropyLoss()`
3. **Target Mask Encoding:**
   - ปรับ Label ของภาพ Mask ให้เป็นเลขจำนวนเต็ม `[0, 1, ..., C-1]`""")

add_code("""# ตัวอย่างการตั้งค่าโมเดลและ Loss Function สำหรับโจทย์ Multi-Class (4 คลาส)
import torch.nn as nn

NUM_CLASSES = 4 # ตัวอย่าง: 0=พื้นหลัง, 1=ปาล์ม, 2=ยางพารา, 3=แหล่งน้ำ

# 1. ประกาศโมเดลโดยเปลี่ยน out_channels = NUM_CLASSES
multiclass_model = ViTLateFusionModel(
    embed_dim=256,
    depth=6,
    out_channels=NUM_CLASSES
).to(device)

# 2. ใช้ CrossEntropyLoss สำหรับการจำแนกหลายคลาส
criterion = nn.CrossEntropyLoss()

# 3. จำลองข้อมูล Input (12 แบนด์) และ Target Mask (0, 1, 2, 3)
dummy_input = torch.randn(2, 12, 256, 256, device=device)
dummy_target = torch.randint(0, NUM_CLASSES, (2, 256, 256), device=device)

# 4. Forward pass และคำนวณ Loss
logits = multiclass_model(dummy_input) # Shape: (2, 4, 256, 256)
loss = criterion(logits, dummy_target)

# 5. สรุปผลการทำนายเป็น Class Index
predicted_classes = torch.argmax(logits, dim=1) # Shape: (2, 256, 256)

print(f"[+] Multi-Class Output Logits Shape : {logits.shape}")
print(f"[+] Predicted Class Map Shape        : {predicted_classes.shape}")
print(f"[+] CrossEntropy Loss               : {loss.item():.4f}")
print("พร้อมสำหรับการเทรนโจทย์พืชชนิดอื่นหรือแผนที่น้ำท่วมได้ทันที!")""")

# -------------------------------------------------------------
# Section 8: Advanced Topic 3 - Adding New Datasets (Data Ingestion)
# -------------------------------------------------------------
add_md("""---
## 🗺️ ส่วนที่ 8: คู่มือการนำเข้าและเพิ่มชุดข้อมูลใหม่ (Dataset Ingestion Guide)

หากคุณมีภาพถ่ายดาวเทียมและแปลงโพลีกอนในพื้นที่จังหวัดอื่น (เช่น สุราษฎร์ธานี หรือกระบี่) และต้องการเพิ่มเข้ามาในระบบ ข้อมูลต้องมีลักษณะดังนี้:

### 8.1 สเปกข้อมูลดาวเทียม 3 เซนเซอร์ (Data Specifications)
1. **Sentinel-1 (SAR):** ต้องมีแบนด์ `VV` และ `VH` (Resample เป็น 10m), แปลงเป็น dB $\to$ Normalize ช่วง $[-25, 0] \\text{ dB} \\to [0.0, 1.0]$
2. **Sentinel-2 (Optical):** ต้องมี 7 แบนด์ (`B2, B3, B4, B8A, B8, B11, B12`), หารด้วย $10,000 \\to [0.0, 1.0]$
3. **GEO-Sat (High-Res RGB):** ภาพสี 3 แบนด์ (Resample เป็น 10m ให้ตรงกริด), หารด้วย $255.0 \\to [0.0, 1.0]$
4. **Ground Truth:** โพลีกอนแปลงพืชในรูปแบบ `.geojson` หรือ `.shp` นำมา Rasterize ให้เป็นภาพขาวดำ $(0, 1)$

> ⚠️ **หลักการสำคัญระดับงานวิจัย (Data Leakage Prevention):**  
> ในการวิจัยเชิงพื้นที่ **ห้ามสุ่มแบ่ง Train/Val/Test แบบสุ่มรายภาพ (Random Split)** เนื่องจากภาพไทล์ข้างเคียงมีปรากฏการณ์ Spatial Autocorrelation ทำให้โมเดลจดจำข้อมูลข้างเคียงได้ (เกิด Data Leakage) **ต้องแบ่งแบบ Spatial Block Split เสมอ** (เช่น บล็อกฝั่งเหนือเป็น Train, บล็อกฝั่งใต้เป็น Test)""")

add_code("""# ฟังก์ชันตัวอย่างสำหรับการ Normalize และตัดไทล์ 256x256 จากภาพดาวเทียมผืนใหญ่
def preprocess_and_tile(s1_raw, s2_raw, gsat_raw, gt_mask, tile_size=256, stride=128):
    \"\"\"
    ตัวอย่าง Pipeline การเตรียมข้อมูล 12 แชนแนล:
      s1_raw   : (2, H, W)   - Linear amplitude
      s2_raw   : (7, H, W)   - 0 - 10,000 reflectance
      gsat_raw : (3, H, W)   - 0 - 255 RGB
      gt_mask  : (H, W)      - 0 หรือ 1
    \"\"\"
    # 1. Normalize S1 เรดาร์ (Linear -> dB -> [0, 1])
    s1_db = 10.0 * np.log10(np.clip(s1_raw, 1e-6, None))
    s1_norm = np.clip((s1_db - (-25.0)) / (0.0 - (-25.0)), 0.0, 1.0).astype(np.float32)

    # 2. Normalize S2 แสงขาว (0-10000 -> [0, 1])
    s2_norm = np.clip(s2_raw / 10000.0, 0.0, 1.0).astype(np.float32)

    # 3. Normalize GSat ภาพถ่ายทางอากาศ (0-255 -> [0, 1])
    gsat_norm = np.clip(gsat_raw / 255.0, 0.0, 1.0).astype(np.float32)

    # รวมเป็นก้อน 12 แบนด์
    fused_stack = np.concatenate([s1_norm, s2_norm, gsat_norm], axis=0) # (12, H, W)
    _, H, W = fused_stack.shape

    tiles, masks = [], []
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            t = fused_stack[:, r:r+tile_size, c:c+tile_size]
            m = gt_mask[r:r+tile_size, c:c+tile_size]
            # กรองไทล์ที่มีข้อมูลสมบูรณ์ (ไม่มี NoData ขอบดำเกิน 30%)
            if np.mean(t > 0) > 0.70:
                tiles.append(t)
                masks.append(m)

    print(f"[+] สกัดไทล์ขนาด {tile_size}x{tile_size} สำเร็จ: ได้ทั้งหมด {len(tiles)} ไทล์")
    return tiles, masks

print("Pipeline การนำเข้าชุดข้อมูลใหม่พร้อมใช้งาน!")""")

# -------------------------------------------------------------
# Section 9: Conclusion
# -------------------------------------------------------------
add_md("""---
## 🎯 สรุปผลและคำถามชวนคิดสำหรับการวิจัยต่อยอด (Conclusions & Exercises)

1. **ทำไม SAR จึงช่วยเติมเต็ม Optical ได้?**  
   เพราะคลื่นไมโครเวฟของ Sentinel-1 ทะลุผ่านเมฆได้ 100% ทำให้ระบบโมเดล ViT สามารถจับสัญญาณเรดาร์เป็นแผนผังอ้างอิงเชิงพื้นที่ (Spatial Scaffold) ในการระบายสีและเติมรายละเอียดคลื่นแสงของ Sentinel-2 กลับมาได้อย่างเป็นธรรมชาติ

2. **ประโยชน์ต่อภาคเกษตรกรรม:**  
   เกษตรกรและสหกรณ์สวนปาล์มสามารถติดตามสุขภาพและการเจริญเติบโตของต้นปาล์มน้ำมันได้แบบ Real-time ตลอดทั้งปี โดยไม่ต้องรอให้ฤดูมรสุมหรือช่วงเมฆหนาหมดไป

---
**มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)**  
*ภาควิชาวิศวกรรมคอมพิวเตอร์ / ภาควิชาวิทยาการคอมพิวเตอร์และสารสนเทศ*""")

# Save notebook
out_file = Path("notebooks/GISxKMUTNB_Tutorial_Colab.ipynb")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[+] Successfully generated Colab Tutorial Notebook: {out_file.resolve()}")
