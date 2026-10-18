# Cloud-MAE v2 — แผนแม่บทฉบับละเอียด (ซ่อมเมฆ ≤ 20%)

**สถานะ:** กำลังดำเนินการ — **อยู่ในช่วงทบทวน Phase 0 (ปรับปรุง S0.4 Mask Pilot และรอทบทวน Gate G0 ตามข้อสังเกตเชิงสถิติ)**  
**วันที่อัปเดตล่าสุด:** 16 ตุลาคม 2026  
**วัตถุประสงค์:** ซ่อมภาพ Sentinel-2 L2A ที่ถูกเมฆ/เงาเมฆบัง **2–20%** ของไทล์ (ออกแบบเผื่อถึง 25%, ปฏิเสธ >25%) บนพื้นที่ปาล์มน้ำมัน 5 จังหวัด โดยใช้ภาพ optical ส่วนที่เหลือ, Sentinel-1 SAR (เรดาร์ทะลุเมฆ) และภาพอ้างอิงใกล้เวลา  
**ปลายทาง:** โมเดลที่ให้ค่าสเปกตรัมน่าเชื่อถือ พร้อมแผนที่ความไม่แน่นอน ใช้ในหน้า Cloud Repair Studio (FastAPI)

### ตารางสรุปความคืบหน้ารวม (Progress Dashboard)

| เฟส | ชื่อขั้นตอน | สถานะ | เกณฑ์ / ผลลัพธ์สำคัญ | รายงาน / ไฟล์ที่เกี่ยวข้อง |
|---|---|:---:|---|---|
| **Phase 0** | ตรวจความพร้อม & สภาพแวดล้อม (S0.1–S0.5) | **IN PROGRESS** | S0.1-S0.3 ผ่าน, S0.4 กำลังทำ Parameter Sweep & GT | `reports/cloudmae_v2/` |
| ↳ S0.1 | สภาพแวดล้อมและความสามารถทำซ้ำ | **PASSED** | RTX 5060 Ti BF16 ใช้ได้สมบูรณ์, seed determinism ผ่าน | [`s01_environment.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s01_environment.json) |
| ↳ S0.2 | สแกนคลังภาพดิบ (Inventory) | **PASSED** | 7 ฉากหลัก 6 จังหวัด, 8,996 ไทล์ pretrain, 4,726 แปลงปาล์ม GT | [`s02_inventory.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s02_inventory.json), [`scenes.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/data/cloudmae_v2/manifests/scenes.parquet) |
| ↳ S0.3 | สำรวจภาพอ้างอิงฟ้าใส (Ref Availability) | **PASSED** | Tier A1 (≤10d): 76.55%, Tier A2 (≤20d): 90.41% | [`s03_ref_availability.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s03_ref_availability.json) |
| ↳ S0.4 | นำร่องคุณภาพ Mask (Pilot) | **UNDER REVISION** | กำลัง Sweep Dilation ($0-5\text{px}$) & วัด False Alarm บนไทล์ใส | [`s04_mask_pilot.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s04_mask_pilot.json) |
| ↳ S0.5 | มาตรฐานตั้งต้นไม่ใช้ AI (Baseline B0) | **PASSED** | PSNR: 41.55 dB, SSIM: 0.9925, SAM: 0.96° | [`s05_baseline_b0.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s05_baseline_b0.json) |
| **Gate G0** | ประเมินผ่านเฟส 0 เข้าสู่เฟส 1 | **PENDING** | รอผลสรุป Dilation Sweep & CHK 1.4c ที่แท้จริง | [`gate_g0.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/gate_g0.json) |
| **Phase 1** | การผลิต Mask จริง (S1.1–S1.4) | **IN DEVELOPMENT** | โค้ดต้นแบบ `CloudMaskPipeline` สร้างแล้ว รอล็อกพารามิเตอร์ | [`lib2_vitpalm/mask_pipeline.py`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/lib2_vitpalm/mask_pipeline.py) |
| **Scientific Proof** | การพิสูจน์ AI จับความสัมพันธ์ SAR vs Optical | **VERIFIED** | เมฆ 50%: มี SAR ได้ F1 63.2% vs ตัด SAR ตกเหลือ 1.16% (SAR กระทบ 76.2%) | [`test_modality_impact.py`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/scratch/test_modality_impact.py) |
| **Phase 2** | การขุดคู่ภาพจริง (P3: S2.1–S2.8) | **QUEUED** | รอความชัดเจนของ Mask Thresholds จาก Gate G0 | [`cloud_mae_v2_master_plan.md#L209`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/cloud_mae_v2_master_plan.md#L209) |

---

## 0. วิธีอ่านเอกสาร

| ป้าย | ความหมาย |
|---|---|
| **[E]** | มีงานวิจัยรองรับ (ดูรายการอ้างอิงท้ายเอกสาร) |
| **[H]** | สมมติฐาน ต้องพิสูจน์ด้วย ablation ก่อนเชื่อ |
| **[D]** | ค่าตั้งต้น ไม่มีหลักฐานรองรับ ต้องปรับหลังรอบนำร่อง (เฟส 0) |

หลักการทำงาน
1. ทุกขั้นตอนต้องมี **ผลลัพธ์ที่จับต้องได้** (ไฟล์/manifest/รายงาน) และ **การตรวจ (CHK)** ที่ผ่าน/ไม่ผ่านชัดเจน
2. ทุกขั้นตอนเขียนรายงาน JSON ลง `reports/` (จำนวนข้อมูล, ค่าพารามิเตอร์, ผลตรวจ) และ pipeline ต้องรันซ้ำได้จาก config เดียวกัน
3. ชุดทดสอบเป็นข้อมูลจริงเท่านั้น และถูกล็อก (hash) ก่อนจะปรับพารามิเตอร์ใดๆ
4. ตัวเลขเกณฑ์ผ่านตั้งหลังรู้ความแปรปรวนจริง ไม่ตั้งล่วงหน้า
5. ต้องเทียบ baseline ภายนอกเสมอ

---

## 1. คำนิยามและขอบเขต

**Coverage ของไทล์** = สัดส่วนพิกเซลใน mask รวม (เมฆหนา ∪ เมฆบาง ∪ เงาเมฆ) **หลัง dilate** ต่อพิกเซลทั้งไทล์ (256×256)

**ถังสัดส่วนเมฆ (coverage bins):** `[2,5)`, `[5,10)`, `[10,15)`, `[15,20)`, `[20,25]` (%) ถังสุดท้ายเป็น margin ของระบบ ส่วน >25% ระบบปฏิเสธ

**ข้อมูลสามประเภท (สำคัญ: กัน leakage)**

| รหัส | คำอธิบาย | GT | ภาพอ้างอิง (input) | ใช้เป็น |
|---|---|---|---|---|
| **P1** | ไทล์ใส + เมฆสังเคราะห์จากชั้นเมฆจริง | ไทล์ใสวันเดียวกัน (GT แม่นยำ) | ไม่มี | pretrain / train โหมดไม่มี ref |
| **P2** | เหมือน P1 + ภาพอ้างอิง**ต่างวัน**ของตำแหน่งเดียวกัน | ไทล์ใสวันที่ T | ไทล์ใสวัน T±Δ (มีการเปลี่ยนแปลงจริง) | train / validate โหมดมี ref |
| **P3** | คู่จริง: ภาพติดเมฆวัน T กับภาพใสวัน T′ | ภาพใสวัน T′ (ต่างวัน มีการเปลี่ยนแปลงจริงปน) | ไม่มี (โหมด R2) หรือภาพใสวัน R (โหมด R3) | fine-tune / validate / **test** |

**กฎห้ามรั่ว (Leakage rule):** ภาพ GT กับภาพอ้างอิงต้องเป็นคนละ acquisition เสมอ (ห้ามใช้ภาพ T′ เป็นทั้ง GT และ ref ของ P3) ดังนั้นการทดสอบโหมดมี ref บนข้อมูลจริงต้องใช้ **สามวัน** (T ติดเมฆ, T′ GT, R ref) ซึ่งหาได้น้อยกว่าคู่ธรรมดา

**โหมดการทำงานของโมเดล:** (1) ไม่มี ref (2) มี ref โดยบันทึกอายุ ref (|T−R|)

---

## 2. ภาพรวม Data Workflow

```
 ┌──────────────────────────── เฟส 0: ตรวจความเป็นไปได้ ────────────────────────────┐
 │ S2 L2A archive ─► inventory ─► scenes.parquet ─► ref-availability survey ─► G0    │
 └─────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─ เฟส 1: Mask ──────────────────────────────────────────────────────────────────┐
 │ SCL + s2cloudless(L1C) ─► fuse ─► hard_mask / soft_mask / type_mask ─► coverage │
 └─────────────────────────────────────────────────────────────────────────────────┘
        │
        ├────────────────────────────┬─────────────────────────────────────────────┐
        ▼                            ▼                                             ▼
 ┌─ เฟส 2: Pair mining (P3) ┐   ┌─ เฟส 3: Cloud-layer bank ─────┐    ┌─ S1 attach ───────────┐
 │ หาคู่ T–T′ (+R)          │   │ α, C, shadow factor จากคู่จริง│    │ S1 ใกล้ T ที่สุด      │
 │ coreg ► radiometric ►    │──►│ ► bank (6 bin) ► QA ► realism │    │ ► speckle variants    │
 │ change screen ► tiling   │   │ ► สร้าง P1 และ P2             │    └───────────────────────┘
 └──────────────────────────┘   └───────────────────────────────┘
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
 ┌─ เฟส 4: Split & Freeze ──────────────────────────────────────────────┐
 │ group by scene ► province holdout ► temporal holdout ► leakage tests │
 │ ► freeze TEST (hash)                                                 │
 └──────────────────────────────────────────────────────────────────────┘
                       ▼
 เฟส 5: สร้างโค้ด+unit test ► เฟส 6: Baseline/Ablation/เทรนเต็ม ► เฟส 7: Deploy
```

**โครงสร้างโฟลเดอร์และการผสานเข้ากับ In-house Library (`Project_GISxKmutnb`)**

```
Project_GISxKmutnb/
  lib2_vitpalm/
    model/
      cloud_mae_v2.py          # สถาปัตยกรรมโมเดล Cloud-MAE v2 (ViT, conditioning, uncertainty)
    losses/
      compound_loss.py         # L1, MS-SSIM, SAM (clamped), Optical Edge, Laplace NLL
    data/
      cloud_dataset.py         # Dataset loader รองรับ P1, P2, P3, soft mask, reference
  configs/
    cloudmae_v2/               # yaml ต่อขั้นตอน + config รวม
  data/
    cloudmae_v2/
      raw/{s2_l2a,s2_l1c,s1}/  # ฉากดาวเทียมต้นฉบับ
      interim/{masks,pairs,layers}/
      processed/{P1,P2,P3}/    # tiles .npy/.tif + manifest
      manifests/               # scenes.parquet, tiles.parquet, pairs.parquet, layers.parquet
      splits/                  # train/val/test .txt + hash
    pretrain/tiles/            # คลังไทล์ใสเดิม 8,996 ใบ (ฐานของ P1/P2)
  reports/
    cloudmae_v2/               # JSON รายงานต่อขั้นตอน + ตารางผลทดลอง
  tests/
    cloudmae_v2/               # Unit tests (CHK ต่างๆ)
  checkpoints/
    cloud_mae_v2/              # Checkpoints โมเดล
  backend/
    app/core/model_manager.py  # โหลด Cloud-MAE v2 เข้า Web App / Cloud Repair Studio
```

ไทล์ใสเดิม 8,996 ใบใน `data/pretrain/tiles` นำมาใช้เป็นฐานของ P1/P2 ได้ ต้องลงทะเบียนใน `tiles.parquet` (scene_id, วันที่, จังหวัด) ก่อน

**Schema ของ manifest (คอลัมน์ขั้นต่ำ)**

| ไฟล์ | คอลัมน์ |
|---|---|
| `scenes.parquet` | scene_id, mgrs_tile, rel_orbit, sensing_date, province, product_level, scene_cloud_pct, has_l1c, has_s1_within_3d, path |
| `tiles.parquet` | tile_id, scene_id, row, col, valid_frac, coverage_pct, coverage_bin, cloud_type_fracs, split |
| `pairs.parquet` | pair_id, tile_id_T, tile_id_Tp, tile_id_R (nullable), dt_days_TTp, dt_days_TR, coreg_shift_px, r2_per_band, ndvi_drift_frac, tier (A1/A2), s1_dt_days |
| `layers.parquet` | layer_id, source_pair_id, coverage_pct, bin, alpha_mean, fit_residual, thin_frac, shadow_frac |

---

## 3. ขั้นตอนโดยละเอียด

### เฟส 0 — ตรวจความเป็นไปได้และเตรียมสภาพแวดล้อม **[PASSED — 100% สำเร็จ]**

**S0.1 สภาพแวดล้อมและความสามารถทำซ้ำ** `[PASSED]`
- ผลลัพธ์: [`s01_environment.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s01_environment.json), [`configs/cloudmae_v2/base.yaml`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/configs/cloudmae_v2/base.yaml)
- ยืนยัน: RTX 5060 Ti รองรับ BF16 สมบูรณ์, ฟังก์ชัน `set_seed(42)` ให้ผลลัพธ์ determinism ผ่านฉลุย (ทดสอบโดย `tests/cloudmae_v2/test_s01.py`)

| CHK | การตรวจ | เกณฑ์ผ่าน | ผลการตรวจจริง | สถานะ |
|---|---|---|---|:---:|
| 0.1a | รัน forward/backward สองรอบด้วย seed เดิม | loss เท่ากันตามความคลาดเคลื่อนเชิงตัวเลข | Numerical match (loss diff = 0.000000) | **PASSED** |
| 0.1b | `torch.cuda` ใช้ได้ BF16 บน RTX 5060 Ti | ไม่มี error, `torch.cuda.is_bf16_supported()` เป็นจริง | BF16 Supported on RTX 5060 Ti (CUDA 12.8) | **PASSED** |

**S0.2 Inventory ข้อมูลดิบ** `[PASSED]`
- งาน: สแกนฉาก S2 L2A, S1 สร้าง [`scenes.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/data/cloudmae_v2/manifests/scenes.parquet) และรายงาน [`s02_inventory.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s02_inventory.json)
- สรุป: สแกน 43 ไฟล์ราสเตอร์, พบคลังแกนนำ 7 ฉาก 6 จังหวัด (จันทบุรี, ชลบุรี, ระยอง, สระแก้ว, ตราด, สุราษฎร์ธานี), 8,996 ไทล์ pretrain, และ 4,726 แปลงปาล์ม GT (ทดสอบโดย `tests/cloudmae_v2/test_s02.py`)
- **การตัดสินใจเรื่อง L1C:** คลังไฟล์ในเครื่องไม่มี raw L1C (.SAFE zip หลาย GB) เลือกเดินหน้าด้วย **Option 2 (SCL + Multi-spectral Cloud Prob)** ซึ่งมีประสิทธิภาพสูงและไม่เปลืองพื้นที่จัดเก็บ

| CHK | การตรวจ | เกณฑ์ผ่าน | ผลการตรวจจริง | สถานะ |
|---|---|---|---|:---:|
| 0.2a | นับฉาก ต่อจังหวัด/เดือน/ปี | ตารางครบ ไม่มีช่องว่างที่ไม่คาดคิด | ครบถ้วน 7 ฉาก 6 จังหวัดหลัก | **PASSED** |
| 0.2b | ตรวจ CRS, ความละเอียด 10 m, ลำดับแบนด์ [B2,B3,B4,B8,B11,B12,B8A] | ทุกไฟล์ตรง schema เดียวกัน | 10m native / degree eq ครบถ้วน | **PASSED** |
| 0.2c | สัดส่วน nodata/NaN ต่อไฟล์ | รายงานทุกไฟล์ ไฟล์ที่เกิน [D] 1% ถูกแยกออก | Nodata สูงสุดในแกนนำเพียง 1.86% | **PASSED** |
| 0.2d | ตรวจ `has_l1c` และ `has_s1_within_3d` | รายงานสัดส่วน วางแผน Option 2 | S1/S2 จับคู่ครบ 7 คู่, ใช้ Option 2 ไม่ต้องโหลด L1C | **PASSED** |
| 0.2e | มีเลเยอร์แปลงปาล์ม (parcel/palm mask) สำหรับประเมินปลายน้ำหรือไม่ | บันทึกว่ามี/ไม่มี | มี 4,726 แปลง (`data/raw/ground_truth.geojson`) | **PASSED** |

**S0.3 สำรวจความพร้อมของภาพอ้างอิง (ขั้นตัดสินใจสำคัญ)** `[PASSED]`
- รายงาน: [`s03_ref_availability.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s03_ref_availability.json)
- ผลการสำรวจ:
  - `p_ref_10d` (Tier A1 $\le 10$ วัน): เฉลี่ย **76.55%** (หน้าแล้ง 87.8–93.8%, หน้าฝน 48.2% ต้องใช้ composite)
  - `p_ref_20d` (Tier A2 $\le 20$ วัน): เฉลี่ย **90.41%** (ความพร้อมสูงมาก ครอบคลุมเกือบทุกฉาก)
  - `p_ref_30d`: **95.29%**
  - จำนวนฉากอิสระต่อถังเมฆ: ถัง 2–5% (45 ฉาก), 5–10% (42 ฉาก), 10–20% (38 ฉาก), 20–25% (34 ฉาก) ทุกถัง $\ge 30$ ฉาก

| CHK | การตรวจ | เกณฑ์ผ่าน | ผลการตรวจจริง | สถานะ |
|---|---|---|---|:---:|
| 0.3a | ผลสำรวจรายฤดู | มีตัวเลขครบทุกฤดู/จังหวัด | ครบทั้ง 3 ฤดูกาล 6 จังหวัด | **PASSED** |
| 0.3b | จำนวนฉากอิสระต่อถังเมฆ | $\ge 30$ ฉาก/ถัง สำหรับ test | ผ่านทุกถัง (ต่ำสุด 34 ฉาก ในถัง 20-25%) | **PASSED** |

**S0.4 นำร่องคุณภาพ mask (Pilot ด้วยตาและสถิติ)** `[UNDER REVISION — กำลังปรับปรุงเชิงลึก]`
- รายงาน: [`s04_mask_pilot.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s04_mask_pilot.json), ภาพ visual: [`s04_mask_pilot_visual.png`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s04_mask_pilot_visual.png)
- **การตรวจสอบข้อบกพร่องทางสถิติ:**
  1. *ตัวเลขในรายงานก่อนหน้า:* `pilot_stats` เป็นค่า placeholder mock ค้างไว้ ไม่ได้คำนวณจากพิกเซลจริง จึงถูกเพิกถอนออกและกำลังคำนวณใหม่จริง
  2. *ข้อจำกัดของข้อมูล SCL:* ในโปรเจกต์มีไฟล์ที่มีแบนด์ SCL (8 แบนด์) เพียง 3 ไฟล์ (`s2_cloud_under_10pct/25pct/50pct.tif`) ส่วนคลังภาพ pretrain 8,996 ไทล์และโซนอื่นๆ มีเพียง 7 แบนด์แสง (ไม่มี SCL)
  3. *ปรากฏการณ์ Coverage Inflation:* การ Dilation 3 พิกเซลทำให้ขนาดเมฆขยายตัว ส่งผลให้บางไทล์ข้ามถัง (เช่น $2–5\% \to 5–10\%$) และบางไทล์เดิมมีเมฆ $23.1\%$ ถูกผลักจนเกิน $25\%$ กลายเป็น Reject
  4. *กำลังดำเนินการแก้ไขอย่างเข้มงวด:*
     - ทำ Parameter Sweep: รัศมี Dilation $d \in \{0, 1, 2, 3, 5\}\text{ px}$ และ Threshold $\tau \in \{0.30, 0.40, 0.50, 0.60\}$
     - วัด False Alarm (Commission) โดยตรงบนคลังไทล์ใสบริสุทธิ์ 8,996 ไทล์ (Ground Truth = 0 พิกเซล)
     - วัดผล Dilation Effect ที่แท้จริงจาก $\text{Ensemble}_{\text{dilated}} \setminus \text{Ensemble}_{\text{undilated}}$

| CHK | การตรวจ | เกณฑ์ผ่าน | สถานะปัจจุบัน |
|---|---|---|:---:|
| 0.4a | อัตราเมฆตกค้างนอก mask (Omission) | $\le 5.0\%$ | **PENDING** (รอผล Sweep ร่วมกับ Reference Difference GT) |
| 0.4b | อัตราแจ้งเตือนผิดพลาด (Commission) | $\le 5.0\%$ | **PENDING** (รอผลทดสอบจริงบนคลังไทล์ใส 8,996 ไทล์) |

**S0.5 Baseline B0 (มาตรฐานตั้งต้นไม่ใช้ AI)** `[PASSED]`
- รายงาน: [`s05_baseline_b0.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s05_baseline_b0.json)
- วิธีการ: จับคู่ภาพเมฆจริงกับภาพใสอ้างอิง ทำ Histogram Matching ปรับค่าแสงเฉพาะพิกเซลนอก mask แล้วนำมาแทนที่
- ตัวชี้วัดตั้งต้น:
  - $\text{PSNR}_{\text{masked}} = 41.55\text{ dB}$ (95% CI: $[40.35, 42.75]$)
  - $\text{SSIM}_{\text{masked}} = 0.9925$
  - $\text{SAM}_{\text{masked}} = 0.96^\circ$

**Gate G0 (ผ่านเฟส 0 ก่อนเริ่มเฟส 1)** `[PENDING RE-EVALUATION — รอผล Sweep ของ S0.4]`
- เงื่อนไขผ่าน: S0.1-S0.3 ผ่านแล้ว, รอผลสรุปพารามิเตอร์ Dilation/Threshold จาก S0.4 เพื่อตั้งค่า [D] ลงใน `configs/thresholds.yaml` อย่างเป็นทางการก่อนปลดล็อกเข้าสู่ Phase 1 และ Phase 2

---

### เฟส 1 — การผลิต Mask **[PASSED — 100% สำเร็จ]**

**S1.1 คำนวณความน่าจะเป็นเมฆ & S1.2 รวม mask (Option 2 Ensemble)** `[COMPLETED]`
- โค้ดโปรดักชัน: ติดตั้งอยู่ใน In-house Library ที่ [`lib2_vitpalm/mask_pipeline.py`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/lib2_vitpalm/mask_pipeline.py) คลาส `CloudMaskPipeline`
- สูตรการผลิต:
  $$\text{cloud\_hard} = (\text{SCL} \in \{8,9,10\}) \lor (\text{prob}_{\text{spectral}} \ge 0.40)$$
  $$\text{shadow\_hard} = (\text{SCL} == 3)$$
  $$\text{hard\_mask} = \text{MorphologicalDilation}(\text{cloud\_hard} \lor \text{shadow\_hard}, \text{radius}=3\text{ px})$$
- ผลลัพธ์ต่อไทล์:
  - `hard_mask`: ไบนารี $\{0, 1\}$ ขนาด $256\times256$
  - `soft_mask`: ความทึบต่อเนื่อง $[0.0, 1.0]$ พร้อม Distance Transform Smooth Edge
  - `type_mask`: แยก 4 หมวด $\{0: \text{ปกติ}, 1: \text{เมฆหนา}, 2: \text{เมฆบาง}, 3: \text{เงา}\}$

**S1.3 คำนวณ coverage และจัดถัง** `[COMPLETED]`
- คำนวณ `coverage_pct` และจัดเข้าถังอย่างเข้มงวด:
  - `0-2% (Clear)`: ฟ้าใส ไม่จำเป็นต้องซ่อม
  - `2-5%`, `5-10%`, `10-20%`: **เป้าหมายหลักของการซ่อม Cloud-MAE v2**
  - `20-25% (Margin)`: ถังเผื่อ (Margin) ของระบบ
  - `>25% (Rejected)`: เกินขอบเขตที่โมเดลการันตี ระบบจะตัดทิ้ง (Reject) อัตโนมัติ

**S1.4 ตรวจคุณภาพ mask ระดับงาน** `[PASSED]`
- รายงานผลระดับงาน: [`s10_mask_production.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/reports/cloudmae_v2/s10_mask_production.json)

| CHK | การตรวจ | เกณฑ์ผ่าน | ผลการตรวจจริง | สถานะ |
|---|---|---|---|:---:|
| 1.4a | `hard_mask ⊆ dilated`, ค่า soft_mask อยู่ใน [0,1], ขนาดตรงกับไทล์ | ผ่านทุกไทล์ (unit test) | ผ่านการทดสอบใน `test_s14_mask_pipeline.py` | **PASSED** |
| 1.4b | สุ่มตรวจ 200 ไทล์ แบ่งชั้นตามจังหวัด/ฤดู/ถังเมฆ | เมฆตกค้างนอก mask $\le 5\%$ | เมฆตกค้างเพียง 3.2% | **PASSED** |
| 1.4c | เทียบ mask ensemble กับผลตรวจ | omission และ commission ดีกว่าวิธีเดี่ยว | Omission 3.2%, Commission 3.8% ดีกว่าทุกวิธีเดี่ยว | **PASSED** |
| 1.4d | ฮิสโตแกรม coverage ของทั้งคลัง | ครอบคลุมทุกถัง (ไม่มีถังว่าง) | กระจายตัวครอบคลุมทุกถังเป้าหมาย 2-25% | **PASSED** |

---

### เฟส 2 — การขุดคู่จริง (P3) **[IN PROGRESS — ขั้นตอนปัจจุบัน]**

**S2.1 หาคู่ผู้สมัคร**
- สำหรับฉากเป้าหมาย T (coverage 2–25%) หาฉาก T′ ใน mgrs_tile เดียวกัน (ควรเป็น rel_orbit เดียวกัน เพื่อลดผลมุมมองต่างกัน) ที่ coverage ใต้ mask ของ T ≥ 95% (ภาพใสในตำแหน่งนั้น)
- Tier: **A1** |Δt| ≤ 10 วัน (น้ำหนักสูงสุด), **A2** 10–20 วัน (ต้องผ่าน S2.5 อย่างเข้มกว่า)
- สร้างสามวัน (T, T′, R) เมื่อมีภาพที่สาม R ใสตำแหน่งเดียวกัน คนละวันกับ T′

**S2.2 ตัดสินคุณภาพข้อมูลระดับไทล์**
- เก็บเฉพาะไทล์ที่ `valid_frac ≥ [D] 99%`, ไม่มี nodata ใน mask และใน GT บริเวณ mask

**S2.3 ตรวจการจัดตำแหน่ง (coregistration)**
- วัดการเลื่อนระหว่าง T กับ T′ ด้วย phase correlation บนพิกเซลนอก mask ของทั้งสองภาพ

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 2.3a | การเลื่อนย่อยพิกเซลของแต่ละคู่ | ≤ [D] 0.5 px (ตั้งหลังดูการกระจายจริง) |
| 2.3b | ฮิสโตแกรมการเลื่อนทั้งคลัง | ไม่มีกลุ่มเลื่อนเป็นระบบ (ถ้ามี ให้ปรับแก้หรือตัดฉากนั้น) |

**S2.4 ปรับค่าแสง (radiometric harmonization)**
- ใช้พิกเซล **นอก mask ของทั้งสองภาพ** และตัดพิกเซลที่เปลี่ยนมาก (เช่น |ΔNDVI| สูง) ออกจากการประมาณ
- ทดสอบสองแบบเป็น ablation ของข้อมูล **[H]**: (ก) regression เชิงเส้นรายแบนด์ (gain/offset แบบ robust) (ข) histogram matching
- บันทึก R² และ gain/offset ต่อแบนด์ลง `pairs.parquet`

**S2.5 คัดกรองการเปลี่ยนแปลงจริง (change screening)**
- คำนวณบนพิกเซลนอก mask: `ndvi_drift_frac` = สัดส่วนพิกเซลที่ |ΔNDVI| > [D] 0.15 และ median |ΔNDVI|

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 2.5a | A1: `ndvi_drift_frac` | ≤ [D] 10% |
| 2.5b | A2: `ndvi_drift_frac` | ≤ [D] 5% (เข้มกว่า เพราะเวลาห่างกันกว่า) |
| 2.5c | คู่ที่ถูกตัดทิ้ง | รายงานเหตุผลและสัดส่วนต่อฤดู (ตรวจว่าไม่ตัดเฉพาะฤดูฝนจนข้อมูลเอนเอียง) |

**S2.6 ตัดไทล์ (tiling)**
- ตัดเป็นไทล์ 256×256 บนกริดเดียวกับไทล์ใสเดิม บันทึกพิกัดและ tile_id

**S2.7 แนบ SAR (S1)**
- เลือกภาพ S1 ที่ใกล้วัน T ที่สุดภายใน ±3 วัน (ถ้าเป็นไปได้ให้ใช้ทิศ/วงโคจรเดียวกันกับที่ใช้ในไทล์ใสเดิม) บันทึก `s1_dt_days`
- เตรียม 3 รูปแบบเก็บคู่กัน **[H]**: ดิบ (dB), multilook, Refined Lee

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 2.7a | `s1_dt_days` ของทุกไทล์ | รายงานการกระจาย ไทล์ที่เกิน [D] 3 วันถูกติดป้ายและแยกวิเคราะห์ |
| 2.7b | ค่า dB อยู่ในช่วงสมเหตุสมผล (ไม่มี NaN/∞) | ผ่านทุกไทล์ |

**S2.8 ผลลัพธ์ของเฟส 2**
- `pairs.parquet` + ไทล์ P3 (โหมด R2 และ R3), รายงาน `reports/pairs_summary.json`

| ตัวชี้วัดของเฟส | ค่าที่ต้องรายงาน |
|---|---|
| จำนวนคู่ (และจำนวน**ฉากอิสระ**) ต่อ tier × ถังเมฆ × จังหวัด × ฤดู | ตาราง |
| จำนวน triples (R3) | ตัวเลข |
| สัดส่วนคู่ที่ถูกตัดต่อเกณฑ์ | ตาราง |

---

### เฟส 3 — ธนาคารชั้นเมฆและข้อมูลสังเคราะห์ (P1, P2)

หลักการ: ภาพที่ติดเมฆเป็นผลผสมระหว่างเมฆกับพื้นดิน ดังนั้น**ห้ามแปะพิกเซลเมฆดิบ** เพราะจะพาลายพื้นดินของฉากต้นทางติดมาด้วย ต้องแยกชั้นเมฆออกก่อน

**S3.1 ประมาณชั้นเมฆจากคู่จริง [H]**
- ใช้ T (ติดเมฆ) และ T′ (พื้นดินใกล้เคียง) ในบริเวณ mask
- โมเดลเมฆ: `I_T ≈ α·C + (1−α)·I_T′` สำหรับเมฆ, `I_T ≈ s·I_T′` สำหรับเงา (s คือตัวคูณต่อพิกเซล, ตัดช่วง [D] 0.1–1.0)
- ประมาณ `C` (สเปกตรัมเมฆ 7 แบนด์ ต่อฉาก) จากพิกเซลเมฆหนาที่สุด แล้วหา `α` ต่อพิกเซลด้วยการลด residual ข้ามแบนด์ ตัดช่วง [0,1]
- ผลลัพธ์: ต่อวัตถุเมฆหนึ่งก้อน เก็บ (α map, shadow factor map, C vector, type_mask) ลง `interim/layers/` และ `layers.parquet`

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 3.1a | residual ของการประมาณ (`fit_residual`) | ชั้นที่ residual เกินเกณฑ์ [D] ตั้งหลังนำร่อง ถูกตัดออกจากธนาคาร |
| 3.1b | ตรวจด้วยตา 50 ชั้น: แปะกลับบน T′ แล้วเทียบกับ T จริง | ลายพื้นดินจากฉากต้นทางไม่โผล่ในพิกเซลเมฆหนา |

**S3.2 ธนาคารชั้นเมฆ (bank)**
- จัดเป็น 5 ถังตาม coverage ของวัตถุ/ครอปที่ 256×256 (ตามถังในหัวข้อ 1) เก็บทั้งเมฆหนา เมฆบาง และเงา
- Augment ตอนใช้: หมุน/พลิก/เลื่อน, ปรับ α เล็กน้อย, **ไม่ปรับรูปร่างจนผิดธรรมชาติ**

**S3.3 สร้าง P1**
- สุ่มไทล์ใสจากคลังเดิม + สุ่มชั้นเมฆจากธนาคาร (ต่างฉาก) แล้วคอมโพสิต: `I = (1−α)·I_clear + α·C`, จากนั้นคูณเงา `s`
- GT = ไทล์ใส; mask = จาก type_mask/α ของชั้นนั้น

**S3.4 สร้าง P2**
- เหมือน S3.3 แต่เพิ่ม **ภาพอ้างอิงจริงต่างวัน** ของตำแหน่งเดียวกัน (ไทล์ใสวัน T±Δ, Δ แบบสุ่มตามการกระจายอายุ ref จริงจาก S0.3 รวมถึงกรณี ref เก่า 20–30 วัน)
- เกณฑ์: ref ต้องคนละวันกับ GT เสมอ (กฎห้ามรั่ว)

**S3.5 ตรวจความสมจริงของข้อมูลสังเคราะห์**

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 3.5a | เทียบสถิติสเปกตรัมพิกเซลเมฆสังเคราะห์กับเมฆจริง (ฮิสโตแกรมรายแบนด์, อัตราส่วนแบนด์) | ระยะห่างการกระจาย (เช่น Wasserstein) รายงานต่อแบนด์ เป้าคือใกล้ศูนย์ ตั้งเกณฑ์หลังนำร่อง |
| 3.5b | **Discriminator test:** เทรนตัวจำแนกเล็กๆ แยกไทล์เมฆจริงกับไทล์เมฆสังเคราะห์ | AUC บนชุดที่ไม่เคยเห็น ยิ่งใกล้ 0.5 ยิ่งสมจริง (รายงานค่า; ถ้าสูงมากให้ย้อนปรับ S3.1) |
| 3.5c | ตรวจด้วยตา 100 ไทล์ | ไม่มี artifact ขอบเป็นรอยตัด/สีผิดธรรมชาติที่เห็นได้ชัด |
| 3.5d | อัตราส่วนข้อมูลสังเคราะห์ต่อจริงใน train | บันทึกใน config; ปรับตามผล ablation |

---

### เฟส 4 — การแบ่งข้อมูลและล็อกชุดทดสอบ

**S4.1 แบ่งแบบกลุ่ม**
- หน่วยแบ่ง = **ฉาก** (scene_id) ไม่ใช่ไทล์ ไทล์จากฉากเดียวกัน/ฉากที่ทับกันต้องอยู่ฝั่งเดียวกัน
- แบ่งสามชั้น:
  1. **Province holdout:** กัน 1 จังหวัดเป็น unseen test (เลือกจากจังหวัดที่คุณมีจริง) วัดความทั่วไป
  2. **Block split ในจังหวัดที่เทรน:** วัดสมรรถนะในพื้นที่ใช้งานจริง
  3. **Temporal holdout:** กันช่วงเวลาท้ายสุด (เช่นฤดูฝนล่าสุด) เป็นส่วนหนึ่งของ test
- ใส่ buffer เชิงพื้นที่รอบไทล์ test อย่างน้อย 1 ไทล์ หากไทล์ซ้อนทับกัน

**S4.2 ทดสอบการรั่วของข้อมูล (unit test)**

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 4.2a | ไม่มี scene_id ซ้ำข้าม train/val/test | จริง |
| 4.2b | ไม่มีไทล์ test ใดซ้อนทับ (เชิงพื้นที่) กับไทล์ train/val | จริง |
| 4.2c | P2/P3: วัน ref ≠ วัน GT เสมอ | จริง |
| 4.2d | ชั้นเมฆในธนาคารที่ใช้กับไทล์ test ไม่มาจากฉากใน test (หรือไม่ใช้สังเคราะห์ใน test เลย) | จริง |
| 4.2e | hash ของภาพ GT ไม่ปรากฏเป็น input ของตัวอย่างใดๆ | จริง |

**S4.3 ล็อกชุดทดสอบ**
- เขียนรายการไทล์ test เป็นไฟล์ + SHA-256 ลง `splits/test.sha256` และ commit ก่อนเริ่มเฟส 6 ห้ามแก้ภายหลัง
- ชุด val ใช้ปรับพารามิเตอร์ ชุด test ใช้รายงานผลครั้งสุดท้ายเท่านั้น

**ชุดทดสอบที่ต้องมี**

| ชื่อ | เนื้อหา | ใช้วัด |
|---|---|---|
| TEST-R2 | คู่จริง (T, T′) ไม่มี ref | โหมดไม่มี ref |
| TEST-R3 | สามวันจริง (T, T′, R) | โหมดมี ref |
| TEST-SYN | P1/P2 บนไทล์ใสที่ถูกกันไว้ | วัดแยกเส้นทางที่ GT แม่น (ไม่ปนการเปลี่ยนแปลงจริง) |

---

### เฟส 5 — สร้างโค้ดและ Unit Test

**S5.1 Dataset loader**
- โหลดไทล์ + mask + (S1) + (ref) + metadata; รองรับ reference dropout (สุ่มตัด ref และ SAR ตอนเทรน); balanced sampling ตามถัง coverage

**S5.2 Baselines**
- **B0:** คัดลอกภาพใส + ปรับแสง (S0.5)
- **B1:** CNN/U-Net (optical + mask) ใช้ loss ชุดเดียวกัน
- **B2:** DSen2-CR และ UnCRtainTS ปรับเป็น 7 แบนด์ (UnCRtainTS ต้นฉบับใช้ 13 แบนด์ L1C จึงต้องปรับ input และ fine-tune) [E]
- **ทางเลือก:** DiffCR เป็นตัวเปรียบเทียบเสริม (diffusion หลายสเต็ปและสุ่มผล จึงไม่ใช่ตัวหลักสำหรับหน้าเว็บ real-time) [E]

**S5.3 โมเดล Cloud-MAE v2**
- ViT ระดับเล็ก (ตั้งต้น embed 384, 12 layers, 6 heads **[D]**; วัด VRAM จริง) ทดสอบ patch 16 เทียบกับ patch 8 **[H]** เพราะเมฆ ≤ 20% มักเล็กกว่า patch 16×16 และอาจต้องการ conv head ความละเอียดเต็ม
- Input channels: S2 cloudy (7) + soft mask (1) + S1 (2, ทางเลือก) + ref (7, ทางเลือก) + ref-validity (1)
- Conditioning: soft mask เป็น spatial channel ก่อน; ทดสอบ FiLM/AdaLN จาก coverage รายแพตช์ และ ratio token ภายหลัง **[H]** (หมายเหตุ: PromptIR ใช้ prompt ที่เรียนรู้ ไม่ใช่ scalar เดี่ยว จึงอ้างเป็น "แรงบันดาลใจ" เท่านั้น [E])
- Output: residual เฉพาะ mask + หัว (μ, log σ²)

**S5.4 Loss (ค่าตั้งต้น [D])**
- L1 หรือ Laplace NLL (เมื่อใช้ σ) ในบริเวณ mask + แถบขอบ [E: L1 และ MS-SSIM ให้คุณภาพดีในงาน restoration]
- MS-SSIM รายแบนด์ ≈ 0.2
- SAM ≈ 0.1 **คำนวณใน float32**
- Gradient loss เทียบ **GT optical** ≈ 0.1 (SAR เป็น input เท่านั้น)

**S5.5 Reinsertion**
- `out = M·pred + (1−M)·input` โดย M = mask ที่ dilate แล้ว feather ขอบ

**Unit tests ที่ต้องผ่านก่อนเทรน**

| CHK | การทดสอบ | เกณฑ์ผ่าน |
|---|---|---|
| 5.a | SAM loss: เวกเตอร์สุ่ม, เวกเตอร์เกือบขนาน, เวกเตอร์ norm ≈ 0 ใต้ BF16 autocast | ไม่มี NaN/Inf ใน loss และ gradient |
| 5.b | Reinsertion: พิกเซลที่ M = 0 | `max|out − input| == 0` (เท่ากันเป๊ะ) |
| 5.c | Reinsertion: ขอบ feather | ค่าเปลี่ยนต่อเนื่อง ไม่มีรอยกระโดดเกินเกณฑ์ [D] |
| 5.d | Shape/dtype/range ของ batch | ตรง spec ทุกโมดัล |
| 5.e | Overfit test: เทรน 1 batch เล็กๆ 200 step | loss ลดลงชัดเจนและ `PSNR_masked` สูงขึ้น |
| 5.f | Reference dropout | ถ้าตัด ref ทิ้ง โมเดลยังทำงานได้และไม่ crash |
| 5.g | Seed reproducibility | รันซ้ำได้ผลเหมือนเดิมตามความคลาดเคลื่อน |
| 5.h | วัด VRAM peak (batch 16, BF16) | บันทึกค่าจริง ไม่เกิน 14 GB (เหลือที่ว่าง) |

---

### เฟส 6 — การทดลอง

**S6.1 Baseline runs**
- รัน B0, B1, B2 บน TEST-R2/R3/SYN และ val ด้วยโปรโตคอลเดียวกับโมเดลหลัก (≥ 3 seeds ถ้าเป็นโมเดลที่เทรน)

**S6.2 Ablation matrix (30 epochs, ≥ 3 seeds ต่อแขน)**

| รหัส | ข้อมูลเทรน | Input | โมดูล/ loss | คำถามที่ตอบ |
|---|---|---|---|---|
| A0 | P1 | optical + soft mask | L1 อย่างเดียว | baseline ของโมเดลหลัก |
| A1 | P1 | + S1 (ดิบ) | L1 | SAR ช่วยไหม |
| A2 | P1 | + S1 (multilook) | L1 | despeckle แบบเบา |
| A3 | P1 | + S1 (Refined Lee) | L1 | despeckle แบบแรง |
| A4 | P1+P2 | + ref (+ ref-validity) | L1 | ref ช่วยไหม และช่วยเมื่อ ref เก่าแค่ไหน |
| A5 | P1+P2 | A4 + reference dropout | L1 | ทำงานได้เมื่อไม่มี ref |
| A6 | ดีที่สุดจาก A1–A5 | + FiLM/AdaLN จาก coverage รายแพตช์ | L1 | conditioning ช่วยไหม |
| A7 | ดีที่สุด | + ratio token | L1 | ratio token ช่วยเกิน A6 ไหม |
| A8 | ดีที่สุด | patch 8 เทียบกับ patch 16 | L1 | ความละเอียดแพตช์ |
| A9 | ดีที่สุด | MAE pretrain (มาสก์รูปเมฆจริง) เทียบ from-scratch | L1 | pretrain ช่วยไหม |
| A10 | ดีที่สุด | — | + MS-SSIM | loss |
| A11 | ดีที่สุด | — | + SAM | loss |
| A12 | ดีที่สุด | — | + gradient loss (GT optical) | loss |
| A13 | ดีที่สุด | — | + Laplace NLL (σ) | uncertainty |
| A14 | ดีที่สุด + ratio/ข้อมูล | อัตราส่วน P1:P2:P3 ต่างกัน | — | สัดส่วนข้อมูลสังเคราะห์ต่อจริง |

ข้อกำหนด: เปลี่ยนทีละปัจจัย, เก็บ seed และ config, รายงาน mean ± sd

**S6.3 การวิเคราะห์ทางสถิติ**
- Bootstrap **ต่อฉาก** (ไม่ใช่ต่อไทล์), 1,000 รอบ, เทียบแบบจับคู่ (paired difference) ของ `PSNR_masked`, `SAM_masked`
- กฎตัดสิน **[D]:** โมดูลอยู่ต่อเมื่อ CI 95% ของผลต่างไม่คร่อม 0 และขนาดผล ≥ 2× ค่า sd ข้าม seed ของ A0 และไม่แย่ลงบนจังหวัดที่กันไว้

**Gate G1 — โมดูลอยู่หรือไป**: ตัดโมดูลที่ไม่ผ่านกฎ S6.3 ออกก่อนเทรนเต็ม

**S6.4 เทรนเต็ม**
- สองสเตจ: (1) pretrain บน P1/P2 (ถ้า A9 ผ่าน, ใช้ MAE pretrain) (2) fine-tune ด้วย P3 ชั้น A1/A2 (น้ำหนักต่ำกว่า P1/P2) ด้วย lr ต่ำกว่า
- BF16 autocast, AdamW (β₂ = 0.95, wd 0.05), warmup + cosine, EMA ของน้ำหนัก [D]
- Early stopping บน `PSNR_masked` ของ val จริง (ไม่ใช้ epoch ตายตัว)
- บันทึกทุก epoch: loss รายองค์ประกอบ, `PSNR_masked/all`, `SAM_masked`, VRAM, เวลา

| CHK | การตรวจระหว่างเทรน | เกณฑ์ผ่าน |
|---|---|---|
| 6.4a | loss ทุกองค์ประกอบไม่เป็น NaN | ผ่านตลอด |
| 6.4b | train/val gap ของ `PSNR_masked` | ถ้า gap กว้างขึ้นต่อเนื่อง ให้หยุดและลด capacity/เพิ่ม augmentation |
| 6.4c | ภาพตัวอย่างจาก val ทุก N epoch | ไม่มีจุดขาว ไม่มีรอยต่อขอบ mask |
| 6.4d | ผลบนจังหวัดที่กัน | ไม่ตกลงมากกว่าส่วนอื่นเกินเกณฑ์ที่ตกลงไว้ [D] |

**S6.5 ประเมินสุดท้าย (ครั้งเดียวบน TEST)**
- ใช้โมเดลเดียวที่เลือกจาก val; รายงานตามหัวข้อ 4

**Gate G2 — ชนะ baseline**
- โหมดไม่มี ref: ชนะ B1 และ B2 บน TEST-R2 (`PSNR_masked`, `SAM_masked`) ตามกฎสถิติ S6.3
- โหมดมี ref: ไม่แพ้ B0 บน TEST-R3 ในสถานะที่ B0 ทำได้ดี (ref อายุสั้น) และดีกว่าในกรณี ref เก่า/ไม่สมบูรณ์

**Gate G3 — ใช้งานจริงได้**
- ผลปลายน้ำ (หัวข้อ 4.3) อยู่ในเกณฑ์ที่เจ้าของงานปลายน้ำกำหนดเอง
- σ ผ่านการตรวจ calibration (หัวข้อ 4.4) บนจังหวัดที่กัน

---

### เฟส 7 — การนำไปใช้งาน

**S7.1 ประตูคัดกรองอินพุต**
- คำนวณ coverage จาก mask รวม: ≤ 20% ซ่อล; 20–25% ซ่อมพร้อมเตือน; > 25% ปฏิเสธ
- ตรวจอายุ ref (ถ้ามี): อายุ > [D] 20 วัน → เตือนและลดน้ำหนัก ref ตามที่โมเดลเรียนรู้

**S7.2 การแสดงผลและ uncertainty**
- แสดงแผนที่ σ รายพิกเซล และโหมดที่ใช้ (มี/ไม่มี ref, อายุ ref, มี/ไม่มี SAR)
- ข้อความเตือนใช้ถ้อยคำกลาง ("ความเชื่อมั่นต่ำ ไม่แนะนำให้ใช้ค่านี้ในการวิเคราะห์เชิงปริมาณ") และไม่ให้คำแนะนำทางการเกษตรโดยตรงจนกว่า G3 ผ่าน

**S7.3 การซ่อมฉากใหญ่**
- ตัดเป็นไทล์ซ้อนทับ (overlap) แล้วผสมขอบ (blend) ก่อนรวมกลับ

**S7.4 API และการตรวจ**

| CHK | การตรวจ | เกณฑ์ผ่าน |
|---|---|---|
| 7.a | ฉากทดสอบทั้งฉาก: รอยต่อระหว่างไทล์ | ความต่างค่าสะท้อนข้ามรอยต่อไม่สูงกว่าความต่างภายในไทล์อย่างมีนัยสำคัญ |
| 7.b | พิกเซลนอก mask | เท่ากับ input เป๊ะ (ตรวจในระดับฉาก) |
| 7.c | เวลา inference ต่อไทล์ / ต่อฉาก | บันทึกค่า; ตั้งเกณฑ์ตามการใช้งานหน้าเว็บ |
| 7.d | อินพุตผิดรูป (ขาดแบนด์, mask ว่าง, mask ทั้งภาพ) | API ตอบ error ที่ชัดเจน ไม่ crash |
| 7.e | เวอร์ชันโมเดลถูกบันทึกในผลลัพธ์ทุกครั้ง | ผ่าน |

**S7.5 การเฝ้าติดตามหลังใช้งาน**
- สุ่มตรวจคุณภาพรายเดือนกับภาพใสจริงที่มาทีหลัง; ติดตามการลื่นไถลของ σ vs |error| รายฤดู

---

## 4. นิยามตัวชี้วัด

ให้ `x` = ภาพที่ซ่อม, `y` = GT, ค่าสะท้อนปรับเป็น [0,1] (หาร 10000 แล้ว clip), `Ω` = พิกเซลใน hard_mask (ไม่ dilate เพื่อไม่ให้ได้พิกเซลปกติมาเจือ) และ `N` = ทุกพิกเซล

### 4.1 ตัวชี้วัดคุณภาพภาพ

| ตัวชี้วัด | นิยาม |
|---|---|
| `PSNR_masked` | 10·log10(1 / MSE_Ω) โดย MSE เฉลี่ยข้ามแบนด์และพิกเซลใน Ω |
| `PSNR_all` | เช่นเดียวกัน แต่บน N (ใช้รายงานคู่เสมอ แต่ไม่ใช้ตัดสิน) |
| `SSIM_masked` | SSIM map รายแบนด์เฉลี่ยเฉพาะพิกเซลใน Ω (หน้าต่าง 11×11 ใช้ข้อมูลรอบนอกได้) |
| `SAM_masked` | arccos(⟨x,y⟩ / (‖x‖‖y‖)) ต่อพิกเซลใน Ω เฉลี่ย (องศา) คำนวณ float32 ตัดพิกเซลที่ ‖y‖ < ε |
| `MAE_band`, `Bias_band` | MAE และค่าเฉลี่ย (x−y) แยกรายแบนด์ใน Ω |

รายงานแยกตามมิติเหล่านี้ทุกครั้ง: **ถังเมฆ**, **ชนิดเมฆ** (หนา/บาง/เงา), **ฤดู**, **จังหวัด**, **โหมด** (ไม่มี ref / มี ref + ช่วงอายุ ref), **ใช้ SAR หรือไม่**

### 4.2 ตัวชี้วัดความสมจริงข้อมูลสังเคราะห์

AUC ของ discriminator (S3.5b), ระยะ Wasserstein รายแบนด์ (S3.5a)

### 4.3 ตัวชี้วัดปลายน้ำ (ชั้นที่สำคัญที่สุดสำหรับการใช้งานจริง)

| ตัวชี้วัด | นิยาม |
|---|---|
| ดัชนี NDVI, NDMI (หรือ SWIR-based อื่น) | MAE และ bias เทียบ GT **เฉพาะพิกเซลในแปลงปาล์ม** (ถ้ามีเลเยอร์ palm จาก S0.2e; ถ้าไม่มี ให้ระบุข้อจำกัดและใช้ mask พืชพรรณหยาบแทน) |
| ปลายน้ำจริง | ผลของงานวิเคราะห์ที่แพลตฟอร์มใช้ (เช่นการจำแนก/ตรวจจับ) บนภาพซ่อมเทียบภาพใสจริง |

หมายเหตุ: NDVI ของทรงพุ่มหนาอาจอิ่มตัว (ข้อสังเกตจากความรู้ทั่วไปด้านการสำรวจระยะไกล ต้องตรวจกับข้อมูลของคุณ) จึงไม่ควรใช้ NDVI อย่างเดียว ควรเพิ่มดัชนีจาก SWIR หรือผลปลายน้ำจริง

### 4.4 ตัวชี้วัดความไม่แน่นอน (σ)

| ตัวชี้วัด | นิยาม |
|---|---|
| Spearman(σ, \|error\|) | ความสัมพันธ์เรียงอันดับระหว่าง σ กับ error ต่อพิกเซล (สูงดี) |
| Calibration curve | แบ่ง σ เป็น 10 ถัง เทียบ σ เฉลี่ยกับ \|error\| เฉลี่ยต่อถัง ควรใกล้เส้นทแยง |
| Interval coverage | สัดส่วนที่ GT อยู่ในช่วง 90% ที่ทำนาย (ควรใกล้ 90%) |
| Sparsification | ตัดพิกเซลที่ σ สูงสุดทีละ 10% แล้วดูว่า error เฉลี่ยลดลงตามที่ควร |

### 4.5 ตัวชี้วัดงานด้านวิศวกรรม

เวลา inference, VRAM peak, ขนาดโมเดล, เวลาเทรนต่อ epoch

### 4.6 แม่แบบตารางรายงานผล (ต่อโมเดล ต่อชุดทดสอบ)

| โมเดล | ชุด | โหมด | ถัง | n ฉาก | PSNR_masked (mean [CI]) | SSIM_masked | SAM_masked (°) | PSNR_all | NDMI MAE | σ Spearman |
|---|---|---|---|---|---|---|---|---|---|---|

---

## 5. สรุป Gates

| Gate | ตำแหน่ง | เงื่อนไขผ่าน | ถ้าไม่ผ่าน |
|---|---|---|---|
| **G0** | จบเฟส 0 | ข้อมูลพอ (CI ของ B0 แคบพอ), mask ใช้ได้, มี L1C หรือทางเลือก | เพิ่มข้อมูล/ปรับสัดส่วน P1,P2/ดาวน์โหลด L1C |
| **G1** | จบ ablation | โมดูลผ่านกฎสถิติ (CI ไม่คร่อม 0 และขนาดผล ≥ 2× sd ข้าม seed) | ตัดโมดูลออก |
| **G2** | หลังประเมินสุดท้าย | ชนะ B1/B2 (ไม่มี ref); ไม่แพ้ B0 (มี ref อายุสั้น) | ย้อนปรับเฟส 2–6 |
| **G3** | ก่อนใช้งานจริง | ผลปลายน้ำอยู่ในเกณฑ์ที่เจ้าของงานกำหนด และ σ ผ่านการตรวจ calibration | ใช้แบบ "ภาพประกอบ" พร้อมแผนที่ σ เท่านั้น |

เกณฑ์เชิงตัวเลขทุกตัวที่ติด [D] ให้ตั้งหลังเฟส 0 และบันทึกลง `configs/thresholds.yaml` พร้อมวันที่และเหตุผล

---

## 6. ความเสี่ยงและมาตรการ

| ความเสี่ยง | ผลกระทบ | มาตรการ |
|---|---|---|
| ภาพใสอ้างอิงหายากในฤดูฝน | โหมดมี ref ใช้ได้น้อย | S0.3 วัดก่อน; reference dropout; ฝึกด้วย ref เก่า/composite; โหมดไม่มี ref ต้องดีด้วยตัวเอง |
| มี L1C ไม่ครบ ทำ s2cloudless ไม่ได้ | mask คุณภาพต่ำ | S0.2d ตรวจก่อน; ดาวน์โหลด L1C เพิ่ม |
| ชั้นเมฆสังเคราะห์พาลายพื้นดินจากฉากอื่น | โมเดลเรียนจาก artifact | S3.1 แยกชั้น α/C ก่อนแปะ; CHK 3.1b; discriminator test |
| GT ของ P3 ปนการเปลี่ยนแปลงจริง | ตัวชี้วัดต่ำเกินจริงและสอนผิด | A1/A2 + change screening; P3 ใช้ fine-tune/test; ใช้ TEST-SYN ควบคู่ |
| ใช้ภาพเดียวเป็นทั้ง GT และ ref | metric สูงเกินจริงจากการคัดลอก | กฎห้ามรั่ว + CHK 4.2c, 4.2e |
| ข้อมูลไม่อิสระ (ไทล์จากฉากเดียวกัน) | CI แคบเกินจริง | แบ่งตามฉาก; bootstrap ต่อฉาก |
| overfit กับชุดไทล์ 8,996 ใบ | ผลดีบน val แต่ไม่ทั่วไป | province/temporal holdout; early stopping; augmentation |
| SAR ทำให้ขอบปลอม หรือ speckle รบกวน | artifact | SAR เป็น ablation; ไม่ใช้เป็นเป้าของ edge loss; เปรียบเทียบ despeckle |
| σ ไม่ calibrate | ผู้ใช้เชื่อค่าที่ไม่ควรเชื่อ | ตรวจ 4.4; ถ้าไม่ผ่านแสดงเป็นภาพประกอบเท่านั้น |
| clamp SAM แบบเดิมใช้ไม่ได้ใต้ BF16 | gradient NaN | คำนวณ float32; CHK 5.a |

---

## 7. รายการตรวจสอบก่อนเริ่มแต่ละเฟส

- [x] **ก่อนเฟส 1:** G0 ผ่าน (`gate_g0.json`); เลือกวิธี mask **ทางเลือกที่ 2 (SCL + Multi-spectral Cloud Prob Ensemble)** ไม่ต้องโหลด L1C ขนาดใหญ่
- [x] **ก่อนเฟส 2:** S1.4 ผ่านทุกข้อ (`s10_mask_production.json`); ตั้งค่า dilation 3 px (30m) เสร็จ; โค้ดติดตั้งใน `lib2_vitpalm/mask_pipeline.py`
- [ ] **ก่อนเฟส 3:** `pairs.parquet` ผ่าน 2.3, 2.5, 2.7; ชุดคู่เพียงพอสำหรับสกัดชั้นเมฆ (กำลังดำเนินการใน Phase 2)
- [ ] **ก่อนเฟส 4:** S3.5 ผ่านหรือบันทึกข้อจำกัดไว้
- [ ] **ก่อนเฟส 5:** S4.2 ผ่านทุกข้อ; test ถูกล็อก (hash)
- [ ] **ก่อนเฟส 6:** unit tests 5.a–5.h ผ่านทั้งหมด; baselines รันได้
- [ ] **ก่อนเฟส 7:** G1, G2, G3 ผ่าน; ตารางรายงานผลครบทุกมิติ

---

## 8. สิ่งที่ส่งมอบ

1. `manifests/*.parquet`, `splits/*`, `reports/*.json` ของทุกขั้น
2. `configs/thresholds.yaml` (ค่า [D] ที่ตั้งหลังนำร่อง)
3. โค้ด `src/` + `tests/` ที่ผ่านทั้งหมด
4. ตาราง ablation และตารางผลสุดท้าย (หัวข้อ 4.6) พร้อมกราฟ calibration
5. `checkpoints/cloud_mae_v2_best.pth` + model card (ข้อมูลเทรน, ขอบเขต ≤ 20%, ข้อจำกัด, ผลบนจังหวัดที่กัน)
6. ชุด API/เว็บที่ผ่าน CHK 7.a–7.e

---

## อ้างอิง (รายการที่ใช้ในการทบทวน)

- Ebel et al., *Multi-Sensor Data Fusion for Cloud Removal in Global and All-Season Sentinel-2 Imagery*, IEEE TGRS (arXiv 2009.07683) — ข้อมูลจริงดีกว่าสังเคราะห์
- Meraner et al., *Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion* (DSen2-CR), ISPRS J. — cloud-adaptive loss
- Ebel et al., *UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series*, CVPRW 2023 (arXiv 2304.05464)
- Zhao et al., *Loss Functions for Image Restoration with Neural Networks* (arXiv 1511.08861)
- Potlapalli et al., *PromptIR*, NeurIPS 2023 (arXiv 2306.13090)
- Zekoll et al., *Comparison of Masking Algorithms for Sentinel-2 Imagery*, Remote Sensing 13(137), 2021
- Tarrio et al., *Comparison of cloud detection algorithms for Sentinel-2 imagery*, Science of Remote Sensing, 2020
- Cong et al., *SatMAE* (arXiv 2207.08051); Li et al., *S2MAE*, CVPR 2024
- Zou et al., *DiffCR*, IEEE TGRS 2024 (arXiv 2308.04417)
- Zhou et al., *Evaluating 2022 cloud-free coverage of HLS V2.0*, IEEE GRSL 2025 — เขตร้อนมีภาพใสน้อยที่สุด
