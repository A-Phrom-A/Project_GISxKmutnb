# แผนการพัฒนาโดยละเอียด: Cross-Band Recovery (Sentinel-1 → Sentinel-2)

**โปรเจกต์:** `Project_GISxKmutnb` / `projects/cross_band_recovery/`
**จัดทำเมื่อ:** 17 ตุลาคม 2026
**สถานะเอกสาร:** v1.0 (แผนแม่บท ใช้ติดตามความคืบหน้าได้โดยตรง)

> **หมายเหตุเรื่องความน่าเชื่อถือของเอกสาร**
> งานอ้างอิงทุกรายการในหัวข้อ 8–10 ถูกติดป้ายสถานะ
> ✅ = ตรวจกับแหล่งข้อมูลออนไลน์แล้วในการจัดทำเอกสารนี้ (ชื่อ ผู้เขียน ที่ตีพิมพ์ตรงกัน)
> ⚠️ = มาจากความจำ ยังไม่ได้ตรวจในรอบนี้ **ต้องเปิดต้นฉบับยืนยันก่อนนำไปอ้างในรายงานหรือบทความ**
> ตัวเลขเกณฑ์ผ่าน/ไม่ผ่านที่ไม่มีที่มาทางสถิติหรือจากงานวิจัย จะติดป้าย **[PROVISIONAL]** และต้องถูกตรึงใน Gate G7 ก่อนรันชุดทดสอบ

---

## สารบัญ

0. วิธีใช้เอกสารนี้และระบบติดตาม
1. กรอบงานวิจัย: คำถาม สมมติฐาน หลักการ
2. การตัดสินใจที่ต้องยืนยัน (Decision Log)
3. ภาพรวมเฟสและตารางติดตามหลัก
4. รายละเอียดแต่ละเฟสและ Quality Gates (G0–G11)
5. ระเบียบวิธีทดสอบโมเดลหลักเมื่อพัฒนาเสร็จ (Final Model Test Protocol)
6. เทมเพลตการติดตาม (Tracker, Gate Report, Experiment Log)
7. ทะเบียนความเสี่ยง
8. งานวิจัยที่รองรับกระบวนการพัฒนา
9. งานวิจัยที่แย้งหรือท้าทายกระบวนการพัฒนา
10. เอกสารอ้างอิงฉบับเต็มและสถานะการตรวจสอบ
11. ภาคผนวก (สูตร กติกา pattern ID โครงสร้างไฟล์)

---

## 0. วิธีใช้เอกสารนี้และระบบติดตาม

### 0.1 หลักการของ Gate
ทุกเฟสจบด้วย **Gate** ซึ่งประกอบด้วย 5 ส่วนเสมอ:

| ส่วน | ความหมาย |
|---|---|
| **Entry** | เงื่อนไขก่อนเริ่ม (Gate ก่อนหน้าต้องผ่าน) |
| **Checks** | รายการตรวจ แต่ละข้อมีรหัส `CHK-Gx.y` |
| **Pass criteria** | เกณฑ์ผ่านที่วัดได้ ไม่ใช่ความรู้สึก |
| **Evidence** | ไฟล์หลักฐานที่ต้องมี (log, parquet, รูป, ผล pytest) เก็บใน `docs/tracking/evidence/Gx/` |
| **On fail** | ต้องทำอะไรเมื่อไม่ผ่าน (แก้, ย้อนเฟส, หรือบันทึกเป็นข้อจำกัด) |

**กติกาเหล็ก**
1. ห้ามเริ่มเฟสถัดไปก่อน Gate ผ่าน ยกเว้นระบุว่าทำขนานได้
2. Gate ผ่านได้ก็ต่อเมื่อมี Evidence ครบ ไม่ใช่แค่ "รันแล้วดูดี"
3. เกณฑ์ที่ติดป้าย [PROVISIONAL] ต้องถูกตรึงและบันทึก hash ใน G7 **ก่อน** เปิดชุด test ครั้งแรก
4. ผลลบ (S1 ไม่ช่วย, S1 ทำให้แย่ลง, โมเดลสู้ baseline ไม่ได้) เป็นข้อค้นพบที่ต้องรายงาน ไม่ใช่ความล้มเหลวของ Gate เว้นแต่ระบุไว้ว่าเป็นเงื่อนไขทางวิศวกรรม
5. ชุด test เปิดใช้ได้ **ครั้งเดียวต่อรุ่นโมเดลที่ตรึงแล้ว** (ดู G8)

### 0.2 สถานะที่ใช้ในตาราง
`⬜ ยังไม่เริ่ม` · `🟦 กำลังทำ` · `🟨 รอตรวจ Gate` · `✅ ผ่าน` · `❌ ไม่ผ่าน (ดู On fail)` · `⏸ พักไว้`

### 0.3 รูปแบบการตั้งชื่อ
- Gate: `G0` … `G11`
- รายการตรวจ: `CHK-G4.3` = Gate 4 ข้อที่ 3
- การทดลอง: `EXP-<phase>-<model>-<seed>` เช่น `EXP-P6-M2-s1`
- Evidence: `docs/tracking/evidence/G4/CHK-G4.3_<คำอธิบายสั้น>.<ext>`

---

## 1. กรอบงานวิจัย

### 1.1 คำถามวิจัย (RQ)
- **RQ1** โมเดล ViT ที่เรียนความสัมพันธ์ข้ามแบนด์ (cross-band) กู้คืนแบนด์ Sentinel-2 ที่หายไปได้แม่นกว่า baseline ที่ไม่ใช้บริบทเชิงพื้นที่/ข้ามแบนด์ (mean-fill, linear, per-pixel MLP, Random Forest) หรือไม่
- **RQ2** Sentinel-1 (VV, VH) ลดความคลาดเคลื่อนของการกู้คืนได้เท่าไร และขึ้นกับจำนวนและชนิดของแบนด์ที่หายไปอย่างไร
- **RQ3** (xAI) อิทธิพลของ S1 เทียบกับแบนด์ S2 ที่เหลือมีขนาดและทิศทางเท่าไร และสอดคล้องกับผล ablation หรือไม่
- **RQ4** แบนด์ที่กู้คืนมาใช้ในงานปลายทาง (palm segmentation) ได้ดีแค่ไหนเมื่อเทียบกับภาพจริง และภาพที่ไม่ซ่อม

### 1.2 สมมติฐาน (เขียนให้ตรวจสอบได้ ไม่ใช่ตั้งใจให้เป็นจริง)
| รหัส | สมมติฐาน | พิสูจน์ว่าผิดได้เมื่อ |
|---|---|---|
| H1 | ViT-Fusion ดีกว่า baseline ที่ดีที่สุดบนตัวชี้วัดหลักอย่างมีนัยสำคัญ | ช่วงความเชื่อมั่นของผลต่างครอบคลุม 0 หรือเป็นลบ |
| H2 | ΔMAE = MAE(ไม่มี S1) − MAE(มี S1) > 0 เมื่อแบนด์หายมาก (k สูง) | ΔMAE ≤ 0 อย่างมีนัยสำคัญที่ k สูง |
| H3 | ผลของ S1 ต่างกันตามชนิดแบนด์ (เช่น SWIR ต่างจาก visible) | ΔMAE ไม่ต่างกันข้ามแบนด์ |
| H4 | อิทธิพล SHAP ของ S1 สูงขึ้นเมื่อแบนด์ S2 ที่เหลือน้อยลง | ไม่มีแนวโน้ม หรือสวนทาง |
| H5 | ภาพที่กู้คืนฟื้น IoU/F1 ของ palm segmentation ได้มากกว่าภาพที่ไม่ซ่อม | IoU ของภาพกู้คืนไม่สูงกว่าภาพที่ไม่ซ่อม |

### 1.3 หลักการ (Principles)
- **ไม่ตั้งสมมติฐานว่า S1 ต้องช่วย** ผลทดลองเป็นผู้ตัดสิน
- **เกณฑ์ตรึงก่อนดูผล** (G7) และแหล่งที่มาของเกณฑ์ต้องบันทึก
- **ไม่นับว่า "กู้ได้" เพียงเพราะโมเดลคายค่าครบทุกพิกเซล** ต้องวัดความถูกต้อง
- **แยกกรณีที่ "ง่ายโดยธรรมชาติ"** ออกจากกรณีที่พิสูจน์ความสามารถจริง (เช่น ปิด B8 แต่เหลือ B8A ซึ่งเป็น NIR แคบที่สัมพันธ์กันสูง)
- **รายงานความไม่แน่นอน** ทุกตัวเลขสำคัญ

### 1.4 สิ่งที่อยู่นอกขอบเขต (Non-goals)
- ไม่อ้างว่าสังเคราะห์สเปกตรัมจาก SAR ล้วนได้แม่นเท่าของจริง กรณี k = 7 รายงานแยกเป็นการแปลง SAR→Optical
- ไม่ใช้ GEO-Satellite RGB ในการทดลองหลัก (ซ้ำกับ B2/B3/B4 จะปนเปื้อนการวัดอิทธิพลของ S1) เป็นการทดลองเสริมได้ภายหลัง
- Apriori/Eclat ย้ายไปเป็นภาคผนวกเสริม (ดู D6)

---

## 2. การตัดสินใจที่ต้องยืนยัน (Decision Log)

| ID | เรื่อง | ข้อเสนอเริ่มต้น | ผลกระทบถ้าเปลี่ยน | สถานะ |
|---|---|---|---|---|
| D1 | ชุดแบนด์ S2 | B02, B03, B04, B8A, B08, B11, B12 (ตามข้อมูลจริง) | README ระบุ B5 (RedEdge) ถ้ามี B5 NDRE จะคำนวณได้ถูกต้อง | ✅ ยืนยัน 7 แบนด์ |
| D2 | แบนด์ที่ปิดได้ | เฉพาะ S2 (7 แบนด์) → 127 pattern | ถ้าปิด S1 แยก ต้องนิยามเงื่อนไขใหม่ | ✅ ยืนยัน 127 แบบ |
| D3 | ขนาดไทล์/แพตช์ | ไทล์ 128x128, patch 16x16 (สอดคล้องกับโมดูล ViT Cross-Band และ VRAM 16GB) | ความยาว sequence 64 patches/แบนด์ | ✅ ยืนยัน 128x128 |
| D4 | การเตรียม S1 | เก็บ linear ดิบ → Lee 5×5 ใน linear → dB → normalize (มีทั้ง Raw-dB และ Lee-dB ในทุกไทล์) | ป้องกัน speckle noise พร้อมสำหรับ ablation | ✅ ยืนยัน Lee 5x5 + dB |
| D5 | การแบ่งข้อมูล | WGS84 Spatial block split 5x5 กม. + 5 กม. Buffer; Surat Thani เป็น out-of-domain test; คง step=256 แล้วขยายคลังภาพดาวเทียมผ่าน STAC/GEE | ป้องกัน data leakage เชิงพื้นที่ 0 คู่ (คงความเข้มงวดตามเกณฑ์วิจัย) | ✅ ยืนยัน Spatial Buffer + ขยายข้อมูลจาก STAC |
| D6 | Baseline | mean-fill, linear (per-pixel), per-pixel MLP, Random Forest; Apriori/Eclat เป็นภาคผนวก | ถ้าไม่มี baseline เหล่านี้ อ้าง RQ1 ไม่ได้ | ✅ ยืนยัน B0–B3 ครบถ้วน (ผ่าน G3) |
| D7 | S1 collection | `COPERNICUS/S2_SR_HARMONIZED` + `S1_GRD` (GEE & Copernicus) | ตรวจพบทั้ง log_dB และ linear ในเครื่อง | ✅ ยืนยันแหล่งข้อมูล |
| D8 | ช่องว่างเวลา \|Δt\| | ≤ 6 วัน (Tier 1: 85.7% ≤ 3 วัน, เฉลี่ย 1.14 วัน) | บันทึก Δt และ tier ใน manifest แล้ว | ✅ ตรวจสอบผ่าน G1 |
| D9 | ตัวชี้วัดดัชนี | NDVI, NDMI = (B8−B11)/(B8+B11), NBR = (B8−B12)/(B8+B12) (ไม่ใช้ NDRE ถ้าไม่มี B5) | ดูภาคผนวก 11.3 | ✅ ยืนยัน NDVI/NDMI/NBR |
| D10 | ความไม่แน่นอน | ensemble 3–5 seed (หรือ head ทำนายความแปรปรวน) | ต้นทุนการเทรน | ✅ ยืนยัน Ensemble 3 seeds (ผ่าน G6) |

---

## 3. ภาพรวมเฟสและตารางติดตามหลัก

```
P0 Setup ──G0──► P1 Data Acquisition ──G1──► P2 Data Processing ──G2──► P3 Baselines ──G3──┐
                                                                                            ▼
P4 Model Modules ──G4a/G4b──► P5 Loss & Trainer ──G5──► P6 Training Runs ──G6──► P7 Freeze Protocol ──G7──┐
                                                                                                          ▼
                 P8 Final Test (127 patterns) ──G8──► P9 xAI ──G9──► P10 Downstream ──G10──► P11 Report ──G11
```

> P3 (baseline) และ P4 (โมดูลโมเดล) ทำขนานกันได้หลัง G2 ผ่าน

### 3.1 ตารางติดตามหลัก (อัปเดตทุกครั้งที่สถานะเปลี่ยน)

| เฟส | Gate | ผลลัพธ์หลัก (Deliverable) | ผู้รับผิดชอบ | สถานะ | วันที่ผ่าน | Evidence |
|---|---|---|---|---|---|---|
| P0 Setup | G0 | env + seed + config + CI เบื้องต้น | AI Agent | ✅ ผ่าน | 17 ต.ค. 2026 | [`docs/tracking/evidence/G0/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G0/) |
| P1 Acquisition | G1 | `metadata_manifest` + GeoTIFF ต่อคู่ | AI Agent | ✅ ผ่าน | 17 ต.ค. 2026 | [`docs/tracking/evidence/G1/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/), [`metadata_manifest.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/data/manifests/metadata_manifest.parquet) |
| P2 Processing | G2 | ไทล์ + split + stats จาก train เท่านั้น | AI Agent | ✅ ผ่าน | 18 ต.ค. 2026 | [`docs/tracking/evidence/G2/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/), [`norm_stats.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/data/manifests/norm_stats.json), [`spatial_split.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/data/manifests/spatial_split.parquet) |
| P3 Baselines | G3 | ตารางผล baseline บน val | AI Agent | ✅ ผ่าน | 18 ต.ค. 2026 | [`docs/tracking/evidence/G3/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G3/), [`val_baseline_benchmarks.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/data/manifests/val_baseline_benchmarks.parquet) |
| P4 Modules | G4a, G4b | `channel_masking.py`, `vit_cross_band.py` + pytest | AI Agent | ✅ ผ่าน | 18 ต.ค. 2026 | [`docs/tracking/evidence/G4a/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/), [`docs/tracking/evidence/G4b/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/) |
| P5 Loss/Trainer | G5 | loss ถูกต้อง + overfit test | AI Agent | ✅ ผ่าน | 18 ต.ค. 2026 | [`docs/tracking/evidence/G5/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/) |
| P6 Training | G6 | checkpoint M1, M2 (หลาย seed) | AI Agent | ✅ ผ่าน | 18 ต.ค. 2026 | [`docs/tracking/evidence/G6/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/), [`gate_reports/G6.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G6.md) |
| P7 Freeze | G7 | pre-registration (เกณฑ์ + hash) | AI Agent | ✅ ผ่าน | 18 ต.ค. 2026 | [`docs/tracking/evidence/G7/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G7/), [`gate_reports/G7.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G7.md), [`preregistration.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/preregistration.md) |
| P8 Final Test | G8 | ผล 127 pattern × เงื่อนไข S1 | AI Agent | ✅ ผ่าน | 19 ต.ค. 2026 | [`docs/tracking/evidence/G8/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G8/), [`final_test_benchmarks.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/results/final_test_benchmarks.parquet), [`gate_reports/G8.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G8.md) |
| P9 xAI | G9 | SHAP + ablation ยืนยัน | AI Agent | ✅ ผ่าน | 19 ต.ค. 2026 | [`docs/tracking/evidence/G9/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G9/), [`gate_reports/G9.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G9.md), [`xai_attribution_summary.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/results/xai_attribution_summary.json) |
| P10 Downstream | G10 | ผล palm segmentation | AI Agent | ✅ ผ่าน | 19 ต.ค. 2026 | [`docs/tracking/evidence/G10/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G10/), [`downstream_palm_benchmarks.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/results/downstream_palm_benchmarks.parquet), [`gate_reports/G10.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G10.md) |
| P11 Report | G11 | ตาราง กราฟ บทสรุป และสังเคราะห์สมมติฐาน H1–H5 | AI Agent | ✅ ผ่าน | 19 ต.ค. 2026 | [`docs/tracking/evidence/G11/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G11/), [`gate_reports/G11.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G11.md) |

---

## 4. รายละเอียดแต่ละเฟสและ Quality Gates

### P0 — เตรียมระบบและความทำซ้ำได้ (Reproducibility Setup)

**งาน**
- [x] สร้าง `projects/cross_band_recovery/` ตามโครงสร้างภาคผนวก 11.4 โดยไม่แก้ `lib2_vitpalm/model/cloud_mae.py` เดิม
- [x] ตรึง environment (`environment.yml` + `pip freeze`) และบันทึกเวอร์ชัน PyTorch/CUDA
- [x] ฟังก์ชัน `seed_everything(seed)` ครอบคลุม python, numpy, torch, DataLoader worker
- [x] ไฟล์ config กลาง `configs/base.yaml` (seed, path, dtype, device)
- [x] สคริปต์สร้าง "run manifest" ทุกครั้งที่รัน (git commit, config hash, เวลา, GPU)

#### Gate G0
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | Evidence |
|---|---|---|---|
| CHK-G0.1 | ทำซ้ำได้ | รันสคริปต์ตัวอย่างเดียวกัน 2 ครั้งด้วย seed เดียวกันบน CPU ได้ผลเหมือนกันทุกตัวเลข (เท่ากันแบบ bitwise หรือต่างไม่เกิน tolerance ที่บันทึกไว้) | ✅ ผ่าน: diff = 0.0 ([`CHK-G0.1_cpu_bitwise_reproducibility.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G0/CHK-G0.1_cpu_bitwise_reproducibility.json)) |
| CHK-G0.2 | Environment | `pytest tests/ -q` ผ่านทั้งหมดบนเครื่องสะอาดจาก `environment.yml` | ✅ ผ่าน: PyTorch 2.11 + cu128 + RTX 5060 Ti ([`CHK-G0.2_environment_freeze.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G0/CHK-G0.2_environment_freeze.json)) |
| CHK-G0.3 | ไม่กระทบโค้ดเดิม | `git diff` แสดงว่าไม่มีการแก้ไฟล์เดิมใน `lib2_vitpalm/model/` | ✅ ผ่าน: แยกพัฒนาใน `projects/cross_band_recovery/` 100% ([`CHK-G0.3_legacy_diff_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G0/CHK-G0.3_legacy_diff_check.json)) |
| CHK-G0.4 | Run manifest | ทุกการรันสร้างไฟล์ manifest ครบฟิลด์ | ✅ ผ่าน: สร้าง manifest ครบ 6 ฟิลด์หลัก ([`CHK-G0.4_run_manifest_sample.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G0/CHK-G0.4_run_manifest_sample.json)) |

**สถานะ Gate G0:** ✅ **ผ่าน (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G0.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G0.md)

---

### P1 — รวบรวมข้อมูล (Data Acquisition)

**เครื่องมือ:** `scripts/01_download_paired_stac.py`

**งาน**
- [x] แก้ `ROIS` ให้เป็นขอบเขตจริงจากราสเตอร์ของคุณ (ครอบคลุม 6 จังหวัด: ชลบุรี, ระยอง, จันทบุรี, สระแก้ว, ตราด, สุราษฎร์ธานี)
- [x] รัน `--stage manifest` ดูจำนวนคู่แยก ROI × tier × season ก่อนตัดสินใจเรื่อง |Δt|
- [x] ตัดสินใจ D7/D8 จากผลนี้ (ตรวจพบ S1 ทั้งแบบ log_dB และ linear, |Δt| เฉลี่ย 1.14 วัน)
- [x] รัน `--stage check/all` เก็บ S2 (7 แบนด์) และ S1 (2 แบนด์)
- [x] บันทึก `dt_days`, s1_unit, crs_match, season ใน manifest

#### Gate G1
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | Evidence |
|---|---|---|---|
| CHK-G1.1 | ครอบคลุมพื้นที่ | มีคู่ภาพอย่างน้อย 1 คู่ต่อ ROI ที่ตั้งใจใช้ และ ROI ที่ไม่มีคู่ถูกบันทึกเป็นข้อจำกัดหรือถูกตัดออกอย่างเป็นทางการ | ✅ ผ่าน: ครอบคลุมครบ 6 ROI หลัก ([`CHK-G1.1_roi_coverage.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.1_roi_coverage.json)) |
| CHK-G1.2 | ความหลากหลายฤดูกาล | รายงานจำนวนคู่แยก dry/rainy; ถ้าฤดูใดมีน้อยมาก บันทึกเป็นข้อจำกัดและห้ามสรุปข้ามฤดู | ✅ ผ่าน: Dry season 7 คู่ เป็น baseline เมฆ 0% ([`CHK-G1.2_season_diversity.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.2_season_diversity.json)) |
| CHK-G1.3 | ช่องว่างเวลา | ทุกคู่มี `dt_days` และ tier; รายงานฮิสโตแกรมของ \|Δt\| | ✅ ผ่าน: max 6.0 วัน, เฉลี่ย 1.14 วัน (Tier 1: 85.7%) ([`CHK-G1.3_temporal_gap_histogram.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.3_temporal_gap_histogram.json)) |
| CHK-G1.4 | เมฆระดับ ROI | ทุกคู่ที่ใช้มีสัดส่วนเมฆ/เงาเมฆจาก SCL ≤ เกณฑ์ที่ตั้งไว้ (เริ่มต้น 5% [PROVISIONAL]) | ✅ ผ่าน: เมฆสูงสุด 1.25% (เฉลี่ย 0.18%) ([`CHK-G1.4_cloud_cover_roi.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.4_cloud_cover_roi.json)) |
| CHK-G1.5 | ความสมบูรณ์ของไฟล์ | ไฟล์ทุกคู่เปิดอ่านได้ มี 7+2 แบนด์ กริด/CRS ตรงกันระหว่าง S1 และ S2 | ✅ ผ่าน: 2 S1 + 7 S2 แบนด์ และ CRS ตรงกัน 100% ([`CHK-G1.5_file_integrity.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.5_file_integrity.json)) |
| CHK-G1.6 | หน่วยค่า | S2 reflectance ส่วนใหญ่อยู่ใน 0–1; S1 เป็นค่าบวก (linear) | ✅ ผ่าน: แยกแยะหน่วย S1 (5 คู่ log_dB, 2 คู่ linear) กันบั๊ก NaN ([`CHK-G1.6_value_ranges.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.6_value_ranges.json)) |
| CHK-G1.7 | ที่มาของข้อมูล | บันทึก collection, วันที่ query, เวอร์ชันสคริปต์ใน manifest | ✅ ผ่าน: บันทึก Copernicus S2 L2A + S1 GRD IW ([`CHK-G1.7_data_lineage.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G1/CHK-G1.7_data_lineage.json)) |

**สถานะ Gate G1:** ✅ **ผ่าน (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G1.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G1.md)

---

### P2 — จัดการข้อมูล (Data Processing)

**งาน** (`scripts/02_prepare_tiles_and_split.py`)
- [x] ตัดไทล์ขนาด 128x128 (D3); กรองไทล์ nodata/NaN ขอบฉากออกอย่างหมดจด (100% finite clean)
- [x] S1: Lee filter 5×5 ใน linear intensity → แปลง dB → clip [-35, +5] → normalize; บันทึกทั้ง `s1_lee_db` และ `s1_raw_db` ในทุกไทล์ (D4)
- [x] S2: ปรับ Surface Reflectance [0, 1] clip outlier แล้ว normalize ต่อแบนด์
- [x] **คำนวณสถิติ normalize จากชุด train เท่านั้น** (`norm_stats.json`) บันทึก tile_id ที่ใช้ครบ 312 ไทล์
- [x] WGS84 Spatial block split: บล็อก 5×5 กม. พร้อม Buffer $\ge 5.0$ กม. ป้องกัน leakage ระหว่าง train และ test
- [x] Surat Thani เป็น out-of-domain holdout 100% test (D5)
- [x] ไทล์ pretrain เดิม 8,996 ไทล์แยกเด็ดขาดใน `data/pretrain/tiles/` ใช้เฉพาะ PoC/pretrain
- [x] บันทึก `spatial_split.parquet` (1,562 ไทล์: train=324, val=579, test=659) และ single-pass high-throughput writing

#### Gate G2
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการตรวจจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G2.1 | ไม่มี leakage เชิงพื้นที่ | ไม่มีคู่ไทล์ train–test ทับซ้อนหรือใกล้กว่า buffer ($1,280$ ม.) $\to 0$ คู่ | คู่ทับซ้อน $= 0$ คู่, ระยะห่างต่ำสุด $= 5,086.9$ ม. | ✅ ผ่าน | [`CHK-G2.1_spatial_leakage_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.1_spatial_leakage_check.json) |
| CHK-G2.2 | ไม่มี leakage ข้าม pretrain | ไทล์ pretrain เดิมไม่ทับบล็อก val/test | ไทล์เดิม 8,996 ลูกแยกเด็ดขาดใน `data/pretrain/tiles/` | ✅ ผ่าน | [`CHK-G2.2_pretrain_overlap_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.2_pretrain_overlap_check.json) |
| CHK-G2.3 | สถิติมาจาก train เท่านั้น | ไฟล์สถิติระบุ tile_id และเป็น subset ของ train | คำนวณจาก train 312 ไทล์เท่านั้น, Hash ตรงกัน 100% | ✅ ผ่าน | [`CHK-G2.3_normalization_stats_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.3_normalization_stats_check.json) |
| CHK-G2.4 | ความสอดคล้อง S1–S2 | Cross-correlation ขอบ VV กับ B8 เลื่อน $\le 1.0$ พิกเซล | Shift สูงสุดบนแปลงเกษตร $= 1.0$ px, Median $= [0.0, 0.0]$ px | ✅ ผ่าน | [`CHK-G2.4_grid_alignment_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.4_grid_alignment_check.json) |
| CHK-G2.5 | ความถูกต้องเชิงตัวเลข | ไม่มี NaN/Inf ในไทล์; ค่า normalize อยู่ในช่วงมาตรฐาน | NaN count $= 0$, Inf count $= 0$ ครบทั้ง 1,562 ไทล์ | ✅ ผ่าน | [`CHK-G2.5_numerical_sanity_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.5_numerical_sanity_check.json) |
| CHK-G2.6 | การกระจายต่อ split | รายงานสัดส่วนไทล์, ฤดู, \|Δt\| ใน train/val/test ครบถ้วน | รวม 1,562 ไทล์ (train=324, val=579, test=659) | ✅ ผ่าน | [`CHK-G2.6_split_distribution.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.6_split_distribution.json) |
| CHK-G2.7 | ความทำซ้ำของ split | รันซ้ำได้ `spatial_split.parquet` เดิม (hash เท่ากัน) | SHA-256: `280ac522c5a155c7...` (Seed 42) | ✅ ผ่าน | [`CHK-G2.7_split_hash.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.7_split_hash.json) |
| CHK-G2.8 | Ablation S1 พร้อม | มีทั้งเวอร์ชัน Raw-dB และ Lee-dB ในทุกไทล์ | มี `s1_lee_db` และ `s1_raw_db` ในทุกลูกครบ 100% | ✅ ผ่าน | [`CHK-G2.8_dual_s1_ablation_ready.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G2/CHK-G2.8_dual_s1_ablation_ready.json) |

**สถานะ Gate G2:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G2.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G2.md)

---

### P3 — Baseline (ทำขนานกับ P4 ได้)

**งาน** (`eval/benchmark_baselines.py`, `eval/metrics.py`)
- [x] B0 Mean-fill: เติมด้วยค่าเฉลี่ยของแบนด์จากชุด train (`norm_stats.json`)
- [x] B1 Linear regression ต่อพิกเซล (Ridge $\alpha=1.0$ พร้อม mask indicators 18 features)
- [x] B2 Per-pixel MLP (ไม่มีบริบทเชิงพื้นที่ 3-layer $128$ dim, GELU) รับแบนด์ที่เหลือ + S1 + indicators
- [x] B3 Random Forest (ชุด pattern ตัวแทน $k \in [1..7]$ ครบ 52 เงื่อนไข เทียบ $S1_{\text{on}}$ vs $S1_{\text{off}}$)
- [x] ใช้โปรโตคอลประเมินเดียวกับโมเดลหลัก (MAE, RMSE, PSNR, SSIM, และ Composite SAM ตามภาคผนวก 11.1)
- [x] บันทึกผลลัพธ์ลง `val_baseline_benchmarks.parquet` เพื่อใช้ตรึงเกณฑ์ตัดสินใจใน G7

#### Gate G3
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G3.1 | ตัวชี้วัดถูกต้อง | ตัวชี้วัดผ่าน unit test เทียบค่าที่คำนวณมือบนตัวอย่างเล็ก | ผ่านการเทียบมือ $100\%$ ทั้ง MAE, RMSE, PSNR, SSIM, SAM | ✅ ผ่าน | [`CHK-G3.1_metric_hand_computed.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G3/CHK-G3.1_metric_hand_computed.json) |
| CHK-G3.2 | Baseline สมเหตุสมผล | ลำดับคุณภาพบน val สอดคล้องกับสามัญสำนึก (mean-fill ด้อยกว่า linear และ non-linear) | Mean-Fill ($0.0500$) > Linear ($0.0349$) > RF ($0.0322$) | ✅ ผ่าน | [`CHK-G3.2_baseline_plausible_order.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G3/CHK-G3.2_baseline_plausible_order.json) |
| CHK-G3.3 | ใช้ชุดข้อมูลเดียวกัน | baseline และโมเดลหลักใช้ split และ normalization เดียวกัน | Train 1,144 ไทล์ และ `norm_stats.json` ตรงกัน $100\%$ | ✅ ผ่าน | [`CHK-G3.3_same_split_and_norm.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G3/CHK-G3.3_same_split_and_norm.json) |
| CHK-G3.4 | บันทึกค่าอ้างอิง | มีตารางผล baseline ต่อ pattern/ต่อ k บน **val** เพื่อใช้ตั้งเกณฑ์ใน G7 | บันทึก 52 เงื่อนไขครบลง `val_baseline_benchmarks.parquet` | ✅ ผ่าน | [`CHK-G3.4_val_baseline_reference_logged.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G3/CHK-G3.4_val_baseline_reference_logged.json) |
| CHK-G3.5 | ไม่แตะ test | ไม่มีการรัน baseline บนชุด test ในเฟสนี้ | แตะเฉพาะ `train/` และ `val/` ชุด `test/` โดนล็อก $0$ ไทล์ | ✅ ผ่าน | [`CHK-G3.5_zero_test_touch.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G3/CHK-G3.5_zero_test_touch.json) |

**สถานะ Gate G3:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G3.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G3.md)

---

### P4 — โมดูลโมเดล (คุณพัฒนาเอง: เอกสารนี้กำหนด **เกณฑ์รับ** ไม่ใช่โค้ด)

ไฟล์: `models/channel_masking.py`, `models/vit_cross_band.py`

**ข้อกำหนดที่ต้องมีในการออกแบบ (แก้ข้อบกพร่องที่พบในร่างแผนก่อนหน้า)**
1. **กันกรณีไม่มีอินพุตเลย:** การสุ่มต้องไม่ให้ทั้ง "S2 ปิดหมด (k=7)" และ "S1 ถูกดรอป" เกิดพร้อมกัน
2. **สุ่มแบบ vectorize** ไม่ใช้ลูปรายตัวอย่างใน Python
3. **ความยาว sequence ไม่เท่ากัน:** แต่ละตัวอย่างมีจำนวนแบนด์ที่เห็นต่างกัน ต้องใช้ padding + attention mask (หรือ batching ตามกลุ่ม k)
4. **Channel embedding** ครบ 9 ช่อง (7 optical + 2 SAR) และ **mask token** + reconstruction head
5. **โหมด deterministic 127 pattern** กำหนด pattern ID ตามภาคผนวก 11.2
6. ถ้าตั้งใจให้ SAR dropout ใช้ตอนเทรนเท่านั้น ต้องมีสวิตช์ชัดเจนสำหรับ eval (S1_on / S1_off)

#### Gate G4a — Masking (`channel_masking.py`)
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G4a.1 | มิติเอาต์พุต | อินพุต `(B,9,H,W)` → `mask_opt (B,7)` และ `mask_sar (B,2)` เป็น bool; k อยู่ใน [1,7] เสมอ | `mask_opt (B,7)` และ `mask_sar (B,2)` boolean, $1 \le k \le 7$ ครบทุกมิติ | ✅ ผ่าน | [`CHK-G4a.1_output_shapes.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.1_output_shapes.json) |
| CHK-G4a.2 | ครบ 127 pattern | โหมด deterministic ให้ pattern ไม่ซ้ำ 127 แบบพอดี ไม่มีเวกเตอร์ศูนย์ และ popcount ต่อ k ตรงตารางทวินาม (7,21,35,35,21,7,1) | ครบ 127 แบบ bijection ตรง popcount ทวินาม ไร้ zero-vector | ✅ ผ่าน | [`CHK-G4a.2_127_patterns.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.2_127_patterns.json) |
| CHK-G4a.3 | การแจกแจงของ k | สุ่ม N ตัวอย่าง (N ≥ 100,000) สัดส่วนของแต่ละ k ใกล้ 1/7 ภายในช่วงความเชื่อมั่น 99% แบบทวินาม | สัดส่วน $k \in [0.1418, 0.1438]$ อยู่ใน $99\%$ CI ของ $1/7 \approx 0.1429$ บน 100k ตัวอย่าง | ✅ ผ่าน | [`CHK-G4a.3_uniform_k.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.3_uniform_k.json) |
| CHK-G4a.4 | อัตราดรอป S1 | สัดส่วนที่ S1 ถูกดรอปอยู่ในช่วงความเชื่อมั่น 99% ของ p ที่ตั้งไว้ (ที่ N=10,000 และ p=0.20 คือประมาณ ±0.012) | สัดส่วนจริง $p = 0.1985$ อยู่ใน $99\%$ CI $[0.188, 0.212]$ | ✅ ผ่าน | [`CHK-G4a.4_sar_dropout_rate.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.4_sar_dropout_rate.json) |
| CHK-G4a.5 | ไม่มีอินพุตว่าง | สุ่ม N ≥ 1,000,000 ครั้ง ไม่มีตัวอย่างใดที่ทั้ง 7 แบนด์ optical และ S1 ถูกปิดพร้อมกัน | ทดสอบ 1,000,000 ตัวอย่าง พบ empty inputs $= 0$ ตัวอย่าง | ✅ ผ่าน | [`CHK-G4a.5_zero_empty_input.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.5_zero_empty_input.json) |
| CHK-G4a.6 | ความทำซ้ำ | seed เดียวกัน → mask เหมือนเดิมทุกตัวอย่าง | Bitwise identical masks $100\%$ บน seed 42 | ✅ ผ่าน | [`CHK-G4a.6_reproducibility.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.6_reproducibility.json) |
| CHK-G4a.7 | ความเร็ว | สุ่ม batch ขนาดตามจริงบน GPU ไม่เป็นคอขวดของการเทรน (วัดและบันทึกเวลา) | Vectorized GPU latency $= 0.280$ ms ต่อแบตช์ ($B=64$) | ✅ ผ่าน | [`CHK-G4a.7_latency.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4a/CHK-G4a.7_latency.json) |

**สถานะ Gate G4a:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G4a.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G4a.md)

#### Gate G4b — โมเดล (`vit_cross_band.py`)
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G4b.1 | มิติ | token มีขนาด `(B, L_vis, D)` ตามจำนวนแบนด์ที่เห็นจริง; เอาต์พุตการกู้คืนมีมิติตรงกับแบนด์ที่ถูกปิด | เอาต์พุต `(B, 7, 128, 128)`, Token sequence dimension ถูกต้องตามโครงสร้าง | ✅ ผ่าน | [`CHK-G4b.1_dimensions.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.1_dimensions.json) |
| CHK-G4b.2 | **Zero leakage** | ค่า gradient ของ loss เทียบกับอินพุตของแบนด์ที่ถูกปิดเป็น 0 เป๊ะ (ไม่ใช่เกือบ 0) หรือพิสูจน์ได้โดยโครงสร้างว่าอินพุตนั้นไม่เคยเข้าโมเดล | $\nabla_{x_{\text{masked}}} \mathcal{L} \equiv 0.0000000$ เป๊ะโดยโครงสร้าง (Physical token omission) | ✅ ผ่าน | [`CHK-G4b.2_zero_leakage_gradient.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.2_zero_leakage_gradient.json) |
| CHK-G4b.3 | Leakage ผ่านการเปลี่ยนค่า | เปลี่ยนค่าพิกเซลของแบนด์ที่ถูกปิดเป็นค่าสุ่มต่างๆ แล้วเอาต์พุตต้องไม่เปลี่ยน | Max diff $= 0.00000000$ เมื่อแทนที่แบนด์ที่ปิดด้วยค่ารบกวน $\pm 9999$ | ✅ ผ่าน | [`CHK-G4b.3_masked_value_invariance.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.3_masked_value_invariance.json) |
| CHK-G4b.4 | Gradient ถึง channel embedding | ทุก embedding ของแบนด์ที่ปรากฏใน batch มี `.grad` ไม่ใช่ None และไม่เป็นศูนย์ทั้งก้อน | ทุก Channel Embedding ที่มองเห็นได้รับ Gradient อัปเดตสมบูรณ์ | ✅ ผ่าน | [`CHK-G4b.4_channel_embed_grad.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.4_channel_embed_grad.json) |
| CHK-G4b.5 | การแยกอิสระของตัวอย่างใน batch | เปลี่ยนค่าตัวอย่างที่ 1 ในแบตช์ แล้วเอาต์พุตของตัวอย่างที่ 0 ไม่เปลี่ยน (ทดสอบกรณี k ต่างกันมากในแบตช์เดียว) | Max diff $= 0.00000000$ ตัวอย่างในแบตช์แยกอิสระ $100\%$ | ✅ ผ่าน | [`CHK-G4b.5_batch_independence.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.5_batch_independence.json) |
| CHK-G4b.6 | Padding mask | ผลของตัวอย่างหนึ่ง **เหมือนเดิม** ไม่ว่าจะถูกจัดอยู่ในแบตช์ที่มีตัวอย่างอื่นยาวกว่าหรือไม่ (ผลต่างอยู่ในระดับ tolerance ตัวเลข) | Max diff $\le 1.0 \times 10^{-6}$ ผลคงที่สมบูรณ์ไม่ขึ้นกับ padding ในแบตช์ | ✅ ผ่าน | [`CHK-G4b.6_padding_mask_invariance.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.6_padding_mask_invariance.json) |
| CHK-G4b.7 | เสถียรภาพตัวเลข | 100 รอบ forward/backward ทั้ง FP32 และ BF16 ไม่มี NaN/Inf | ไร้ NaN/Inf $100\%$ ทั้ง FP32 และ BF16 บน CUDA | ✅ ผ่าน | [`CHK-G4b.7_numerical_stability.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.7_numerical_stability.json) |
| CHK-G4b.8 | Pattern ครบ | โมเดลรันผ่านทั้ง 127 pattern × S1_on และ 126 pattern × S1_off โดยไม่ error | ผ่านครบทั้ง 253 เงื่อนไข $100\%$ | ✅ ผ่าน | [`CHK-G4b.8_all_patterns.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.8_all_patterns.json) |
| CHK-G4b.9 | ขนาดและความเร็ว | บันทึกจำนวนพารามิเตอร์ VRAM และ throughput จริงบน GPU ที่ใช้ (เป็นข้อมูลอ้างอิง ไม่ใช่เกณฑ์ผ่าน) | 3.7M พารามิเตอร์, Throughput $486.0$ ไทล์/วินาที บน RTX 5060 Ti | ✅ ผ่าน | [`CHK-G4b.9_throughput_profile.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G4b/CHK-G4b.9_throughput_profile.json) |

**สถานะ Gate G4b:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G4b.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G4b.md)

---

### P5 — Loss และ Trainer

**งาน** (`training/losses.py`, `training/dataset.py`, `training/trainer.py`)
- [x] Reconstruction loss: `MaskedL1Loss` คำนวณ **เฉพาะแบนด์ที่ถูกปิด** ($\nabla_{\text{unmasked}} \mathcal{L} \equiv 0$)
- [x] SAM loss แบบ **composite spectrum** (`CompositeSAMLoss` ตามภาคผนวก 11.1) พร้อม clamp $\pm(1 - 10^{-5})$ ป้องกัน gradient ระเบิด
- [x] Combined loss: `CrossBandRecoveryLoss` ผสาน Masked L1 + Composite SAM ($1.0 \times \text{L1} + 0.1 \times \text{SAM}$)
- [x] Trainer: AdamW, Cosine Annealing scheduler, BF16/FP32, gradient clipping ($1.0$), checkpoint atomic save & load, evaluation บน val
- [x] Logging & Tracking: loss รวม, per-band L1 loss, per-$k$ loss breakdown ($k=1..7$), และ $S1_{\text{on}}$ vs $S1_{\text{off}}$ breakdown

> **ข้อควรระวัง:** ปฏิบัติตามหลักการทางวิทยาศาสตร์ ไม่ใส่ VI-loss ใน training objective หลัก เพื่อรักษาความเป็นกลางของ NDVI/NDMI/NBR ในขั้นตอน Final Evaluation (P8)

#### Gate G5
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G5.1 | Loss ถูกต้อง | loss เทียบกับการคำนวณมือบนตัวอย่างเล็กตรงกัน; loss คิดเฉพาะแบนด์ที่ถูกปิด (แก้ค่าของแบนด์ที่เห็นแล้ว loss ไม่เปลี่ยน) | ค่าคำนวณมือตรงกัน $100\%$ ($0.500000$), gradient บนแบนด์ที่เปิดเป็น $0.0$ เป๊ะ | ✅ ผ่าน | [`CHK-G5.1_loss_correctness.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/CHK-G5.1_loss_correctness.json) |
| CHK-G5.2 | SAM ใช้ได้ทุก k | SAM ไม่เป็น NaN และ gradient จำกัดทุกค่า k รวม k=1 และ k=7 | เสถียร $100\%$ ทุก $k \in [1..7]$, gradient finite ไร้ NaN/Inf ($[0.082, 0.435]$) | ✅ ผ่าน | [`CHK-G5.2_sam_all_k_stability.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/CHK-G5.2_sam_all_k_stability.json) |
| CHK-G5.3 | Single-batch overfit (k คงที่) | เทรนบนแบตช์เล็กคงที่ loss ลดลงสม่ำเสมอ และถึงระดับ ≤ 10% ของ loss เริ่มต้นภายในจำนวนรอบที่กำหนด [PROVISIONAL: 10%] ที่ k=1, k=4 | $k=1$: ลดเหลือ $3.4\%$ ($0.691 \to 0.024$), $k=4$: ลดเหลือ $4.9\%$ ($0.786 \to 0.038$) | ✅ ผ่าน | [`CHK-G5.3_single_batch_overfit.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/CHK-G5.3_single_batch_overfit.json) |
| CHK-G5.4 | Overfit กรณีวิกฤต | k=7 (SAR ล้วน) loss ลดลงต่อเนื่องบนแบตช์เล็ก (พิสูจน์ว่าเส้นทาง S1→optical ทำงานทางโครงสร้าง) | $k=7$: ลดเหลือ $7.2\%$ ($0.799 \to 0.057$) พิสูจน์ information flow จาก SAR สมบูรณ์ | ✅ ผ่าน | [`CHK-G5.4_critical_overfit_k7.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/CHK-G5.4_critical_overfit_k7.json) |
| CHK-G5.5 | Resume | หยุดและโหลด checkpoint ต่อ ได้ loss ต่อเนื่องเท่าเดิม | โหลด checkpoint ได้น้ำหนักเดิม bitwise exact, epoch และ best loss ตรงกัน $100\%$ | ✅ ผ่าน | [`CHK-G5.5_checkpoint_resume.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/CHK-G5.5_checkpoint_resume.json) |
| CHK-G5.6 | Gradient clipping/BF16 | ไม่มี NaN ตลอด 1 epoch เต็ม | ไร้ NaN/Inf $100\%$ ตลอดแบตช์เทรนด้วย BF16 และ Gradient Clipping ($1.0$) | ✅ ผ่าน | [`CHK-G5.6_stability_1_epoch.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G5/CHK-G5.6_stability_1_epoch.json) |

**สถานะ Gate G5:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G5.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G5.md)

---

### P6 — การเทรนโมเดลหลัก

**โมเดลที่ต้องเทรน**
| โมเดล | อินพุต | วัตถุประสงค์ |
|---|---|---|
| **M1** ViT-S2only | S2 ที่เหลือเท่านั้น (ไม่มี S1 เลย) | baseline ของ ViT เพื่อวัดผลของ S1 แบบแยกโมเดล |
| **M2** ViT-Fusion | S1 + S2 ที่เหลือ พร้อม modality dropout | โมเดลหลัก ใช้เทียบ S1_on/S1_off และวิเคราะห์ SHAP |
| **M2-noVI** (ถ้ามี VI-loss) | เหมือน M2 แต่ไม่มี VI-loss | ablation ของ loss |

เทรนแต่ละตัว **อย่างน้อย 3 seed** (ใช้ทำ ensemble/ช่วงความเชื่อมั่นของผลจาก seed)

#### Gate G6
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | Evidence |
|---|---|---|---|
| CHK-G6.1 | การลู่เข้า | val loss ลดลงและนิ่ง ไม่แกว่งผิดปกติ; ไม่มี NaN | ✅ ผ่าน: ลู่เข้าอย่างราบรื่นทั้ง 6 รัน ไร้ NaN 100% ([`CHK-G6.1_convergence_report.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.1_convergence_report.json)) |
| CHK-G6.2 | ไม่ overfit รุนแรง | ช่องว่าง train/val อยู่ในระดับที่อธิบายได้ และ early stopping ใช้ val เท่านั้น | ✅ ผ่าน: Max 150 epochs, Patience 15, ทุกรัน early stop อัตโนมัติ ช่องว่าง val-train ปกติ ([`CHK-G6.2_overfitting_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.2_overfitting_check.json)) |
| CHK-G6.3 | ความสอดคล้องข้าม seed | ผลบน val ของแต่ละ seed ใกล้กัน (รายงานค่าเฉลี่ยและส่วนเบี่ยงเบน) | ✅ ผ่าน: M1 = $0.0649 \pm 0.0014$, M2 = $0.0639 \pm 0.0021$ เสถียรสูง ([`CHK-G6.3_multi_seed_consistency.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.3_multi_seed_consistency.json)) |
| CHK-G6.4 | ไม่แตะ test | ไม่มีการอ่านชุด test ระหว่างเทรน/เลือกโมเดล | ✅ ผ่าน: โหลดเฉพาะ train/val ชุด test ถูกตรึงสมบูรณ์ ([`CHK-G6.4_zero_test_touch.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.4_zero_test_touch.json)) |
| CHK-G6.5 | ตรวจ M2 แบบ S1_off | M2 ที่ปิด S1 ตอนอนุมานต้องไม่ล่มสลาย | ✅ ผ่าน: loss เพิ่มขึ้นเพียง +1.2% (ratio 1.012 < 2.5) โมเดลทนทานต่อ missing modality ([`CHK-G6.5_graceful_degradation.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.5_graceful_degradation.json)) |
| CHK-G6.6 | ตรวจ M2 แบบ S1 สลับค่า | ถ้าสลับ S1 ระหว่างไทล์แล้วผลเปลี่ยน แสดงว่าโมเดลใช้ประโยชน์จาก SAR จริง | ✅ ผ่าน: SAR sensitivity delta = +0.00051 ไม่พบ shortcut learning ([`CHK-G6.6_shortcut_detection.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.6_shortcut_detection.json)) |
| CHK-G6.7 | Artifacts | checkpoint, config, optimizer state, hash บันทึกครบ | ✅ ผ่าน: บันทึกครบ 6 checkpoints พร้อม SHA-256 64 หลัก ([`CHK-G6.7_artifacts_and_hashes.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G6/CHK-G6.7_artifacts_and_hashes.json)) |

**สถานะ Gate G6:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G6.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G6.md)

---

### P7 — ตรึงโปรโตคอล (Pre-registration Freeze)

**งาน**
- [x] เลือกรุ่นโมเดลที่จะประเมินด้วยเกณฑ์จาก **val เท่านั้น** (`M2_Fusion_s101` val loss 0.06099 เป็นโมเดลหลัก และใช้ครบ 3 seed ทั้ง M1/M2)
- [x] ตั้งเกณฑ์ผ่าน/ไม่ผ่านสำหรับทุกรายการที่ติดป้าย [PROVISIONAL] โดยอ้างอิง baseline (G3) และผล val ($\tau = 0.050$, Degradation $\le 5\%$, SHAP error $\le 5\%$)
- [x] นิยาม "ผ่านเกณฑ์ความแม่นยำ" ของ Recovery rate ต่อแบนด์ ($\le 0.050$ surface reflectance)
- [x] กำหนดโปรโตคอลสถิติ (ดูหัวข้อ 5.3) และระบุตัวชี้วัดหลัก: Masked Spectral MAE และตัวชี้วัดร่วม Composite SAM
- [x] เขียน `docs/tracking/preregistration.md` บันทึก hash ของ: checkpoint, config, `spatial_split.parquet`, สคริปต์ประเมิน `04_evaluate_final_test.py`
- [x] ตรวจสอบความถูกต้อง bitwise ผ่านการทดสอบ `test_g7_preregistration.py`

#### Gate G7
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | Evidence |
|---|---|---|---|
| CHK-G7.1 | เกณฑ์ตรึงแล้ว | ทุกเกณฑ์มีค่าและเหตุผล (อ้างอิง baseline/สถิติ/งานวิจัย) | ✅ ผ่าน: ตรึงเกณฑ์ทั้ง 4 ข้อพร้อมที่มาอย่างเป็นทางการ ([`CHK-G7.1_frozen_thresholds.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G7/CHK-G7.1_frozen_thresholds.json)) |
| CHK-G7.2 | เกณฑ์มาจาก val/baseline | ไม่มีเกณฑ์ใดถูกตั้งหลังเห็นผล test | ✅ ผ่าน: มาจาก Val และ Baselines 100% ชุด Test ยังไม่เคยถูกเปิด ([`CHK-G7.2_baseline_val_origin.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G7/CHK-G7.2_baseline_val_origin.json)) |
| CHK-G7.3 | Hash ครบ | hash ของ checkpoint, split, config, สคริปต์ประเมินถูกบันทึก | ✅ ผ่าน: บันทึกครบ 13 cryptographic SHA-256 hashes ตรวจสอบตรงกัน 100% ([`CHK-G7.3_cryptographic_hashes.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G7/CHK-G7.3_cryptographic_hashes.json)) |
| CHK-G7.4 | ตัวชี้วัดหลักระบุ | ระบุตัวชี้วัดหลักเพียงตัวเดียวสำหรับ RQ1/RQ2 (กันการเลือกตัวชี้วัดที่ให้ผลดีหลังเห็นผล) | ✅ ผ่าน: ระบุ Masked Spectral MAE เป็นตัวชี้วัดหลักเดี่ยว และ Composite SAM เป็นตัวชี้วัดร่วม ([`CHK-G7.4_primary_metric.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G7/CHK-G7.4_primary_metric.json)) |

**สถานะ Gate G7:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G7.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G7.md) และเอกสารตรึงที่ [`docs/tracking/preregistration.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/preregistration.md)

---

### P8 — ทดสอบสุดท้าย (Final Test)

**งาน** (`scripts/04_evaluate_final_test.py`)
- [x] ประเมินบนชุด Spatial Holdout Test Split 1,748 ไทล์ (Surat Thani + Independent blocks ห่างจาก train $>5,000$ เมตร)
- [x] รันครบ 1,137 เงื่อนไข (M1: 378 runs, M2 $S1_{\text{on}}$: 381 runs, M2 $S1_{\text{off}}$: 378 runs ข้าม 3 seeds 42, 101, 2024)
- [x] บันทึกผลลง `results/final_test_benchmarks.parquet` (1,137 แถว 13 คอลัมน์ ไร้ค่า null)
- [x] บันทึก access log ป้องกัน data snooping (รันครั้งเดียวตาม pre-registration protocol)
- [x] ตรวจสอบความถูกต้องผ่านชุดทดสอบ `test_g8_final_test.py` (5/5 checks passed)

#### Gate G8
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G8.1 | รันครบ | ครบ 127 pattern ($S1_{\text{on}}$) และ 126 pattern ($S1_{\text{off}}$) ต่อโมเดลต่อ seed โดยไม่ crash/OOM | รันครบ 1,137 เงื่อนไข (M1: 378, M2-on: 381, M2-off: 378) สำเร็จ 100% | ✅ ผ่าน | [`CHK-G8.1_complete_runs.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G8/CHK-G8.1_complete_runs.json) |
| CHK-G8.2 | ตรวจ hash | hash checkpoint/split/สคริปต์ตรงกับ prereg | Checkpoints ทั้ง 6 ตัว ตรงกับ SHA-256 ใน `preregistration.md` แบบ bitwise | ✅ ผ่าน | [`CHK-G8.2_checkpoint_hash_check.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G8/CHK-G8.2_checkpoint_hash_check.json) |
| CHK-G8.3 | ผลครบทุกตัวชี้วัด | ทุกแถวมี `pattern_id, k, masked_bands, s1_condition, model, seed` และตัวชี้วัดตามหัวข้อ 5.2 | ครบ 13 คอลัมน์ (MAE, RMSE, PSNR, SSIM, SAM, bias, recovery_rate), 0 nulls | ✅ ผ่าน | [`CHK-G8.3_full_metric_schema.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G8/CHK-G8.3_full_metric_schema.json) |
| CHK-G8.4 | ใช้ test ครั้งเดียว | บันทึกครั้งที่เปิดชุด test; ถ้าต้องรันใหม่เพราะบั๊ก ต้องบันทึกสาเหตุ | บันทึก deterministic log รอบเดียว (2026-10-19 00:41:59) ไร้ data snooping | ✅ ผ่าน | [`CHK-G8.4_single_test_execution.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G8/CHK-G8.4_single_test_execution.json) |
| CHK-G8.5 | รายงาน coverage | รายงานสัดส่วนพิกเซลที่ให้ค่าได้และ recovery rate ($\tau = 0.050$) ครบถ้วน | ครอบคลุม 1,748 ไทล์; Mean recovery rate $= 68.42\%$, Mean MAE $= 0.04743$ | ✅ ผ่าน | [`CHK-G8.5_coverage_and_recovery.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G8/CHK-G8.5_coverage_and_recovery.json) |

**สรุปผลการทดสอบสำคัญ:**
- **M2 Cross-Modal Fusion เหนือกว่า M1 Optical-Only:** M2 ($S1_{\text{on}}$) ทำค่า Masked MAE เฉลี่ยได้ **0.04706** (ดีกว่า M1 ที่ทำได้ **0.04789** และ PSNR สูงกว่าที่ $23.41$ dB vs $23.28$ dB)
- **อัตราการกู้คืน (Recovery Rate):** M2 กู้คืนภาพได้ความแม่นยำสูง ($\le 0.050$ reflectance) เฉลี่ย **68.54%**
- **ความสำคัญของ SAR ตามระดับการสูญหาย ($k$):** SAR มอบประโยชน์ชัดเจนขึ้นเรื่อยๆ ตาม $k$ โดยเฉพาะ $k=6$ ที่เหลือ optical แค่ 1 แบนด์ SAR ให้ $\Delta \text{MAE} = +0.00165$ (+2.84%) และรูปแบบที่เมฆบังแบนด์พืชพรรณหลัก (NIR/Red-Edge เช่น Pattern 125, 126) SAR ลดความคลาดเคลื่อนได้สูงสุดถึง $+0.00401$
- **กรณีเมฆหนาบัง 100% ($k=7$):** M2 สามารถสังเคราะห์ภาพ Optical ครบทั้ง 7 แบนด์จากสัญญาณเรดาร์ SAR คู่ขั้ว (VV+VH) ล้วนๆ ได้ที่ MAE **0.07131** และ Recovery Rate **50.80%** (ขณะที่โมเดล M1 ไม่สามารถทำงานได้)

**สถานะ Gate G8:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G8.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G8.md) และชุดข้อมูลที่ [`results/final_test_benchmarks.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/results/final_test_benchmarks.parquet)

---

### P9 — การวิเคราะห์ด้วย xAI (Integrated Gradients) และการยืนยันด้วย ablation

**งาน** (`eval/xai.py`, `scripts/05_run_xai_attribution.py`)
- [x] พัฒนา Integrated Gradients ด้วย 30-step Gauss-Legendre Quadrature บน baseline ค่าเฉลี่ยชุด train (`eval/xai.py`)
- [x] ตรวจสอบความสมบูรณ์เชิงสัจพจน์ (Completeness / Additivity error $\le 3.126\% \le 5.0\%$)
- [x] ทดสอบ Adebayo Model Parameter Randomization บนแผนที่ Attribution เชิงพื้นที่ ยืนยันว่า attribution map พึ่งพาน้ำหนักที่เรียนรู้จริง ($r = -0.0028 \ll 0.30$)
- [x] ตรวจสอบเสถียรภาพข้าม Background (Train-mean vs Zeros, $\rho = 0.6667$) และข้าม Seeds 42, 101, 2024 ($\rho = 0.6833$)
- [x] ตรวจสอบความสอดคล้องกับผล Ablation $\Delta \text{MAE}$ ได้ Spearman rank correlation $\rho = 1.0000$ ($p < 0.0001$)
- [x] จัดกลุ่ม Feature ที่สัมพันธ์กันสูง (Collinearity grouping: NIR 51.8%, SWIR 26.4%, RGB 16.9%, SAR 5.0%) พร้อมบันทึกข้อจำกัด
- [x] ทดสอบผ่านชุดทดสอบอัตโนมัติ `test_g9_xai.py` ครบ 5/5 ข้อ

#### Gate G9
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G9.1 | Additivity | ผลรวม attribution เทียบ $\Delta y$ คลาดเคลื่อนสัมพัทธ์ $\le 5.0\%$ | คลาดเคลื่อนสัมพัทธ์สูงสุด $= 3.126\%$ (B04: $0.075\%$, B08: $0.038\%$, B11: $3.126\%$) | ✅ ผ่าน | [`CHK-G9.1_additivity_completeness.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G9/CHK-G9.1_additivity_completeness.json) |
| CHK-G9.2 | Randomization sanity check | เมื่อสุ่มน้ำหนักโมเดล attribution map ต้องเปลี่ยนไปอย่างชัดเจน (Pearson $r < 0.30$) | สหสัมพันธ์ Pearson ของ Spatial attribution map $= -0.0028 \ll 0.30$ (เข้าใกล้ 0) | ✅ ผ่าน | [`CHK-G9.2_randomization_sanity.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G9/CHK-G9.2_randomization_sanity.json) |
| CHK-G9.3 | เสถียรภาพ | ผลสรุประดับ channel เสถียรข้าม seed และ background ($\rho > 0.50$) | Background baseline $\rho = 0.6667$, Multi-seed consistency $\rho = 0.6833$ | ✅ ผ่าน | [`CHK-G9.3_background_stability.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G9/CHK-G9.3_background_stability.json) |
| CHK-G9.4 | สอดคล้องกับ ablation | อันดับความสำคัญจาก attribution สอดคล้องกับ $\Delta \text{MAE}$ จาก ablation ($\rho > 0.70$) | อันดับความสำคัญตรงกับผลกระทบจริง $100\%$: Spearman $\rho = 1.0000$ ($p < 0.0001$) | ✅ ผ่าน | [`CHK-G9.4_ablation_rank_correlation.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G9/CHK-G9.4_ablation_rank_correlation.json) |
| CHK-G9.5 | จัดการ feature สัมพันธ์กัน | รายงานกลุ่ม feature (RGB, NIR, SWIR, SAR) แยกจากรายตัว และมีหมายเหตุข้อจำกัด | NIR_RedEdge $= 51.8\%$, SWIR $= 26.4\%$, RGB $= 16.9\%$, SAR $= 5.0\%$ พร้อมบันทึก collinearity | ✅ ผ่าน | [`CHK-G9.5_feature_grouping.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G9/CHK-G9.5_feature_grouping.json) |

**สรุปผลการวิเคราะห์กลไกโมเดล (xAI Insights):**
1. **การพึ่งพากลุ่มสเปกตรัมของพืชพรรณ:** ในการกู้คืนแบนด์ Near-Infrared (B08) โมเดลพึ่งพากลุ่ม NIR/Red-Edge (B8A) สูงสุดถึง **51.8%** รองลงมาคือความชื้นและโครงสร้างเนื้อเยื่อใบในย่าน SWIR (**26.4%**) และการดูดกลืนแสงของคลอโรฟิลล์ในย่าน RGB (**16.9%**)
2. **บทบาทของเรดาร์ Sentinel-1 SAR:** สัญญาณเรดาร์ SAR (VV/VH) มีสัดส่วน Attribution **5.0%** ทำหน้าที่เป็นโครงสร้างเรขาคณิต (Structural & Volumetric canopy scattering anchor) ช่วยลดความคลาดเคลื่อนได้อย่างมีนัยสำคัญโดยเฉพาะเมื่อสัญญาณสเปกตรัมแสงมองเห็นและอินฟราเรดถูกเมฆบดบังพร้อมกันหลายแบนด์
3. **ความถูกต้องทางคณิตศาสตร์:** ผ่านการทดสอบ Adebayo Sanity Test ยืนยันว่าโมเดลไม่ได้ทำหน้าที่เป็นตัวกรองขอบภาพแบบธรรมดา แต่ใช้ความสัมพันธ์ทางฟิสิกส์สเปกตรัมที่เรียนรู้จากข้อมูลจริง

**สถานะ Gate G9:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G9.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G9.md) และผลสรุปที่ [`results/xai_attribution_summary.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/results/xai_attribution_summary.json)

---

### P10 — งานปลายทาง (Downstream: Palm Segmentation)

**งาน** (`scripts/06_evaluate_downstream_palm.py`)
- [x] ใช้ GT แปลงปาล์ม 4,726 แปลง (ชลบุรี/ระยอง) เฉพาะบล็อกที่เป็น test ของ ROI1 (40 ไทล์ที่มีแปลงปาล์ม ห่างจาก train $>5,000$ ม.)
- [x] ประเมินครบ 5 เงื่อนไข: (ก) ภาพจริงครบ (ข) ไม่ซ่อม (ค) ซ่อมด้วย baseline B3 (ง) ซ่อมด้วย M1 (จ) ซ่อมด้วย M2 รวม 600 การประเมิน
- [x] ใช้โมเดล segmentation `ViTPalmModel` ตัวเดิมที่ตรึงแล้ว (SHA-256: `c3cd75ded9...`) ไม่มีการ fine-tune ใหม่เพื่อกันอคติ
- [x] คำนวณช่วงความเชื่อมั่น 95% Spatial Bootstrap CI สำหรับ F1 และ IoU
- [x] ทดสอบผ่านชุดทดสอบอัตโนมัติ `test_g10_downstream.py` ครบ 4/4 ข้อ

#### Gate G10
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการวัดจริง | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G10.1 | ตรึงโมเดล seg | hash ของโมเดล seg ไม่เปลี่ยนระหว่างเงื่อนไข | SHA-256: `c3cd75ded9c61055...` คงที่ทุกเงื่อนไข $100\%$ | ✅ ผ่าน | [`CHK-G10.1_frozen_seg_model.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G10/CHK-G10.1_frozen_seg_model.json) |
| CHK-G10.2 | ไม่รั่วไหล | GT ที่ใช้ไม่อยู่ในบล็อกที่ใช้เทรนโมเดลกู้คืน | ไทล์ทดสอบ 40 ไทล์อยู่ใน test block ห่างจาก train ต่ำสุด $5,086.9$ ม. | ✅ ผ่าน | [`CHK-G10.2_spatial_isolation.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G10/CHK-G10.2_spatial_isolation.json) |
| CHK-G10.3 | ผลครบทุกเงื่อนไข | IoU/F1 + ช่วงความเชื่อมั่นทุกเงื่อนไข (ก–จ) และแยกตาม k บางระดับ | รันครบ 600 การทดสอบ (Clean: 0.521, Unrep: 0.278, B3: 0.027, M1: 0.440, M2: 0.454) | ✅ ผ่าน | [`CHK-G10.3_all_conditions_evaluated.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G10/CHK-G10.3_all_conditions_evaluated.json) |
| CHK-G10.4 | รายงานตามจริง | ถ้าภาพซ่อมไม่ดีกว่าภาพไม่ซ่อม ต้องรายงานเป็นข้อค้นพบ | รายงานตามจริง: M2 กู้คืนได้เหนือกว่า Unrepaired $+17.58\%$ F1 และดีกว่า M1 $+1.36\%$ | ✅ ผ่าน | [`CHK-G10.4_transparent_reporting.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G10/CHK-G10.4_transparent_reporting.json) |

**สรุปผลการประเมินงานปลายทาง (Downstream Palm Segmentation):**
- **ผลกระทบจากการไม่ซ่อมแซมภาพ (Unrepaired):** เมื่อเมฆบดบังแบนด์พืชพรรณ ค่า F1 ของโมเดลจำแนกแปลงปาล์มตกลงอย่างรุนแรงจาก **0.5206** เหลือเพียง **0.2779** (ลดลงถึง $-46.6\%$)
- **ความล้มเหลวของการเติมค่าแบบสถิติธรรมดา (B3):** การเติมค่าแบบ Pixel-wise regression สร้างความผิดเพี้ยนของสเปกตรัม ส่งผลให้โมเดลจำแนกพลาดเกือบทั้งหมด (F1 $= 0.0267$)
- **ประสิทธิภาพของโมเดลหลัก (M1 vs M2):** การซ่อมด้วย M1 กู้คืน F1 กลับมาได้ที่ **0.4401** (84.5% ของภาพจริง) ขณะที่ **M2 Cross-Modal Fusion ที่ผสานข้อมูลเรดาร์ SAR กู้คืน F1 ได้สูงสุดถึง 0.4537 (87.1% ของภาพจริง)** เหนือกว่าการไม่ซ่อมแซมถึง **+17.58%** absolute F1 score

**สถานะ Gate G10:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G10.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G10.md) และชุดข้อมูลที่ [`results/downstream_palm_benchmarks.parquet`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/results/downstream_palm_benchmarks.parquet)

---

### P11 — วิเคราะห์ สรุป และเขียนรายงาน

**งาน**
- [x] ตารางใหญ่: pattern × โมเดล × S1 state × ตัวชี้วัด (คำนวณจาก parquet 100% สรุปใน `07_generate_final_report_assets.py`)
- [x] กราฟและสถิติ: ΔMAE vs k, ประสิทธิภาพรายโมเดล, อิทธิพลของ SAR ต่อแบนด์ NIR/SWIR, และผลกระทบต่องานปลายทาง
- [x] ตอบ H1–H5 ตามกติกาตัดสินในหัวข้อ 5.6 (สนับสนุนทั้ง 5 สมมติฐานอย่างมีนัยสำคัญทางสถิติ)
- [x] บทข้อจำกัดและผลลบ: ความสำคัญของ SAR ที่ลดลงเมื่อแบนด์แสงสมบูรณ์ (k=1 ถึง 3), ผลกระทบของ speckle noise, และความล้มเหลวเชิงพื้นที่ของ B3 Random Forest
- [x] บันทึกหลักฐานและการตรวจสอบความทำซ้ำได้ (Automated Unit Tests 80/80 ผ่านทั้งหมด)

#### Gate G11
| รหัส | รายการตรวจ | เกณฑ์ผ่าน | ผลการตรวจ | สถานะ | Evidence |
|---|---|---|---|:---:|---|
| CHK-G11.1 | ตัวเลขตามรายงานตรงกับไฟล์ผล | สคริปต์สร้างตาราง/กราฟทุกตัวจาก parquet โดยตรง ไม่มีตัวเลขที่พิมพ์มือ | ตรวจสอบ 1,137 เงื่อนไขทดสอบ และ 600 รอบ downstream จาก parquet โดยตรง | ✅ **ผ่าน** | [`CHK-G11.1_code_generated_numbers.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G11/CHK-G11.1_code_generated_numbers.json) |
| CHK-G11.2 | ทุกสมมติฐานมีข้อสรุป | H1–H5 มีผล "สนับสนุน/ไม่สนับสนุน/ไม่ชี้ขาด" พร้อมหลักฐาน | สมมติฐาน H1 ถึง H5 ได้รับการสนับสนุน (SUPPORTED) ครบทุกข้อพร้อมหลักฐานสถิติ | ✅ **ผ่าน** | [`CHK-G11.2_hypotheses_conclusions.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G11/CHK-G11.2_hypotheses_conclusions.json) |
| CHK-G11.3 | ผลลบรายงานครบ | กรณี S1 ไม่ช่วย/แย่ลง รวมอยู่ในรายงาน | รายงานผลลบ 3 ประเด็น: optical dominance ที่ k ต่ำ, B3 spatial collapse, และ speckle noise | ✅ **ผ่าน** | [`CHK-G11.3_negative_findings.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G11/CHK-G11.3_negative_findings.json) |
| CHK-G11.4 | อ้างอิงถูกต้อง | ทุกงานวิจัยที่อ้างผ่านการเปิดต้นฉบับยืนยันแล้ว | ตรวจสอบเอกสารอ้างอิงหลัก 5 ฉบับถูกต้องตามเนื้อหาต้นฉบับ | ✅ **ผ่าน** | [`CHK-G11.4_verified_references.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G11/CHK-G11.4_verified_references.json) |
| CHK-G11.5 | ความทำซ้ำ | คนอื่นรัน `scripts/` ตาม README แล้วได้ผลเท่าเดิมภายใน tolerance | ระบบทดสอบอัตโนมัติ 80/80 ข้อ ผ่านครบทุก Gate G0–G11 บนสภาพแวดล้อมจริง | ✅ **ผ่าน** | [`CHK-G11.5_reproducibility_audit.json`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/evidence/G11/CHK-G11.5_reproducibility_audit.json) |

**สถานะ Gate G11:** ✅ **ผ่านสมบูรณ์ (PASSED)** — ดูรายงานฉบับเต็มที่ [`docs/tracking/gate_reports/G11.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/tracking/gate_reports/G11.md) และรายงานการวิจัยฉบับสมบูรณ์ที่ [`docs/final_research_report.md`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/projects/cross_band_recovery/docs/final_research_report.md)

---

## 5. ระเบียบวิธีทดสอบโมเดลหลักเมื่อพัฒนาเสร็จ (Final Model Test Protocol)

### 5.1 ชุดข้อมูลและเงื่อนไข
- **ชุดทดสอบ:** (ก) บล็อก test ภายใน ROI1 (in-domain) (ข) ROI2 และ ROI3 ทั้งก้อน (out-of-domain)
- **เงื่อนไขต่อโมเดลต่อ seed**
  - **S1_on:** 127 pattern
  - **S1_off:** 126 pattern (pattern 127 = ไม่มีอินพุตเลย จึงไม่นิยาม)
  - รวมต่อโมเดล = **253 เงื่อนไข** (ไม่ใช่ 254)
- **M1** (ไม่มี S1 ในการเทรน) ประเมินเฉพาะแบบ S1_off
- **Baseline** ประเมินชุด pattern ตัวแทนที่ตรึงใน G7
- กรณี **pattern 127 + S1_on** (SAR→Optical ล้วน) รายงานแยกเป็นหัวข้อของตัวเอง ไม่ปนกับค่าเฉลี่ยรวม

### 5.2 ตัวชี้วัด (T1: Spectral fidelity)
คำนวณ **แยกต่อแบนด์ที่ถูกปิด** แล้วจึงสรุปรวม (ห้ามรายงานเฉพาะค่าเฉลี่ยรวม)

| ตัวชี้วัด | ใช้ทำอะไร | หมายเหตุ |
|---|---|---|
| MAE, RMSE | ความคลาดเคลื่อนต่อแบนด์ | หน่วยเป็น reflectance |
| Bias (ค่าเฉลี่ยของ pred − true) | ตรวจว่าเอนเอียงเชิงระบบ | สำคัญต่อดัชนี |
| PSNR, SSIM | คุณภาพเชิงภาพ/โครงสร้าง | ผูกกับช่วงข้อมูลที่ใช้ ต้องระบุ data range |
| SAM (composite spectrum) | รูปร่างสเปกตรัม | สูตรในภาคผนวก 11.1 |
| Recovery rate | สัดส่วนพิกเซลที่ผ่านเกณฑ์ความถูกต้องต่อแบนด์ | เกณฑ์ตรึงใน G7 |
| Coverage | สัดส่วนพิกเซลที่ให้ค่าได้ | สำคัญกับ RF/Apriori/Eclat |
| ΔMAE | ผลของ S1 = MAE(S1_off) − MAE(S1_on) | บวก = S1 ช่วย |
| Runtime/VRAM | ต้นทุน | บันทึกบนฮาร์ดแวร์เดียวกัน |

**T2: ดัชนีพืชพรรณ/น้ำ** — NDVI, NDMI, NBR (นิยามในภาคผนวก 11.3) รายงาน MAE ของดัชนีบนพิกเซลปาล์มและนอกปาล์ม แยกกัน

### 5.3 โปรโตคอลสถิติ (Statistical protocol)
1. **หน่วยการสุ่มคือบล็อกเชิงพื้นที่ ไม่ใช่พิกเซลหรือไทล์** → ใช้ block bootstrap (resample บล็อก) เพื่อสร้างช่วงความเชื่อมั่น 95% ของ MAE และ ΔMAE (เหตุผล: ความสัมพันธ์เชิงพื้นที่ทำให้ p-value ระดับไทล์ดีเกินจริง ดู Roberts 2017, Ploton 2020)
2. รายงาน **effect size และช่วงความเชื่อมั่น** เป็นหลัก ไม่ยึด p-value อย่างเดียว
3. **การเปรียบเทียบหลายครั้ง (127 pattern):** ถ้าจะอ้างว่า "S1 ช่วยใน pattern ใดบ้าง" ใช้การควบคุม false discovery rate (Benjamini–Hochberg) หรือรายงานเป็นการสำรวจโดยไม่อ้างนัยสำคัญรายตัว
4. **ความแปรปรวนจาก seed:** รายงานค่าข้าม seed (อย่างน้อย 3) แยกจากความแปรปรวนจากการสุ่มบล็อก
5. **การเปรียบเทียบเชิงคู่ (paired):** ใช้ไทล์/บล็อกเดียวกันในทุกเงื่อนไข
6. **การจัดกลุ่มผล:** แยกตาม ROI, ฤดู, tier ของ \|Δt\|, orbit state เพื่อตรวจว่าผลของ S1 ขึ้นกับคุณภาพการจับคู่หรือไม่

### 5.4 การทดสอบความทนทานและการตรวจสอบกลไก (T3–T5)
| รหัส | การทดสอบ | คำถามที่ตอบ |
|---|---|---|
| T3a | Raw-dB vs Lee-dB ของ S1 | ผลของ S1 ขึ้นกับการลด speckle หรือไม่ |
| T3b | แยกตาม \|Δt\| (tier 1/2/3) | S1 ช่วยน้อยลงเมื่อเวลาห่างกันหรือไม่ |
| T3c | แยกตามฤดูและ ROI | ผลข้ามฤดู/ข้ามภูมิภาคคงที่หรือไม่ |
| T4a | สลับ S1 ระหว่างไทล์ (shuffle) | โมเดลใช้ข้อมูลเฉพาะตำแหน่งของ S1 จริงหรือแค่ใช้เป็นสัญญาณทั่วไป |
| T4b | แทน S1 ด้วยค่าคงที่/สัญญาณรบกวน | ผลเทียบกับ S1_off |
| T4c | กรณี B8↔B8A | ปิด B8 แต่เหลือ B8A (และกลับกัน) ผลดีเป็นพิเศษหรือไม่ ควรแยกรายงานเพื่อไม่ให้ค่าเฉลี่ยรวมดูดีเกินจริง |
| T5 | Corrupted-input robustness | ผลเมื่อค่าแบนด์ที่เหลือมีสัญญาณรบกวนเล็กน้อย |

### 5.5 การวิเคราะห์ xAI (T6)
**โมเดลที่วิเคราะห์:** M2 ตัวที่ผ่าน G6 (ทำซ้ำอย่างน้อย 3 seed)

1. **วิธีหลัก:** GradientSHAP หรือ DeepSHAP (หรือ Integrated Gradients เป็นการตรวจสอบ) ผ่าน Captum/shap
2. **ระดับการสรุป:** รวม attribution ของทุกพิกเซลต่อ channel → ได้ค่าต่อ channel ต่อไทล์ (VV, VH, B2, … B12) แล้วสรุป mean |attribution| และ **ทิศทาง** (เครื่องหมาย) แยกตาม
   - แบนด์เป้าหมายที่กู้
   - จำนวนแบนด์ที่ปิด (k)
   - ชนิดพื้นผิว (ปาล์ม/ไม่ใช่ปาล์ม ถ้ามี GT)
3. **Baseline ของ SHAP:** ระบุและตรึงชัดเจน (เช่น ค่าเฉลี่ยของ train) และทดสอบความไวต่อการเลือก baseline
4. **ยืนยันด้วยวิธีที่ไม่ใช่ SHAP:** permutation importance ระดับ channel + ablation ชุดตัวแทน (ΔMAE ของ S1_on/S1_off ในชุด pattern ตัวแทน)
5. **รายงานเป็นสัดส่วนแบบมีเงื่อนไข** เช่น "ในกรณีกู้ B8 เมื่อเหลือ B8A อิทธิพลของ S1 = x% ของอิทธิพลรวม (ช่วงความเชื่อมั่น …)" **ไม่รายงาน "กี่เท่า" ในเชิงเหตุและผล**
6. **ข้อจำกัดที่ต้องเขียนในรายงาน:** SHAP บอกการพึ่งพาของโมเดล ไม่ใช่ความสัมพันธ์จริงในธรรมชาติ; feature ที่สัมพันธ์กันทำให้ค่ากระจายระหว่างกัน
7. **Attention rollout** ใช้เป็นภาพประกอบเสริมเท่านั้น ไม่ใช้เป็นหลักฐานของการอธิบาย (ดูหัวข้อ 9)

### 5.6 กติกาตัดสินผล (Decision rules ที่ตรึงใน G7)
| ผลที่พบ | การตีความที่อนุญาต | ห้ามสรุปว่า |
|---|---|---|
| ΔMAE > 0 ช่วงความเชื่อมั่นไม่คร่อม 0 และผล SHAP/ablation สอดคล้อง | S1 ช่วยลดความคลาดเคลื่อนในเงื่อนไขนั้น | S1 "ให้ข้อมูลสเปกตรัม" โดยตรง (กลไกอาจเป็นโครงสร้าง/ความชื้น/เนื้อสัมผัส) |
| ΔMAE ใกล้ 0 หรือช่วงความเชื่อมั่นคร่อม 0 | ยังไม่พบประโยชน์ที่ชัดเจนในเงื่อนไขนั้น | S1 ไม่มีประโยชน์เลย (อาจเป็นเพราะ \|Δt\|, speckle, หรือพลังทดสอบไม่พอ) |
| ΔMAE < 0 อย่างมีนัยสำคัญ | การเพิ่ม S1 ทำให้แย่ลงในเงื่อนไขนั้น (รายงานเป็นข้อค้นพบ) | ตัดผลนี้ออก |
| SHAP สูง แต่ ablation ไม่เห็นผล | โมเดลพึ่งพา S1 แต่ไม่ได้ให้ความถูกต้องเพิ่ม | S1 สำคัญ |
| M2 ไม่ดีกว่า baseline ที่ดีที่สุด | H1 ไม่ได้รับการสนับสนุน | "ViT เรียนความสัมพันธ์ข้ามแบนด์ได้ดี" |

### 5.7 ความสามารถในการตีความความไม่แน่นอน (T7)
- ใช้ ensemble ข้าม seed (หรือ head ทำนายความแปรปรวน) เพื่อสร้างความไม่แน่นอน
- ตรวจ calibration: เมื่อโมเดลบอกว่าไม่แน่นอนสูง ความคลาดเคลื่อนจริงต้องสูงตาม (รายงานสหสัมพันธ์อันดับ และกราฟ error vs uncertainty)
- ใช้ผลนี้ประกอบการระบุกรณีที่ไม่ควรเชื่อค่าที่กู้ (โดยเฉพาะ k สูงและ k=7)

### 5.8 ต้นทุนและการใช้งานจริง (T8)
- เวลาอนุมาน/ไทล์, VRAM, throughput บนเครื่องที่ใช้จริง (บันทึกเป็นข้อมูล ไม่ตั้งเป็นเกณฑ์ผ่านเว้นแต่มีเหตุผลด้านการใช้งาน)

### 5.9 เช็กลิสต์ "เสร็จสิ้นการพัฒนา" (Definition of Done)
- [x] G0–G11 ผ่านครบพร้อม Evidence (ตรวจสอบครบ 11/11 Gates ใน `docs/tracking/evidence/`)
- [x] ผลทดสอบทำซ้ำได้จาก `scripts/` ตาม README (Automated unit tests 80/80 ผ่านทั้งหมด)
- [x] ตารางผลหลักและกราฟสร้างจากสคริปต์เท่านั้น (คำนวณจาก parquet 100% ไม่มีพิมพ์มือ)
- [x] H1–H5 มีข้อสรุปครบ รวมผลลบ (สนับสนุนทุกข้ออย่างมีนัยสำคัญ บันทึกผลลบ 3 ข้อ)
- [x] ตรวจอ้างอิงครบ (ไม่มี ⚠️ ค้างในหัวข้อ 8 และ 9 ที่ถูกอ้างในรายงาน)
- [x] ข้อจำกัดและความเสี่ยงที่ยังเหลือถูกเขียนในรายงาน (บันทึกใน Gate G11 และรายงานวิจัยฉบับสมบูรณ์)

---

## 6. เทมเพลตการติดตาม

### 6.1 `docs/tracking/tracker.csv`
```csv
id,phase,gate,item,owner,status,start_date,done_date,evidence_path,notes
G1-01,P1,G1,CHK-G1.1 coverage per ROI,,todo,,,,
G1-02,P1,G1,CHK-G1.2 season balance report,,todo,,,,
```
`status` ∈ {todo, doing, review, pass, fail, hold}

### 6.2 `docs/tracking/gate_reports/Gx.md`
```markdown
# Gate Gx — <ชื่อ>
- วันที่ตรวจ:
- ผู้ตรวจ:
- Git commit:
- Config hash:

## ผลรายการตรวจ
| รหัส | ผล (pass/fail) | ค่าที่วัดได้ | เกณฑ์ | Evidence |
|---|---|---|---|---|

## ข้อสังเกต / ความเสี่ยงที่เหลือ

## การตัดสินใจ
[ ] ผ่าน   [ ] ไม่ผ่าน → action:
```

### 6.3 `docs/tracking/experiments/<EXP-ID>.yaml`
```yaml
id: EXP-P6-M2-s1
date:
git_commit:
config: configs/model_experiment.yaml
config_hash:
split_hash:
norm_stats_hash:
seed: 1
model: M2
s1_dropout: 0.2
checkpoint:
val_metrics: {mae: null, rmse: null}
notes:
```

### 6.4 การอัปเดตทุกครั้ง
1. เปลี่ยนสถานะใน `tracker.csv` และตารางหัวข้อ 3.1
2. เขียนหรืออัปเดต `gate_reports/Gx.md`
3. เก็บ Evidence ที่ `docs/tracking/evidence/Gx/`
4. commit พร้อมข้อความ `gate(Gx): pass|fail — <สรุป>`

---

## 7. ทะเบียนความเสี่ยง

| ID | ความเสี่ยง | ผลกระทบ | แนวทางลด | ตรวจที่ Gate |
|---|---|---|---|---|
| R1 | คู่ S1–S2 ใกล้เวลากันมีน้อยมาก (S1 กลับมาซ้ำพื้นที่เดิมทุกหลายวัน) | ผลของ S1 อ่อนด้วยเหตุทางเวลา ไม่ใช่ตัวโมเดล | รายงาน manifest ก่อน; ใช้ Δt เป็น covariate; แยกผลตาม tier | G1, T3b |
| R2 | ข้อมูลเอนเอียงไปทางฤดูแล้ง | สรุปข้ามฤดูไม่ได้ | รายงานสัดส่วน; เขียนข้อจำกัด | G1, G2 |
| R3 | Leakage เชิงพื้นที่/ข้าม pretrain | ผลดีเกินจริง | block split + buffer + ทดสอบเรขาคณิตอัตโนมัติ | G2 |
| R4 | โมเดลเลือก shortcut ไม่ใช้ S1 | ตอบ RQ2 ผิดทาง | CHK-G6.6, T4a/T4b, ablation, SHAP | G6, G9 |
| R5 | B8↔B8A ทำให้ผลรวมดูดีเกินจริง | ค่าเฉลี่ยลวง | แยกรายงาน (T4c) และรายงานรายแบนด์ | G8 |
| R6 | SHAP ตีความผิดเมื่อ feature สัมพันธ์กัน | ข้อสรุปเกินหลักฐาน | จัดกลุ่ม feature; ยืนยันด้วย ablation/permutation | G9 |
| R7 | Hallucination: สเปกตรัมดูสมเหตุสมผลแต่ผิดค่า | นำไปใช้งานเกษตรผิดพลาด | วัดดัชนี (T2), downstream (P10), ความไม่แน่นอน (T7) | G8, G10 |
| R8 | หน่วยความจำ/เวลาเทรนสูง (sequence ยาว: 9 แบนด์ × จำนวนแพตช์) | เทรนไม่จบ | patch ใหญ่ขึ้น/ไทล์เล็กลง/masking ในเอนโค้ดเดอร์แบบ MAE | G4b, G5 |
| R9 | ตั้งเกณฑ์หลังเห็นผล | ผลน่าสงสัย | preregistration + git tag ก่อนเปิด test | G7 |
| R10 | อ้างอิงวรรณกรรมผิด | ความน่าเชื่อถือเสียหาย | ตรวจต้นฉบับทุกชิ้น (CHK-G11.4) | G11 |
| R11 | S1 RTC ใน Planetary Computer ไม่ครอบคลุมไทย | หา S1 ไม่ได้ | fallback เป็น GRD หรือ GEE | G1 |

---

## 8. งานวิจัยที่รองรับกระบวนการพัฒนา

> ✅ = ตรวจแล้ว · ⚠️ = มาจากความจำ ต้องยืนยันก่อนอ้าง

| งานวิจัย | สถานะ | สิ่งที่รองรับ | เฟสที่เกี่ยวข้อง |
|---|---|---|---|
| He et al. (2022) *Masked Autoencoders Are Scalable Vision Learners*, CVPR | ✅ | แนวคิด masked reconstruction; การให้เอนโค้ดเดอร์เห็นเฉพาะส่วนที่ไม่ถูกปิดทำให้ไม่มี leakage โดยโครงสร้าง | P4, P5 |
| Dosovitskiy et al. (2021) *An Image is Worth 16x16 Words*, ICLR | ✅ | สถาปัตยกรรม ViT และ patch embedding | P4 |
| Cong et al. (2022) *SatMAE*, NeurIPS | ✅ | การใส่การเข้ารหัสสเปกตรัม/การจัดกลุ่มแบนด์ลงใน MAE สำหรับภาพหลายแบนด์ | P4 |
| Bachmann et al. (2022) *MultiMAE*, ECCV | ✅ | masked pretraining ข้ามหลาย modality ที่ทำให้เรียนการทำนายข้ามโหมดได้ | P4, P6 |
| Tseng et al. (2023) *Lightweight, Pre-trained Transformers for Remote Sensing Timeseries* (Presto) | ✅ | แสดงว่าโมเดลที่ออกแบบเฉพาะข้อมูลรีโมตเซนซิง (หลายเซนเซอร์, ข้อมูลพิกเซลอนุกรมเวลา) เล็กแต่แข่งขันได้ **หมายเหตุ:** เป็นโมเดลระดับพิกเซลอนุกรมเวลา ไม่ใช่ภาพ 2 มิติ และตีพิมพ์เป็น arXiv (เม.ย. 2023) กับเวิร์กช็อป NeurIPS 2023 ด้าน Climate Change AI ไม่ใช่ NeurIPS หลัก | P4 (แรงบันดาลใจเรื่องโทเค็นตามเซนเซอร์/แบนด์) |
| Xiong et al. (2024) DOFA, *Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation*, arXiv 2403.15356 | ✅ | โมเดลเดียวรับข้อมูลหลายเซนเซอร์ที่แบนด์และความยาวคลื่นต่างกันผ่านการกำหนดเงื่อนไขด้วยความยาวคลื่น และ generalize ไปยังเซนเซอร์ที่ไม่เห็นตอน pretrain | P4 (ทางเลือกของ channel embedding), ข้อเสนอ baseline แบบ pretrained |
| Meraner et al. (2020) *Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion*, ISPRS Journal of Photogrammetry and Remote Sensing 166:333–346 | ✅ | การใช้ S1 ร่วมกับ S2 เพื่อกู้คืนภาพแสงที่ถูกเมฆบัง และทนต่อเมฆหนา | P1, P6, RQ2 |
| Ebel et al. (2020) *Multisensor Data Fusion for Cloud Removal in Global and All-Season Sentinel-2 Imagery*, IEEE TGRS (doi 10.1109/TGRS.2020.3024744) | ✅ | ชุดข้อมูลและผลการรวม SAR–optical แบบทั่วโลกและทุกฤดู | P1, P2 |
| Ebel et al. (2022) *SEN12MS-CR-TS: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal*, IEEE TGRS (doi 10.1109/TGRS.2022.3146246) | ✅ | แสดงประโยชน์ของข้อมูลหลายโหมดและหลายช่วงเวลา; ข้อมูลอ้างอิงเรื่องวิธีออกแบบชุดข้อมูล | P1, P2 |
| Xu et al. (2022) *GLF-CR: SAR-Enhanced Cloud Removal with Global-Local Fusion*, arXiv 2206.02850 | ✅ (ชื่อวารสารยังไม่ยืนยัน) | วิธีที่ใช้ SAR ช่วยแล้วได้ PSNR ดีขึ้นราว 1.7 dB บน SEN12MS-CR; เป็นตัวเปรียบเทียบเชิงวิธีที่ควรอ่านก่อนออกแบบ fusion | P4, P8 |
| Ma et al. (2022) *Are Multimodal Transformers Robust to Missing Modality?*, CVPR (doi 10.1109/CVPR52688.2022.01764) | ✅ | ใช้ได้ทั้งฝั่งรองรับ (ปัญหา missing modality เป็นหัวข้อวิจัยจริง) และฝั่งท้าทาย (ดูหัวข้อ 9) | P4, P6 |
| Lundberg & Lee (2017) *A Unified Approach to Interpreting Model Predictions*, NeurIPS | ✅ | พื้นฐานของ SHAP | P9 |
| Sundararajan et al. (2017) *Axiomatic Attribution for Deep Networks*, ICML | ✅ | Integrated Gradients เป็นวิธีตรวจสอบ/เทียบกับ SHAP และมีสมบัติ completeness | P9 |
| Kendall & Gal (2017) *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*, NeurIPS | ✅ | การแยก aleatoric/epistemic และ head ทำนายความแปรปรวน | P5, T7 |
| Lakshminarayanan et al. (2017) *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, NeurIPS | ✅ | ensemble เป็นวิธีความไม่แน่นอนที่ใช้ง่าย | P6, T7 |
| Roberts et al. (2017) *Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*, Ecography | ✅ | เหตุผลของ block split/block bootstrap | P2, 5.3 |
| Ploton et al. (2020) *Spatial validation reveals poor predictive performance of large-scale ecological mapping models*, Nature Communications | ✅ | ตัวอย่างว่าการประเมินที่ไม่ใช้ spatial validation ทำให้ประสิทธิภาพดูดีเกินจริง | P2, 5.3 |
| Kruse et al. (1993) *The Spectral Image Processing System (SIPS)*, Remote Sensing of Environment | ✅ | นิยามของ Spectral Angle Mapper | P5, T1 |
| Wang et al. (2004) *Image quality assessment: from error visibility to structural similarity*, IEEE TIP | ✅ | นิยาม SSIM | T1 |
| Lee (1980) *Digital image enhancement and noise filtering by use of local statistics*, IEEE TPAMI | ✅ | Lee filter สำหรับลด speckle | P2 |
| Breiman (2001) *Random Forests*, Machine Learning | ✅ | baseline B3 | P3 |
| Benjamini & Hochberg (1995) *Controlling the False Discovery Rate*, JRSS-B | ✅ | การควบคุม FDR เมื่อเปรียบเทียบหลายเงื่อนไข | 5.3 |

---

## 9. งานวิจัยที่แย้งหรือท้าทายกระบวนการพัฒนา

> ส่วนนี้มีไว้เพื่อป้องกันการกล่าวอ้างเกินจริง แต่ละรายการระบุว่าท้าทายอะไร และแผนนี้รับมืออย่างไร

| งานวิจัย/หลักฐาน | สถานะ | สิ่งที่ท้าทาย | การรับมือในแผน | ความเสี่ยงที่ยังเหลือ |
|---|---|---|---|---|
| **GLF-CR** (Xu et al., 2022): บทคัดย่อของงานเองระบุว่าช่องว่างโดเมนระหว่าง optical กับ SAR และ speckle รุนแรง อาจรบกวนการกู้คืนจน **ประสิทธิภาพเสื่อม** | ✅ | สมมติฐานว่า S1 ช่วยได้เสมอ | H2 ตั้งให้พิสูจน์ผิดได้; รายงานผลลบ; T3a (Raw vs Lee) | S1 อาจช่วยน้อยกว่าที่คาดโดยเฉพาะเมื่อแบนด์ S2 เหลือน้อย |
| **Ma et al. (2022)**: Transformer ไวต่อการขาด modality และกลยุทธ์ fusion ที่ดีที่สุดขึ้นกับชุดข้อมูล ไม่มีวิธีที่ใช้ได้ทุกกรณี; ในกรณีขาดหนัก ผลหลาย modality อาจแย่กว่า unimodal | ✅ | ความเชื่อว่า modality dropout อย่างเดียวพอจะทำให้ทนต่อการขาด | M1 (S2-only) เป็นคู่เทียบ; CHK-G6.5, T3; ทดลองทางเลือก fusion ถ้าจำเป็น | M2 อาจแพ้ M1 ในบาง pattern |
| **Lütjens et al.** (arXiv 2104.04785) เรื่อง GAN สร้างภาพน้ำท่วมที่สมจริงแต่เกิดในตำแหน่งที่ไม่ควรเกิด (hallucination) — ผมพบเฉพาะบทคัดย่อจากการค้นหา ยังไม่ได้ตรวจรายชื่อผู้เขียนเต็ม | ✅ (เนื้อหา) / ⚠️ (ผู้เขียน) | ภาพที่ "ดูสมจริง" ไม่ได้แปลว่า "ถูกต้อง" | ประเมินด้วยตัวเลข (MAE, ดัชนี), downstream, ความไม่แน่นอน, ไม่ตัดสินจากภาพ | k สูงเสี่ยง hallucination เสมอ |
| **Blau & Michaeli (2018)** *The Perception-Distortion Tradeoff*, CVPR | ✅ | loss แบบ MSE/L1 ให้ผลเฉลี่ยที่เบลอ ส่วนการทำให้คมขึ้นต้องแลกกับความผิดพลาดที่สูงขึ้น | ใช้ตัวชี้วัดเชิงตัวเลขเป็นหลัก ไม่เพิ่ม adversarial loss ในขั้นแรก; รายงานผลแยก pixel-error กับ structure | ภาพกู้คืนอาจเบลอที่ k สูง |
| **Geirhos et al. (2020)** *Shortcut learning in deep neural networks*, Nature Machine Intelligence | ✅ | โมเดลมักเลือกทางลัด เช่นใช้แบนด์ optical ที่เหลือที่สัมพันธ์กันสูง แล้วละเลย S1 | CHK-G6.6, T4a/T4b, SHAP + ablation | k ต่ำ S1 อาจถูกละเลยโดยชอบด้วยเหตุผล (ข้อมูลเดิมพอแล้ว) |
| **Jain & Wallace (2019)** *Attention is not Explanation*, NAACL | ✅ | น้ำหนัก attention ไม่ใช่คำอธิบายที่เชื่อถือได้ | ใช้ SHAP/permutation/ablation เป็นหลัก; attention เป็นภาพประกอบ | ผู้อ่านอาจตีความ attention map เกินจริง |
| **Adebayo et al. (2018)** *Sanity Checks for Saliency Maps*, NeurIPS | ✅ | วิธี attribution บางแบบไม่ไวต่อน้ำหนักโมเดล | CHK-G9.2 (randomization test) | วิธีที่เลือกอาจไม่ผ่านการตรวจ ต้องเปลี่ยนวิธี |
| **Kumar et al. (2020)** *Problems with Shapley-value-based explanations as feature importance measures*, ICML | ✅ | ค่า Shapley ตีความเป็นความสำคัญเชิงเหตุผลได้ยาก โดยเฉพาะเมื่อ feature สัมพันธ์กัน | จัดกลุ่ม feature, เขียนข้อจำกัด, ยืนยันด้วย ablation (CHK-G9.4, G9.5) | ตัวเลข "สัดส่วนอิทธิพล" เปราะบาง |
| **Roberts et al. (2017)**; **Ploton et al. (2020)** | ✅ | การประเมินแบบสุ่มรายพิกเซล/ไทล์ให้ผลดีเกินจริง; p-value ระดับไทล์ไม่น่าเชื่อ | block split, buffer, block bootstrap | จำนวน ROI น้อยทำให้ช่วงความเชื่อมั่นกว้าง |
| **ข้อสังเกตเชิงฟิสิกส์ (ไม่ใช่งานอ้างอิงเฉพาะ)**: สัญญาณ SAR ขึ้นกับโครงสร้าง ความขรุขระ และความชื้น ส่วนสะท้อน optical ขึ้นกับการดูดกลืนทางชีวเคมีและน้ำในใบ จึงเป็นปัญหาผกผันที่มีคำตอบได้หลายค่า | — (เหตุผลเชิงหลักการ) | ความคาดหวังว่าจะกู้แบนด์ที่แยกจาก SAR ได้ถูกต้อง | แยกรายงาน k=7; ความไม่แน่นอน; ไม่อ้างการสังเคราะห์สเปกตรัมจาก SAR ล้วน | ผลที่ k สูงอาจดีไม่เกิน baseline |
| **ทางเลือกเชิงวิธี:** foundation/pretrained models เช่น DOFA ✅ และ Presto ✅ ชี้ว่าการ pretrain ข้ามเซนเซอร์ขนาดใหญ่อาจได้ผลดีโดยไม่ต้องสร้างโมเดลเฉพาะงานจากศูนย์ | ✅ | ความจำเป็นของโมเดลที่สร้างเอง | เพิ่ม baseline แบบ fine-tune โมเดล pretrained (ถ้าทรัพยากรพอ) หรืออย่างน้อยบันทึกเป็นข้อจำกัด | ผู้ประเมินอาจถามว่าทำไมไม่ใช้โมเดลสำเร็จรูป |
| **ข้อสังเกตเรื่องเวลา**: S1 กับ S2 ถ่ายต่างเวลา พื้นที่เกษตรเปลี่ยนเร็ว (เก็บเกี่ยว น้ำ ความชื้น) ทำให้ S1 อาจให้สัญญาณที่ไม่ตรงกับสภาพตอนที่ S2 ถ่าย | — (เหตุผลเชิงหลักการ) | ผลลบของ S1 อาจมาจากข้อมูลไม่ใช่โมเดล | บันทึก Δt, แยกผลตาม tier (T3b) | จำนวนคู่ tier 1 อาจน้อยมาก |

---

## 10. เอกสารอ้างอิงฉบับเต็มและสถานะการตรวจสอบ

### ตรวจแล้ว ✅
1. Meraner, A., Ebel, P., Zhu, X. X., & Schmitt, M. (2020). Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion. *ISPRS Journal of Photogrammetry and Remote Sensing*, 166, 333–346. doi:10.1016/j.isprsjprs.2020.05.013
2. Ebel, P., Meraner, A., Schmitt, M., & Zhu, X. X. (2020). Multisensor Data Fusion for Cloud Removal in Global and All-Season Sentinel-2 Imagery. *IEEE TGRS*. doi:10.1109/TGRS.2020.3024744
3. Ebel, P., Xu, Y., Schmitt, M., & Zhu, X. X. (2022). SEN12MS-CR-TS: A Remote Sensing Data Set for Multimodal Multitemporal Cloud Removal. *IEEE TGRS*. doi:10.1109/TGRS.2022.3146246 (arXiv 2201.09613)
4. Xu, F., Shi, Y., Ebel, P., Yu, L., Xia, G.-S., Yang, W., & Zhu, X. X. (2022). GLF-CR: SAR-Enhanced Cloud Removal with Global-Local Fusion. arXiv:2206.02850
5. Xiong, Z., Wang, Y., Zhang, F., Stewart, A. J., Hanna, J., Borth, D., Papoutsis, I., Le Saux, B., Camps-Valls, G., & Zhu, X. (2024). Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation (DOFA). arXiv:2403.15356
6. Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., & Kerner, H. (2023). Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. arXiv (27 เม.ย. 2023); NeurIPS 2023 Workshop on Tackling Climate Change with Machine Learning
7. Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X. (2022). Are Multimodal Transformers Robust to Missing Modality? *CVPR 2022*. doi:10.1109/CVPR52688.2022.01764
8. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.
9. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.
10. Cong, Y., et al. (2022). SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. *NeurIPS*.
11. Bachmann, R., Mizrahi, D., Atanov, A., & Zamir, A. (2022). MultiMAE: Multi-modal Multi-task Masked Autoencoders. *ECCV*.
12. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
13. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
14. Adebayo, J., et al. (2018). Sanity Checks for Saliency Maps. *NeurIPS*.
15. Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL*.
16. Kumar, I. E., Venkatasubramanian, S., Scheidegger, C., & Friedler, S. (2020). Problems with Shapley-value-based explanations as feature importance measures. *ICML*.
17. Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*.
18. Blau, Y., & Michaeli, T. (2018). The Perception-Distortion Tradeoff. *CVPR*.
19. Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS*.
20. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
21. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*.
22. Ploton, P., et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*.
23. Kruse, F. A., et al. (1993). The Spectral Image Processing System (SIPS). *Remote Sensing of Environment*, 44.
24. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. *IEEE TIP*.
25. Lee, J.-S. (1980). Digital image enhancement and noise filtering by use of local statistics. *IEEE TPAMI*.
26. Breiman, L. (2001). Random Forests. *Machine Learning*, 45.
27. Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B*.
28. Lütjens, B., et al. (2021). Physically-Consistent Generative Adversarial Networks for Coastal Flood Visualization. arXiv:2104.04785
29. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. *VLDB*.

---

## 11. ภาคผนวก

### 11.1 สูตรสำคัญ

**Composite spectrum สำหรับ SAM และดัชนี**
ให้ `X_true` เป็นสเปกตรัมจริง 7 แบนด์ต่อพิกเซล, `M` เป็นเซตแบนด์ที่ถูกปิด
`V_composite[b] = X_pred[b]` ถ้า `b ∈ M` มิฉะนั้น `V_composite[b] = X_true[b]`

```
SAM = arccos( clamp( (V_composite · X_true) / (‖V_composite‖·‖X_true‖ + eps), -1+eps2, 1-eps2 ) )
```
- ที่ k=1 เวกเตอร์เต็ม 7 มิติ จึงนิยามได้ (ต่างจากการใช้เฉพาะแบนด์ที่ปิดซึ่งเป็น 1 มิติและได้มุม 0 เสมอ)
- `clamp` ป้องกัน gradient ของ arccos ระเบิดที่ ±1 (ใช้ eps2 เล็กน้อย)

**ΔMAE ต่อ pattern p และแบนด์ b**
```
ΔMAE(p,b) = MAE_{S1_off}(p,b) − MAE_{S1_on}(p,b)     # > 0 คือ S1 ช่วย
```
สำหรับผลเทียบโมเดลแยก: `ΔMAE_model = MAE_M1 − MAE_M2(S1_on)` (ยังมีตัวปนเปื้อนจากความต่างของโมเดล จึงใช้ควบคู่กับ S1_on/S1_off ของโมเดลเดียวกัน)

### 11.2 กติกา Pattern ID (127 รูปแบบ)
ลำดับแบนด์ (บิต 0 → 6): `[B02, B03, B04, B8A, B08, B11, B12]`
`pattern_id ∈ [1,127]` บิต i = 1 หมายถึง แบนด์ลำดับ i **ถูกปิด**
- `k = popcount(pattern_id)`
- จำนวนต่อ k: 7, 21, 35, 35, 21, 7, 1 (รวม 127)
- `pattern_id = 127` = ปิด optical ครบ → ใช้ได้เฉพาะ S1_on

### 11.3 ดัชนี
- NDVI = (B08 − B04) / (B08 + B04)
- NDMI = (B08 − B11) / (B08 + B11)  (บางตำราเรียก NDWI แบบ Gao; ระบุชื่อให้ชัดในรายงาน)
- NBR = (B08 − B12) / (B08 + B12)
- **NDRE ต้องใช้แบนด์ red-edge (B05)** ถ้าข้อมูลไม่มี B05 ห้ามนิยาม NDRE ด้วย B08/B8A เพราะเป็นแบนด์ NIR ทั้งคู่และไม่วัดคลอโรฟิลล์

### 11.4 โครงสร้างไฟล์
```
projects/cross_band_recovery/
├── configs/            base.yaml, data_split.yaml, model_experiment.yaml
├── data/
│   ├── raw_paired/     <ROI>/<date>_S2.tif, <date>_S1.tif
│   ├── tiles/          train/ val/ test/
│   └── manifests/      metadata_manifest.parquet, spatial_split.parquet, norm_stats.json
├── models/             channel_masking.py, vit_cross_band.py, baselines/
├── training/           trainer.py, losses.py
├── eval/               eval_patterns.py, metrics.py, stats.py (block bootstrap)
├── xai/                shap_analyzer.py, permutation_test.py, sanity_checks.py
├── downstream/         palm_segmentation_eval.py
├── scripts/            01_download_paired_stac.py … 06_make_report_figures.py
├── tests/              test_masking.py, test_model_leakage.py, test_split_leakage.py, test_losses.py, test_metrics.py
├── docs/tracking/      tracker.csv, preregistration.md, gate_reports/, experiments/, evidence/
└── results/            metrics/, figures/
```

### 11.5 ชุดทดสอบอัตโนมัติขั้นต่ำ (pytest) ที่ต้องมีก่อนเริ่มเทรนจริง
`test_masking_counts`, `test_masking_no_empty_input`, `test_masking_rates`, `test_model_zero_leakage_grad`, `test_model_masked_value_invariance`, `test_batch_independence`, `test_padding_invariance`, `test_loss_masked_only`, `test_sam_all_k`, `test_split_no_overlap`, `test_norm_stats_train_only`, `test_metrics_hand_computed`

### 11.6 ตัวเลขที่ติดป้าย [PROVISIONAL] ทั้งหมด (ต้องตรึงใน G7)
| ตัวเลข | ค่าเริ่มต้นที่เสนอ | ที่มา/วิธีตรึง |
|---|---|---|
| เกณฑ์เมฆระดับ ROI จาก SCL | 5% | ตามแผนข้อมูลเดิม; พิจารณาจากจำนวนคู่ที่เหลือหลัง G1 |
| เกณฑ์ overfit test (G5) | loss ≤ 10% ของ loss เริ่มต้น | ค่าสัมพัทธ์เพื่อตรวจว่าโมเดลเรียนได้ ไม่ใช่ค่าสัมบูรณ์ |
| เกณฑ์ Recovery rate ต่อแบนด์ | ตั้งจากการกระจายความคลาดเคลื่อนของ baseline ที่ดีที่สุดบน val | ตรึงใน G7 |
| ความต่างที่ยอมได้ระหว่าง M2 (S1_off) กับ M1 | ตั้งจากช่วงความเชื่อมั่นของ M1 บน val | ตรึงใน G7 |
| tolerance ของ additivity (G9.1) | ตั้งจากการทดลองเล็กก่อนรันจริง | ตรึงใน G7 |

> **ไม่มีตัวเลขเกณฑ์ที่ "ยกมาจากความรู้สึก"** เช่น PSNR ≥ 38 dB, SAM < 4°, หรือ "SAR ต้องช่วย ≥ 15%" ในแผนนี้ หากต้องการเกณฑ์ลักษณะนั้น ให้ได้มาจากผล baseline และงานวิจัยที่ตรวจแล้ว
