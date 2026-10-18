# แผนแม่บทการพัฒนาโมดูลโมเดลใหม่: Cross-Band Masked Autoencoder & SAR-Guided Recovery

**สถานะ:** เอกสารแผนแม่บทและการตรวจสอบเชิงวิชาการ (Master Development & Verification Plan)  
**วันที่จัดทำ:** 17 ตุลาคม 2026  
**เป้าหมายโครงการ:** พัฒนาสถาปัตยกรรมโมเดล Transformer สำหรับการเรียนรู้ความสัมพันธ์ข้ามแถบคลื่น (Cross-Band Relationship Learning) และการฟื้นฟูแถบคลื่นแสง Sentinel-2 (B2, B3, B4, B8A, B8, B11, B12) ที่สูญหายหรือถูกเมฆบดบัง โดยใช้ข้อมูลเรดาร์ Sentinel-1 (VV, VH) เป็นตัวนำทางโครงสร้าง พร้อมประเมินผลครบทั้ง **127 รูปแบบ (Combinatorial Missing Band Patterns: $2^7 - 1$)**  
**ที่ตั้งโมดูลเป้าหมาย:**  
- `models/channel_masking.py` (หรือ `lib2_vitpalm/model/channel_masking.py`)
- `models/vit_cross_band.py` (หรือ `lib2_vitpalm/model/vit_cross_band.py`)

---

## สารบัญ (Table of Contents)
1. [ภาพรวมเชิงกลยุทธ์และการเปลี่ยนกระบวนทัศน์ (Paradigm Shift)](#1-ภาพรวมเชิงกลยุทธ์และการเปลี่ยนกระบวนทัศน์)
2. [สถาปัตยกรรมและข้อกำหนดเชิงคณิตศาสตร์ (Mathematical Specifications)](#2-สถาปัตยกรรมและข้อกำหนดเชิงคณิตศาสตร์)
   - 2.1 [โมดูล `channel_masking.py`](#21-โมดูล-channel_maskingpy)
   - 2.2 [โมดูล `vit_cross_band.py` (Channel Embedding & Reconstruction Head)](#22-โมดูล-vit_cross_bandpy)
3. [ตารางขั้นตอนการพัฒนาและ Quality Gates (กาดตรวจสอบความถูกต้อง)](#3-ตารางขั้นตอนการพัฒนาและ-quality-gates)
   - [Gate M0: การตรวจสอบโครงสร้างและมิติข้อมูล (Shape & Masking Integrity)](#gate-m0-การตรวจสอบโครงสร้างและมิติข้อมูล)
   - [Gate M1: การตรวจสอบการไหลของเกรเดียนต์และคณิตศาสตร์ (Gradient Flow & Math Guard)](#gate-m1-การตรวจสอบการไหลของเกรเดียนต์และคณิตศาสตร์)
   - [Gate M2: การทดสอบ Overfitting บนชุดข้อมูลเล็ก (Single-Batch Sanity Gate)](#gate-m2-การทดสอบ-overfitting-บนชุดข้อมูลเล็ก)
   - [Gate M3: การพิสูจน์อิทธิพลของ SAR (Cross-Modal Ablation Gate)](#gate-m3-การพิสูจน์อิทธิพลของ-sar)
   - [Gate M4: การประเมินผลอัตโนมัติครบ 127 รูปแบบ (127-Pattern Benchmark Gate)](#gate-m4-การประเมินผลอัตโนมัติครบ-127-รูปแบบ)
4. [ระเบียบวิธีทดสอบโมเดลหลักหลังสิ้นสุดการพัฒนา (Post-Development Testing Protocol)](#4-ระเบียบวิธีทดสอบโมเดลหลักหลังสิ้นสุดการพัฒนา)
5. [การทบทวนวรรณกรรม: งานวิจัยที่รองรับกระบวนการพัฒนา (Supporting Literature)](#5-การทบทวนวรรณกรรม-งานวิจัยที่รองรับกระบวนการพัฒนา)
6. [การทบทวนวรรณกรรม: งานวิจัยที่แย้งและวิพากษ์วิจารณ์ข้อจำกัด (Critical Literature & Counter-Arguments)](#6-การทบทวนวรรณกรรม-งานวิจัยที่แย้งและวิพากษ์วิจารณ์ข้อจำกัด)
7. [แผนที่การสร้างไฟล์และโค้ดตัวอย่างสำหรับการทดสอบ (Implementation & Test File Roadmap)](#7-แผนที่การสร้างไฟล์และโค้ดตัวอย่างสำหรับการทดสอบ)

---

## 1. ภาพรวมเชิงกลยุทธ์และการเปลี่ยนกระบวนทัศน์

### 1.1 ที่มาและการปรับทิศทางทางวิจัย (Strategic Motivation)
จากการประเมินร่วมกับผู้เชี่ยวชาญ (พี่ Want) โครงการได้เปลี่ยนโฟกัสจากการทำ Inpainting เชิงพื้นที่แบบดั้งเดิม (Cloud-MAE Spatial Inpainting) ที่มักเผชิญปัญหาภาพเบลอ (blurry textures) หรือการสร้างภาพลวงตา (hallucination) ในบริเวณเมฆหนา มาสู่การมุ่งเน้น **"การเรียนรู้ความสัมพันธ์ระหว่างแถบคลื่นและการฟื้นฟูสเปกตรัมโดยใช้เรดาร์เหนี่ยวนำ" (Cross-Band Spectral Relationship Learning & SAR-Guided Missing Band Recovery)**

```
               [อินพุตดาวเทียม 9 แบนด์]
         S2 Optical (7 แบนด์) + S1 SAR (2 แบนด์)
                        │
                        ▼
           ┌──────────────────────────┐
           │   channel_masking.py     │  สุ่มปิดแบนด์ Optical 1–7 แบนด์
           │  (Masking & S1 Dropout)  │  + สุ่มดรอป S1 (p=0.2)
           └────────────┬─────────────┘
                        │ ข้อมูลแบนด์ที่มองเห็น (Visible Channels) + Mask Indices
                        ▼
           ┌──────────────────────────┐
           │     vit_cross_band.py    │  • Band-Specific Patch Projection
           │   (Cross-Band ViT)       │  • Learnable Channel Embedding E_band
           │                          │  • Cross-Band Self-Attention
           └────────────┬─────────────┘
                        │ Latent Tokens (D=512)
                        ▼
           ┌──────────────────────────┐
           │   Reconstruction Head    │  แมปเวกเตอร์แฝงกลับเป็นค่าสะท้อน
           │   (Per-Band Decoder)     │  สเปกตรัมของแบนด์ที่ถูกปิด (0.0 – 1.0)
           └────────────┬─────────────┘
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
[วัดผล 127 รูปแบบที่ขาดหาย]      [Downstream Palm Segmentation]
MAE, RMSE, PSNR, SSIM, SAM       mIoU, F1 บนภาพที่ฟื้นฟู
```

### 1.2 วัตถุประสงค์หลักเชิงวิศวกรรม
1. **แบบจำลองโมดูลาร์ระดับสูง:** ไม่แตะต้องโค้ดเดิมใน `lib2_vitpalm/model/cloud_mae.py` เพื่อป้องกัน Regression แต่สร้างโมดูลใหม่ที่เชื่อมต่อเข้ากับระบบได้อย่างเป็นอิสระ
2. **รองรับสภาวะ Missing Bands ทุกมิติ:** รองรับตั้งแต่กรณีแบนด์หายเพียง 1 แบนด์ (เช่น B8 โดนเมฆละออง) ไปจนถึงกรณีรุนแรงที่แบนด์แสงหายหมดทั้ง 7 แบนด์ (มีเฉพาะเรดาร์ S1 VV/VH)
3. **การทดสอบความถูกต้องอย่างเข้มงวด:** มี Quality Gates ครอบคลุมตั้งแต่ระดับ Unit Test, Gradient Flow ไปจนถึงความหมายทางสรีรวิทยาของพืช (NDVI/NDRE Preservation)

---

## 2. สถาปัตยกรรมและข้อกำหนดเชิงคณิตศาสตร์

### 2.1 โมดูล `channel_masking.py`

#### นิยามแถบคลื่น (Band Definitions)
- **Optical Bands ($\mathcal{C}_{\text{opt}}$, 7 แบนด์):**  
  $\{B2\text{ (Blue)}, B3\text{ (Green)}, B4\text{ (Red)}, B8A\text{ (Narrow NIR)}, B8\text{ (Broad NIR)}, B11\text{ (SWIR-1)}, B12\text{ (SWIR-2)}\}$  
  โดยมีดัชนี $c \in \{0, 1, 2, 3, 4, 5, 6\}$
- **SAR Bands ($\mathcal{C}_{\text{sar}}$, 2 แบนด์):**  
  $\{VV, VH\}$ มีดัชนี $s \in \{0, 1\}$ (รวมอินพุตทั้งหมด $C_{\text{total}} = 9$ แบนด์)

#### กลไกการสุ่มปิดแบนด์ (Masking Sampling Strategy)
ในระหว่างการฝึก (Training Phase):
1. **การสุ่มจำนวนแบนด์ที่จะปิด ($k$):**  
   สุ่มจำนวนแบนด์ที่ถูกบดบัง $k \sim \text{Uniform}(\{1, 2, 3, 4, 5, 6, 7\})$  
   *(หมายเหตุ: การสุ่ม $k$ แบบ Uniform ช่วยให้โมเดลฝึกรับมือกับทั้งกรณีที่แบนด์หายน้อยและหายมากอย่างเท่าเทียม ต่างจากการสุ่มอิสระแบบ Bernoulli ที่มักเกาะกลุ่มอยู่ตรงกลาง $k \approx 3.5$)*
2. **การสุ่มชุดแบนด์ที่ถูกปิด ($\mathcal{M}_{\text{opt}} \subset \mathcal{C}_{\text{opt}}$):**  
   สุ่มสับเปลี่ยนและเลือกคู่ย่อยขนาด $k$ จาก $\binom{7}{k}$ รูปแบบ
3. **การสุ่มดรอปข้อมูลเรดาร์ (SAR Modality Dropout):**  
   กำหนดค่าความน่าจะเป็น $p_{\text{sar}} \in [0.15, 0.25]$ (ค่าตั้งต้น $p_{\text{sar}} = 0.20$)  
   $$\text{Mask}_{\text{sar}} = \begin{cases} 1 \text{ (ปิด SAR ทั้ง 2 แบนด์)} & \text{ด้วยความน่าจะเป็น } p_{\text{sar}} \\ 0 \text{ (เปิดใช้งาน SAR)} & \text{ด้วยความน่าจะเป็น } 1 - p_{\text{sar}} \end{cases}$$  
   *วัตถุประสงค์:* ป้องกันไม่ให้โมเดลเกิดภาวะ Co-adaptation หรือพึ่งพา SAR มากเกินไปจนไม่สามารถฟื้นฟูแบนด์แสงได้ในวันที่ไม่มีภาพดาวเทียมเรดาร์

#### กลไกการประเมินผลแบบชัดเจน (Deterministic 127-Pattern Evaluation)
ในขั้นตอนการทดสอบ (Inference / Evaluation Phase):
- สร้างชุดเวกเตอร์ไบนารี $\mathbf{m} \in \{0, 1\}^7 \setminus \{(0,0,0,0,0,0,0)\}$ ทั้งหมด $2^7 - 1 = 127$ รูปแบบ
- สามารถเลือกทดสอบได้ 2 โหมด:  
  1. **SAR-Assisted (S1 Present):** ทดสอบทั้ง 127 รูปแบบโดยมี SAR ประกบ
  2. **Optical-Only (S1 Absent):** ทดสอบทั้ง 127 รูปแบบโดยดรอป SAR ออกทั้งหมด เพื่อเปรียบเทียบหา Delta Contribution $(\Delta_{\text{SAR}})$

#### ข้อกำหนดมิติของ Tensor (Tensor Shapes)
- อินพุต $X$: `(B, 9, H, W)` โดยที่ $9 = [B2, B3, B4, B8A, B8, B11, B12, VV, VH]$
- เอาต์พุต Mask Vector:
  - `mask_opt`: `(B, 7)` เป็น Boolean Tensor (`True` = Masked/Missing, `False` = Visible)
  - `mask_sar`: `(B, 2)` เป็น Boolean Tensor
  - `visible_indices`: รายการดัชนีของแบนด์ที่ยังมองเห็นได้ในแต่ละตัวอย่าง

---

### 2.2 โมดูล `vit_cross_band.py`

#### ก. การสร้างสเปกตรัมโทเค็นและ Channel Embedding (Spectral Tokenization & Embedding)
สถาปัตยกรรม ViT แบบดั้งเดิมจะยุบแบนด์ทั้งหมดผ่าน `nn.Conv2d(in_channels=9, embed_dim=D, kernel_size=P)` ซึ่งทำให้สูญเสียอัตลักษณ์ของแต่ละแถบคลื่นเมื่อมีบางแบนด์ขาดหายไป  
ในโมดูล `vit_cross_band.py` เราใช้โครงสร้าง **Band-Wise Patch Projection**:

1. **Band Projection:**  
   แต่ละแบนด์ $c$ มีชั้น Projection เฉพาะตัว (หรือใช้ Shared 2D Conv ขนาด $1 \times P \times P \to D$):
   $$\mathbf{t}_{c, p} = \text{Conv2D}_c(x_{c, p}) \in \mathbb{R}^D \quad \text{โดยที่ } p \in \{1, \dots, N_p\}, \, N_p = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$$
2. **Learnable Embeddings คู่ขนาน:**
   - **Spatial Positional Embedding ($\mathbf{E}_{\text{pos}}$):** ระบุพิกัดเชิงพื้นที่ของแพตช์ $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N_p \times D}$
   - **Channel / Band Embedding ($\mathbf{E}_{\text{band}}$):** ระบุอัตลักษณ์ทางฟิสิกส์และความยาวคลื่น $\mathbf{E}_{\text{band}} \in \mathbb{R}^{9 \times D}$  
     โดยแบ่งเป็น:
     $$\mathbf{E}_{\text{band}} = [\mathbf{e}_{B2}, \mathbf{e}_{B3}, \mathbf{e}_{B4}, \mathbf{e}_{B8A}, \mathbf{e}_{B8}, \mathbf{e}_{B11}, \mathbf{e}_{B12}, \mathbf{e}_{VV}, \mathbf{e}_{VH}]^T$$
3. **การรวมโทเค็น (Token Composition):**  
   สำหรับแพตช์ $p$ ในแบนด์ $c$:
   $$\mathbf{z}_{c, p} = \mathbf{t}_{c, p} + \mathbf{E}_{\text{pos}}[p] + \mathbf{E}_{\text{band}}[c]$$
   หากแบนด์ $c$ เป็นแบนด์ที่มองเห็น ($\text{Visible}$) โทเค็น $\mathbf{z}_{c, p}$ จะถูกส่งเข้าสู่ Transformer Encoder  
   ความยาวลำดับอินพุตทั้งหมด: $L_{\text{vis}} = C_{\text{vis}} \times N_p$

#### ข. โครงสร้าง Transformer Encoder (Cross-Band Self-Attention)
ใช้ Standard Multi-Head Self-Attention (MHSA) บล็อกจำนวน $L$ ชั้น:
$$\mathbf{Z}^{(l)} = \text{TransformerBlock}(\mathbf{Z}^{(l-1)})$$
เนื่องจากโทเค็นของทั้งภาพเรดาร์และภาพแสงถูกนำมาเรียงต่อกันในระนาบลำดับ (Sequence dimension) กลไก Full Self-Attention จะคำนวณ:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
ทำให้เกิด **Cross-Band Queries** อัตโนมัติ:
- โทเค็นเรดาร์ $\mathbf{z}_{VV, p}$ สามารถ Query หาบริบทจากแบนด์แสง $\mathbf{z}_{B4, p}$
- โทเค็นแบนด์แสงที่เหลืออยู่ สามารถดึงโครงสร้างขอบแปลงและความชื้นจากเรดาร์มาช่วยอนุมานได้

#### ค. ส่วนฟื้นฟูข้อมูล (Reconstruction Head)
สำหรับแต่ละแบนด์ $m \in \mathcal{M}_{\text{opt}}$ ที่ถูกบดบัง:
1. ป้อน **Learnable Mask Token ($\mathbf{e}_{\text{mask}} \in \mathbb{R}^D$)** บวกด้วย $\mathbf{E}_{\text{pos}}[p] + \mathbf{E}_{\text{band}}[m]$ เข้าสู่ Shallow Reconstruction Decoder (Transformer Blocks 2–4 ชั้น หรือ Lightweight Conv-MLP)
2. ชั้นขยายพิกัด (Pixel Reconstruction Layer):  
   แมปจากเวกเตอร์แฝงมิติ $D$ กลับสู่พิกเซลขนาด $P \times P$:
   $$\hat{x}_{m, p} = \text{Linear}(D, P^2) \quad \text{หรือ} \quad \text{ConvTranspose2d}$$
3. จัดเรียง (Unpatchify) โทเค็นกลับเป็นภาพเต็มขนาด $\hat{X}_m \in \mathbb{R}^{H \times W}$

#### ง. ฟังก์ชันวัตถุประสงค์ (Loss Formulation)
ใช้ Loss ผสมผสาน 3 ส่วน คำนวณ **เฉพาะบนแบนด์ที่ถูกปิด ($\mathcal{M}_{\text{opt}}$)**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{SAM}} \mathcal{L}_{\text{SAM}} + \lambda_{\text{VI}} \mathcal{L}_{\text{VI}}$$

1. **Reconstruction MSE Loss:**
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{|\mathcal{M}_{\text{opt}}|} \sum_{m \in \mathcal{M}_{\text{opt}}} \frac{1}{HW} \|\hat{X}_m - X_m\|_2^2$$
2. **Spectral Angle Mapper (SAM) Loss:** ป้องกันสเปกตรัมบิดเบือนเชิงมุม:
   $$\mathcal{L}_{\text{SAM}} = \frac{1}{HW} \sum_{i, j} \arccos\left( \frac{\hat{\mathbf{v}}_{i, j} \cdot \mathbf{v}_{i, j}}{\|\hat{\mathbf{v}}_{i, j}\|_2 \|\mathbf{v}_{i, j}\|_2 + \epsilon} \right)$$
3. **Vegetation Index Preservation Loss ($\mathcal{L}_{\text{VI}}$):**  
   หากแบนด์สีแดง (B4) หรือ NIR (B8) ถูกฟื้นฟู ให้คำนวณความคลาดเคลื่อนของค่าดัชนีพืชพรรณ:
   $$\mathcal{L}_{\text{VI}} = \|\text{NDVI}(\hat{X}_{B8}, \hat{X}_{B4}) - \text{NDVI}(X_{B8}, X_{B4})\|_1$$

---

## 3. ตารางขั้นตอนการพัฒนาและ Quality Gates

เพื่อให้การพัฒนามีระบบตรวจสอบที่โปร่งใส ตรวจสอบย้อนกลับได้ และป้องกันข้อผิดพลาดล่วงหน้า เรากำหนด **5 Quality Gates (M0 ถึง M4)** ดังนี้:

```
[Phase 1: Code Modules]
  ├── channel_masking.py  ──► [Gate M0: Shape & Permutation Check] ──► PASS
  └── vit_cross_band.py   ──► [Gate M1: Gradient Flow & Math Check] ──► PASS
                                              │
[Phase 2: Sanity Check]                       ▼
  └── Single-Batch Overfit ─► [Gate M2: Loss Convergence < 0.001]   ──► PASS
                                              │
[Phase 3: Scientific Proof]                   ▼
  └── S1 vs No-S1 Ablation ─► [Gate M3: Delta Gain of SAR >= 15%]   ──► PASS
                                              │
[Phase 4: Full Benchmark]                     ▼
  └── 127 Patterns Sweep   ─► [Gate M4: Zero-Crash & Complete Log]  ──► READY
```

---

### Gate M0: การตรวจสอบโครงสร้างและมิติข้อมูล (Shape & Masking Integrity)

**เป้าหมาย:** ยืนยันว่าโมดูล `channel_masking.py` และ `vit_cross_band.py` มีมิติข้อมูลถูกต้องและทำงานสอดประสานกัน

| รหัสตรวจ | รายการตรวจสอบ (Checklist Item) | วิธีการทดสอบ / โค้ดทดสอบ | เกณฑ์การผ่าน (Pass Criteria) |
|---|---|---|---|
| **CHK-M0.1** | มิติเอาต์พุตของการสุ่มปิดแบนด์ | ป้อน Tensor ขนาด `(4, 9, 256, 256)` เข้า `ChannelMasker` | เอาต์พุตคืนค่า `mask_opt` ขนาด `(4, 7)`, `mask_sar` ขนาด `(4, 2)` และ $k \in [1, 7]$ เสมอ |
| **CHK-M0.2** | ความครบถ้วนของการครอบคลุม 127 รูปแบบ | รันโหมด deterministic ใน `channel_masking.py` | ได้ชุด Combination ครบ 127 แบบพอดี ไม่ซ้ำ และไม่มีเวกเตอร์ที่เป็นศูนย์ทั้งหมด `[0,0,0,0,0,0,0]` |
| **CHK-M0.3** | อัตราการดรอปเรดาร์ ($p_{\text{sar}}$) | สุ่มตัวอย่าง 10,000 ครั้ง ด้วย $p_{\text{sar}} = 0.20$ | สัดส่วนที่ SAR ถูกดรอปต้องอยู่ในช่วง $19.0\% - 21.0\%$ ($p$-value $> 0.01$) |
| **CHK-M0.4** | การประกอบ Channel Embedding | ตรวจสอบมิติใน `vit_cross_band.py` | Tensor ของโทเค็นต้องมีขนาด `(B, C_vis * N_p, D)` โดย $D=512$ และบวก Channel Embedding ตรงแบนด์ |
| **CHK-M0.5** | มิติเอาต์พุตของ Reconstruction Head | ป้อน latent tokens เข้า Reconstruction Head | คืนค่าภาพขนาด `(B, C_mask, 256, 256)` มิติตรงกับจำนวนแบนด์ที่ถูกปิดในตัวอย่างนั้นๆ |

---

### Gate M1: การตรวจสอบการไหลของเกรเดียนต์และคณิตศาสตร์ (Gradient Flow & Math Guard)

**เป้าหมาย:** ป้องกันข้อผิดพลาดเงียบ (Silent Bugs) เช่น ข้อมูลรั่วไหล (Data Leakage) หรือเกรเดียนต์ไม่ไหลไปยังพารามิเตอร์ของแบนด์

| รหัสตรวจ | รายการตรวจสอบ (Checklist Item) | วิธีการทดสอบ / โค้ดทดสอบ | เกณฑ์การผ่าน (Pass Criteria) |
|---|---|---|---|
| **CHK-M1.1** | **กฎการไม่รั่วไหล (Zero Leakage Check)** | ป้อนภาพโดยให้แบนด์ $B8$ ถูก Mask จากนั้นคำนวณ $\frac{\partial \mathcal{L}}{\partial X_{B8_{\text{input}}}}$ | เกรเดียนต์ต่ออินพุตของแบนด์ที่ถูกปิดต้องเป็น **ศูนย์แน่นอน (Strictly 0.0)** ป้องกันการแอบมองข้อมูล |
| **CHK-M1.2** | การไหลของเกรเดียนต์ไปยัง Channel Embedding | ทำ Backward pass ของ Loss รวม | พารามิเตอร์ `channel_embed` ของทุกแบนด์ที่ปรากฏต้องมี `.grad` ที่ไม่ใช่ `None` และไม่ใช่ `0` |
| **CHK-M1.3** | เสถียรภาพตัวเลขใน BF16 / FP32 | รัน 100 Forward-Backward passes ด้วยค่าสุ่มปลายแถว | ไม่มีค่า `NaN`, `Inf` หรือ Gradient Explosion ใน Loss หรือ Attention Weights |
| **CHK-M1.4** | การแยกอิสระของ Batch Elements | ป้อน Batch ขนาด 2 ที่ Sample 0 ปิดแบนด์ 1 แบนด์ และ Sample 1 ปิด 6 แบนด์ | Loss ของ Sample 0 ต้องไม่เปลี่ยนแปลงเมื่อแก้ไขค่าพิกเซลของ Sample 1 |

---

### Gate M2: การทดสอบ Overfitting บนชุดข้อมูลเล็ก (Single-Batch Sanity Gate)

**เป้าหมาย:** พิสูจน์ว่าโครงข่าย Transformer มีความจุเชิงทฤษฎี (Theoretical Capacity) เพียงพอในการจดจำและสร้างความสัมพันธ์ระหว่างแบนด์

| รหัสตรวจ | รายการตรวจสอบ (Checklist Item) | วิธีการทดสอบ / เงื่อนไข | เกณฑ์การผ่าน (Pass Criteria) |
|---|---|---|---|
| **CHK-M2.1** | Overfit กรณีปิด 1 แบนด์ ($k=1$, เช่น B8) | เทรนบน Batch เล็ก (8 ไทล์) เป็นเวลา 300 epochs ด้วย AdamW ($\text{lr}=3\times 10^{-4}$) | $\mathcal{L}_{\text{MSE}} < 0.0010$, PSNR $> 38.0\text{ dB}$, $\text{SSIM} > 0.970$ |
| **CHK-M2.2** | Overfit กรณีปิด 4 แบนด์ ($k=4$, เช่น B4, B8A, B8, B11) | เทรนบน Batch เดิม 500 epochs | $\mathcal{L}_{\text{MSE}} < 0.0025$, PSNR $> 33.0\text{ dB}$, $\text{SSIM} > 0.930$ |
| **CHK-M2.3** | Overfit กรณีวิกฤต ปิดแสงหมด 7 แบนด์ ($k=7$) | เทรนโดยใช้เพียง S1 VV, VH ฟื้นฟูทั้ง 7 แบนด์แสง | โมเดลต้องลด Loss ได้ต่อเนื่อง ($\mathcal{L}_{\text{epoch 500}} < 0.10 \times \mathcal{L}_{\text{epoch 0}}$) |

---

### Gate M3: การพิสูจน์อิทธิพลของ SAR (Cross-Modal Ablation Gate)

**เป้าหมาย:** พิสูจน์ในทางสถิติว่าดาวเทียมเรดาร์ Sentinel-1 (VV, VH) ส่งมอบคุณค่าเชิงข้อมูลจริง ไม่ใช่แค่สัญญาณรบกวน

| รหัสตรวจ | รายการตรวจสอบ (Checklist Item) | วิธีการทดสอบ / เงื่อนไข | เกณฑ์การผ่าน (Pass Criteria) |
|---|---|---|---|
| **CHK-M3.1** | การเปรียบเทียบ M1 (No-SAR) vs M2 (With-SAR) ในระดับ $k \ge 5$ | ประเมินผลบน Validation Set ขนาด 200 ไทล์ (Spatial Split) | M2 (มี SAR) ต้องมี MSE ต่ำกว่า M1 อย่างมีนัยสำคัญทางสถิติ ($p < 0.01$, Paired t-test) |
| **CHK-M3.2** | อิทธิพลต่อขอบแปลงปาล์ม (Edge & Boundary Fidelity) | วัดขอบด้วย Sobel Filter บนแบนด์ NIR ที่ฟื้นฟู | M2 มี Edge Reconstruction Error ต่ำกว่า M1 อย่างน้อย $15\%$ (เรดาร์ช่วยคุมขอบ) |
| **CHK-M3.3** | การไม่พังทลายเมื่อไม่มี SAR (Graceful Degradation) | ทดสอบนำ M2 ไปรันโดยปิด SAR (จำลองสถานการณ์ SAR ขาดหาย) | ผลลัพธ์ของ M2 ต้องไม่แย่กว่า M1 เกิน $5\%$ (พิสูจน์ว่า SAR Dropout ทำงานได้ผล) |

---

### Gate M4: การประเมินผลอัตโนมัติครบ 127 รูปแบบ (127-Pattern Benchmark Gate)

**เป้าหมาย:** ระบบทดสอบอัตโนมัติสามารถรันครบทุกรูปแบบ $2^7-1=127$ แบบโดยไม่มีการขัดข้อง และสร้างรายงานเชิงวิเคราะห์ได้ทันที

| รหัสตรวจ | รายการตรวจสอบ (Checklist Item) | รายละเอียดที่ต้องตรวจสอบ | เกณฑ์การผ่าน (Pass Criteria) |
|---|---|---|---|
| **CHK-M4.1** | เสถียรภาพของสคริปต์วนลูป 127 รูปแบบ | รันสคริปต์ `run_127_patterns_benchmark.py` บน Test Set | รันจนจบครบ 127 รูปแบบโดยไม่มี Out-Of-Memory (OOM) หรือ Crash |
| **CHK-M4.2** | การบันทึกสถิติแบบครบวงจร | ตรวจสอบไฟล์ผลลัพธ์ `results/metrics/benchmark_127_patterns.parquet` | บันทึกครบทุกมิติ: `pattern_id (1-127)`, `k (1-7)`, `masked_bands`, `MAE`, `RMSE`, `PSNR`, `SSIM`, `SAM`, `NDVI_MAE` |
| **CHK-M4.3** | แผนภูมิสรุปและบทวิเคราะห์ | ตรวจสอบภาพกราฟที่สร้างขึ้นใน `results/figures/` | มีกราฟแท่งแยกตามกลุ่ม $k=1..7$, Radar chart แสดงความแม่นยำรายแบนด์ และ Correlation Matrix |

---

## 4. ระเบียบวิธีทดสอบโมเดลหลักหลังสิ้นสุดการพัฒนา (Post-Development Testing Protocol)

เมื่อผ่าน Quality Gates M0–M4 แล้ว โมเดลจะเข้าสู่กระบวนการทดสอบมาตรฐานระดับงานวิจัย (Formal Benchmark Protocol) ประกอบด้วย 5 ชุดการทดสอบ:

```
                  [ชุดการทดสอบโมเดลหลัก 5 มิติ]
                               │
   ┌───────────────┬───────────┼───────────┬───────────────┐
   ▼               ▼           ▼           ▼               ▼
[มิติที่ 1]      [มิติที่ 2]   [มิติที่ 3]   [มิติที่ 4]     [มิติที่ 5]
Spectral        Vegetation  Downstream   Explainable AI  System &
Fidelity        Indices     Palm Seg.    (SHAP Weights)  Latency
(MAE/PSNR/SAM)  (NDVI/NDRE) (mIoU/F1)    Attribution     (ms/VRAM)
```

### 4.1 มิติที่ 1: การประเมินความแม่นยำของสเปกตรัม (Radiometric & Spectral Fidelity)
- **ตัวชี้วัดที่ใช้วัด:**
  - **Mean Absolute Error (MAE) & Root Mean Square Error (RMSE):** วัดค่าระดับพิกเซลต่อแบนด์ (ระดับความสะท้อน $0.0 - 1.0$)
  - **Peak Signal-to-Noise Ratio (PSNR):** ค่าเป้าหมายเฉลี่ย $> 32\text{ dB}$ ในกรณี $k \le 3$
  - **Structural Similarity Index (SSIM):** ค่าเป้าหมายเฉลี่ย $> 0.90$
  - **Spectral Angle Mapper (SAM):** วัดการเบี่ยงเบนของรูปทรงสเปกตรัม (หน่วยเป็นองศาหรือเรเดียน) ค่าเป้าหมาย $< 4.0^\circ$

### 4.2 มิติที่ 2: ความคงอยู่ของดัชนีพืชพรรณทางการเกษตร (Vegetation Index Consistency)
เนื่องจากแบบจำลองนำไปใช้ในการเกษตรและปาล์มน้ำมัน ค่าสะท้อนแสงสัมพัทธ์ระหว่างแบนด์จึงมีความสำคัญยิ่งกว่าค่าเดี่ยวๆ:
- **NDVI Consistency:** $\text{NDVI} = \frac{B8 - B4}{B8 + B4}$ เปรียบเทียบ $\text{MAE}_{\text{NDVI}} = |\text{NDVI}_{\text{pred}} - \text{NDVI}_{\text{true}}|$
- **NDRE Consistency (ดัชนีคลอโรฟิลล์ปาล์ม):** $\text{NDRE} = \frac{B8 - B8A}{B8 + B8A}$
- **NDWI Consistency (ความชื้นทรงพุ่ม):** $\text{NDWI} = \frac{B8 - B11}{B8 + B11}$
- **เกณฑ์ประเมิน:** ค่าความผิดพลาด $\text{MAE}_{\text{NDVI}} < 0.04$ บนแปลงปาล์มน้ำมัน

### 4.3 มิติที่ 3: ประสิทธิภาพบนงานปลายทาง (Downstream Palm Segmentation Transferability)
ทดสอบนำภาพที่ผ่านการฟื้นฟูด้วยโมเดล ไปป้อนเข้าสู่โมเดลจำแนกแปลงปาล์มน้ำมัน (`SegDecoder`):
- **ชุดข้อมูล:** แปลงปาล์มน้ำมันจริง Ground Truth 4,726 แปลง ในชลบุรีและระยอง
- **การทดลอง:**
  1. ทดสอบ Baseline: ภาพจริงฟ้าใส (Clean GT) $\to$ วัดค่า Baseline mIoU และ F1-score
  2. ภาพที่จำลองแบนด์หาย $k$ แบนด์โดยไม่ซ่อม $\to$ วัดความตกต่ำของ mIoU
  3. ภาพที่ผ่านการฟื้นฟูด้วย Cross-Band ViT $\to$ วัดการฟื้นตัวของ mIoU (Target: กู้คืนประสิทธิภาพได้ไม่น้อยกว่า $85\%$ ของ Baseline)

### 4.4 มิติที่ 4: การอธิบายผลลัพธ์เชิงเหตุผล (Explainable AI: Attention & SHAP Analysis)
- **Cross-Attention Rollout:** ดึงน้ำหนัก Attention จาก Transformer Block ชั้นสุดท้าย เพื่อดูว่าเมื่อแบนด์ $B8$ ขาดหาย โมเดลให้น้ำหนักความสนใจไปยังแบนด์ $B8A$ (Optical พี่น้อง) หรือ $VV/VH$ (SAR) มากกว่ากันในแต่ละประเภทพื้นดิน
- **Modality Attribution:** ยืนยันว่าการมีอยู่ของ SAR ช่วยลดความไม่แน่นอน (Uncertainty Entropy) ในการคาดการณ์บริเวณรอยต่อแปลงพืช

### 4.5 มิติที่ 5: การวัดทรัพยากรและเวลาประมวลผล (System Latency & Throughput)
- ทดสอบบนการ์ดจอ RTX 5060 Ti 16 GB ในเครื่อง:
  - Throughput: $\ge 40$ ไทล์/วินาที (Batch size 16, BF16)
  - VRAM Consumption: $\le 8.5\text{ GB}$ (เปิดพื้นที่เหลือให้ระบบอื่นทำงานได้ราบรื่น)

---

## 5. การทบทวนวรรณกรรม: งานวิจัยที่รองรับกระบวนการพัฒนา (Supporting Literature)

การออกแบบโมดูล `channel_masking.py` และ `vit_cross_band.py` อิงตามหลักวิชาการจากงานวิจัยชั้นนำระดับสากล ได้แก่:

### 5.1 SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery (NeurIPS 2022)
- **คณะผู้วิจัย:** Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, Stefano Ermon (Stanford University)
- **การค้นพบสำคัญ:** SatMAE เป็นงานแรกที่ชี้ให้เห็นว่าสถาปัตยกรรม ViT แบบดั้งเดิมที่ปฏิบัติกับแบนด์ดาวเทียมเหมือนภาพ RGB ทั่วไปล้มเหลวในการดึงศักยภาพทางสเปกตรัม ผู้วิจัยเสนอแนวคิด **Spectral Grouping** และ **Spectral Positional Encoding** โดยกำหนดเวกเตอร์ระบุอัตลักษณ์ความยาวคลื่นของแต่ละแบนด์ลงในโทเค็น ทำให้โมเดลสามารถเรียนรู้ความสัมพันธ์ของแบนด์ที่มีความละเอียดเชิงพื้นที่ต่างกันได้
- **การนำมาปรับใช้ในโครงการเรา:** เรานำหลักการ Channel Embedding ของ SatMAE มาใช้ใน `vit_cross_band.py` เพื่อให้ Transformer ทราบว่าโทเค็นที่กำลังประมวลผลมาจากความยาวคลื่นใดอย่างชัดเจน

### 5.2 Presto: Lightweight, Pre-trained Transformers for Remote Sensing Timeseries (NeurIPS 2023)
- **คณะผู้วิจัย:** Gabriel Tseng, Ruben Cartuyvels, Ivan Zvonkov, Mirali Purohit, David Rolnick, Hannah Kerner (McGill University & Mila)
- **การค้นพบสำคัญ:** Presto ออกแบบสถาปัตยกรรมที่แปลงแต่ละแบนด์สเปกตรัมและประเภทเซนเซอร์ (Optical, SAR, Weather) เป็นโทเค็นแยกกัน พร้อมสุ่ม Masking ข้ามทั้งมิติเวลา (Time) และมิติแถบคลื่น (Bands) งานวิจัยพิสูจน์ว่า **Channel-level Tokenization** ทำให้โมเดลสามารถทำงานในลักษณะ Zero-shot inference ต่อชุดข้อมูลที่มีแบนด์ไม่สมบูรณ์ (Arbitrary missing modalities) ได้อย่างทรงพลัง
- **การนำมาปรับใช้ในโครงการเรา:** เป็นรากฐานตรงของโมดูล `channel_masking.py` ในการสุ่มปิดแบนด์ $1–7$ แบนด์ เพื่อบีบให้โมเดลสร้างฟังก์ชันการเชื่อมโยงข้ามแบนด์

### 5.3 DOFA: Dynamic Omnimodal Foundation Model for Earth Observation (CVPR 2024)
- **คณะผู้วิจัย:** Xuanhua He, Chenyang Liu, Keyan Chen, Haotian Zhang, Lixiang Ru, Zhengxi Zou, Zhenwei Shi (Beihang University)
- **การค้นพบสำคัญ:** นำเสนอ Wavelength-conditioned Continuous Embedding ที่รับมือกับดาวเทียมข้ามแพลตฟอร์มได้อย่างไร้รอยต่อ โดยพิสูจน์ว่าการจัดโครงสร้างโทเค็นแบบ Cross-modality ช่วยให้แบบจำลองสามารถดึงคุณลักษณะเชิงโครงสร้างจากเรดาร์ไปเติมเต็มช่องว่างของสเปกตรัมแสงได้อย่างแม่นยำ
- **การนำมาปรับใช้ในโครงการเรา:** ยืนยันสมมติฐานว่าการนำ SAR (VV, VH) มารวมในกระบวนการ Cross-attention ร่วมกับ Channel Embedding จะช่วยยึดโยงรูปทรงเรขาคณิตของพืชได้

### 5.4 MultiMAE: Multi-modal Multi-task Masked Autoencoders (ECCV 2022)
- **คณะผู้วิจัย:** Roman Bachmann, David Mizrahi, Andrei Atanov, Amir Zamir (EPFL)
- **การค้นพบสำคัญ:** ชี้ให้เห็นว่าเมื่อทำการสุ่ม Mask ข้ามโหมดข้อมูล (Modality Masking) เครือข่าย Transformer จะถูกบังคับให้ส่งผ่านข้อมูลข้ามโดเมนอย่างมีประสิทธิภาพ และกลไก **Modality Dropout** เป็นกุญแจสำคัญที่ป้องกันไม่ให้เครือข่ายพังเมื่อโหมดใดโหมดหนึ่งหายไป
- **การนำมาปรับใช้ในโครงการเรา:** การสุ่มดรอป S1 ($p_{\text{sar}} = 0.20$) ใน `channel_masking.py` ถูกสร้างขึ้นโดยอิงตามทฤษฎี Modality Dropout ของ MultiMAE เพื่อสร้างความทนทานต่อข้อมูล

### 5.5 DSen2-CR & GLF-CR: Deep Residual Fusion & Global-Local Fusion for Cloud Removal (RSE 2020 / TGRS 2022)
- **คณะผู้วิจัย:** Meraner et al. (Remote Sensing of Environment, 2020) และ Xu et al. (IEEE TGRS, 2022)
- **การค้นพบสำคัญ:** พิสูจน์ด้วยหลักฐานเชิงประจักษ์ว่า Sentinel-1 C-band SAR มีความยาวคลื่นประมาณ $5.6\text{ cm}$ ซึ่งทะลุผ่านละอองน้ำในก้อนเมฆได้อย่างสมบูรณ์ และการรวมข้อมูล SAR ร่วมกับแบนด์แสงที่รอดชีวิตสามารถกู้คืนค่าการสะท้อนแสงของแบนด์แสงได้อย่างถูกต้องเมื่อเทียบกับข้อมูลภาคพื้นดิน
- **การนำมาปรับใช้ในโครงการเรา:** ใช้เป็นหลักฐานรองรับทางฟิสิกส์ของการนำ SAR มาเป็นสมอชี้นำ (Guiding Anchor) ในการฟื้นฟูแบนด์แสง

---

## 6. การทบทวนวรรณกรรม: งานวิจัยที่แย้งและวิพากษ์วิจารณ์ข้อจำกัด (Critical Literature & Counter-Arguments)

ในการทำงานวิจัยที่น่าเชื่อถือ จำเป็นต้องตระหนักถึง **ข้อจำกัดพื้นฐานทางฟิสิกส์และจุดอ่อนที่ได้รับการทักท้วงในวงการวิชาการ** เพื่อไม่ให้เกิดการกล่าวอ้างเกินจริง (Overclaiming):

### 6.1 ความไม่สอดคล้องเชิงฟิสิกส์ระหว่างโดเมน (Physical Domain Incongruence & Information Deficit)
- **งานวิจัยที่ชี้ข้อจำกัด:**  
  *Ebel et al. (2021)* "Multisensor Data Fusion for Cloud Removal in Global and All-Weather Conditions" (IEEE TGRS) และ *Ebel et al. (2022)* "CR-TS: Cloud Removal in Sentinel-2 Time Series with SAR Guidance"
- **ข้อโต้แย้งทางวิชาการ:**  
  สัญญาณสะท้อนกลับของเรดาร์ SAR (Backscatter Coefficient: $\sigma^0$) ถูกควบคุมโดย **โครงสร้างเรขาคณิต ความขรุขระของพื้นผิว และค่าคงที่ไดอิเล็กทริก (ความชื้น)** ในทางตรงกันข้าม ค่าสะท้อนแสงในย่าน Optical ของ Sentinel-2 ถูกควบคุมโดย **การดูดกลืนพลังงานของโมเลกุลทางชีวเคมี** (เช่น คลอโรฟิลล์ แคโรทีนอยด์ และน้ำในเซลล์พืช)  
  *ดังนั้น เรดาร์ SAR ไม่มีทางวัดระดับคลอโรฟิลล์หรือรงควัตถุของพืชได้โดยตรง* ในทางคณิตศาสตร์ นี่คือ **Ill-Posed Inverse Problem** ที่สภาวะแสงหลายแบบอาจให้ค่าการกระเจิงเรดาร์ที่เท่ากันทุกประการ
- **มาตรการป้องกันในงานของเรา:**  
  เราไม่เคลมว่าโมเดลสามารถสังเคราะห์แสงขึ้นมาจากเรดาร์เพียงอย่างเดียวแบบ $100\%$ แต่ในแผนการประเมิน 127 รูปแบบ เราแบ่งการวิเคราะห์ออกเป็นระดับ $k=1..4$ (มีแบนด์แสงช่วยชี้นำ) และ $k=7$ (มีเฉพาะ SAR) พร้อมรายงานค่าความไม่แน่นอน (Uncertainty) อย่างตรงไปตรงมา

### 6.2 ความเสี่ยงของการสร้างภาพลวงตาและการบิดเบือนข้อมูลปลายทาง (Hallucination & Semantic Fabrication)
- **งานวิจัยที่ชี้ข้อจำกัด:**  
  *Grohnfeldt et al. (2018)* "Evaluation of SAR-optical data fusion for cloud removal" และ *He et al. (2023)* "Pitfalls of Deep Generative Models in Earth Observation"
- **ข้อโต้แย้งทางวิชาการ:**  
  เมื่อข้อมูลสเปกตรัมแสงหายไปเป็นจำนวนมาก โครงข่ายแบบ Generative มักสร้างค่าสเปกตรัมที่ "ดูสมเหตุสมผลด้วยตาเปล่า" (Plausible Textures) แต่ค่าตัวเลขจริงทางรังสีวิทยา (Radiometry) อาจผิดเพี้ยนไปจากสภาพแปลงพืชจริง เช่น หากแปลงปาล์มถูกตัดโค่นหรือเกิดโรคใบร่วง โมเดลอาจดึงข้อมูลเฉลี่ยของบริบทโดยรอบมาสังเคราะห์เป็นแปลงปาล์มสีเขียวสมบูรณ์ ซึ่งนำไปสู่การประเมินผลผลิตที่ผิดพลาดอย่างร้ายแรง
- **มาตรการป้องกันในงานของเรา:**  
  เรากำหนดชุดทดสอบ **Vegetation Index Preservation Loss ($\mathcal{L}_{\text{VI}}$)** และ **Downstream Palm Segmentation Test** ใน Gate M3/M4 เพื่อพิสูจน์ว่าสเปกตรัมที่ฟื้นฟูรักษาความถูกต้องเชิงชีวภาพ ไม่ใช่แค่ภาพลวงตา

### 6.3 ปัญหาสัญญาณรบกวน Speckle และความบิดเบือนเชิงเรขาคณิตของเรดาร์ (Speckle Noise & Geometric Distortions)
- **งานวิจัยที่ชี้ข้อจำกัด:**  
  *Lee et al.* "Speckle filtering and SAR image analysis" และ *Schmitt et al. (2018)* "SEN1-2: A dataset for deep learning in SAR-optical data fusion"
- **ข้อโต้แย้งทางวิชาการ:**  
  1. ภาพ SAR C-band มีสัญญาณรบกวนแบบทวีคูณ (Multiplicative Coherent Speckle Noise) โดยธรรมชาติ หากแบบจำลองทำการเชื่อมโยงข้อมูลโดยตรง สัญญาณ Speckle อาจรั่วไหล (Leakage) เข้าไปรบกวนค่าความสะท้อนแสง ทำให้ภาพ Optical ที่ฟื้นฟูมีเกรนสาก
  2. การถ่ายภาพของเรดาร์เป็นแบบมองข้าง (Side-looking geometry) ทำให้เกิดความบิดเบือน เช่น การทอดเงาเรดาร์ (Radar Shadow), การเอียงลาดของยอดไม้หรือเนินเขา (Layover and Foreshortening) ซึ่งทำให้พิกเซลของ SAR และ Optical บนพื้นที่ลาดชันไม่ตรงกันอย่างสมบูรณ์
- **มาตรการป้องกันในงานของเรา:**  
  ในการนำเข้า S1 เข้าสู่ Patch Projection จะใช้ตัวกรองปรับเรียบเชิงพื้นที่ระดับแพตช์ ($16\times 16$) ซึ่งทำหน้าที่เฉลี่ยสัญญาณ (Spatial Multi-Looking) โดยธรรมชาติ ช่วยลดผลกระทบของ Speckle ลงในระดับโทเค็น

### 6.4 การเรียนรู้ทางลัดและการพังทลายของโหมด (Shortcut Learning & Modality Collapse)
- **งานวิจัยที่ชี้ข้อจำกัด:**  
  *Geirhos et al. (2020)* "Shortcut learning in deep neural networks" (Nature Machine Intelligence) และ *Chen et al. (2021)*
- **ข้อโต้แย้งทางวิชาการ:**  
  ในธรรมชาติ แถบคลื่นแสงมีความสัมพันธ์เชิงสถิติระหว่างกันสูงมาก (เช่น $B2, B3, B4$ มีค่าสัมประสิทธิ์สหสัมพันธ์ $R^2 > 0.90$) หากโมเดลถูกฝึกโดยให้ทั้งแสงและเรดาร์ โมเดลจะเลือกทางลัด (Shortcut) โดยการทำนายแบนด์แสงที่หายไปด้วยแบนด์แสงที่เหลืออยู่ และละเลยข้อมูลจาก SAR เกือบทั้งหมด ($SAR \text{ Under-utilization}$)
- **มาตรการป้องกันในงานของเรา:**  
  เราออกแบบการสุ่มแบบ **Uniform $k \in \{1..7\}$** ร่วมกับการสุ่มดรอปแสงทั้งหมด ($k=7$) และสลับกับการดรอป SAR ($p_{\text{sar}}=0.20$) เพื่อบังคับให้เครือข่ายต้องสลับโหมดการเรียนรู้และไม่สามารถเลือกทางลัดได้เพียงทางเดียว

---

## 7. แผนที่การสร้างไฟล์และโค้ดตัวอย่างสำหรับการทดสอบ (Implementation & Test File Roadmap)

เพื่อให้สามารถติดตามการทำงานและเริ่มลงมือเขียนโค้ดได้อย่างเป็นรูปธรรม โครงการจะจัดระเบียบไฟล์ดังนี้:

### 7.1 โครงสร้างไฟล์ที่จะพัฒนา (Target Directory Structure)
```
Project_GISxKmutnb/
├── models/  (หรือ lib2_vitpalm/model/)
│   ├── channel_masking.py       # [ใหม่] สุ่มปิดแบนด์ 1-7 แบนด์ + S1 Modality Dropout
│   ├── vit_cross_band.py        # [ใหม่] ViT with Channel Embedding & Recon Head
│   └── __init__.py
├── tests/
│   ├── test_channel_masking.py  # Unit Test สำหรับ Gate M0 (CHK-M0.1 - M0.3)
│   ├── test_vit_cross_band.py   # Unit Test สำหรับ Gate M0 & M1 (CHK-M0.4, M1.1 - M1.2)
│   └── test_sanity_overfit.py   # Sanity Test สำหรับ Gate M2 (Single-batch overfit)
├── scripts/cross_band/
│   ├── run_sanity_overfit.py    # สคริปต์รัน Gate M2
│   ├── run_modality_ablation.py # สคริปต์เปรียบเทียบ M1 vs M2 สำหรับ Gate M3
│   └── run_127_patterns_eval.py # สคริปต์รันประเมินครบ 127 รูปแบบ สำหรับ Gate M4
└── results/
    ├── metrics/                 # เก็บผลลัพธ์ parquet/json ของ 127 รูปแบบ
    └── figures/                 # กราฟ Radar, Bar Chart, และ Heatmap
```

### 7.2 โค้ดต้นแบบโครงสร้าง (Interface Draft)

#### `models/channel_masking.py` (Interface Draft)
```python
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class ChannelMasker(nn.Module):
    """
    โมดูลจัดการ Masking แถบคลื่นสเปกตรัม:
    - สุ่มปิด Optical bands (1 ถึง 7 แบนด์)
    - สุ่มดรอป SAR bands (VV, VH) ด้วยความน่าจะเป็น p_sar
    - รองรับโหมด Deterministic 127 patterns สำหรับการประเมินผล
    """
    def __init__(self, p_sar_dropout: float = 0.20):
        super().__init__()
        self.p_sar_dropout = p_sar_dropout
        self.optical_band_names = ["B2", "B3", "B4", "B8A", "B8", "B11", "B12"]
        self.sar_band_names = ["VV", "VH"]

    def sample_random_mask(self, B: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        สุ่ม k ในช่วง [1, 7] แบบ Uniform สำหรับแต่ละตัวอย่างใน Batch
        Returns:
            mask_opt: (B, 7) Boolean Tensor (True = Masked)
            mask_sar: (B, 2) Boolean Tensor
        """
        mask_opt = torch.zeros(B, 7, dtype=torch.bool, device=device)
        for b in range(B):
            k = int(torch.randint(low=1, high=8, size=(1,)).item())
            perm = torch.randperm(7, device=device)[:k]
            mask_opt[b, perm] = True
            
        # SAR modality dropout
        sar_drop = (torch.rand(B, 1, device=device) < self.p_sar_dropout)
        mask_sar = sar_drop.expand(B, 2)
        return mask_opt, mask_sar

    def get_deterministic_pattern(self, pattern_idx: int, device: torch.device) -> torch.Tensor:
        """คืนค่า Mask ตามรหัส pattern_idx (1 ถึง 127)"""
        assert 1 <= pattern_idx <= 127, "Pattern ID ต้องอยู่ในช่วง 1 ถึง 127"
        bits = [(pattern_idx >> i) & 1 for i in range(7)]
        return torch.tensor(bits, dtype=torch.bool, device=device)
```

#### `models/vit_cross_band.py` (Interface Draft)
```python
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class ViTCrossBand(nn.Module):
    """
    Vision Transformer สำหรับการกู้คืนแถบคลื่นดาวเทียม:
    - Band-specific Patch Projection
    - Channel Embedding แยก 9 แบนด์
    - Cross-Band Self-Attention
    - Reconstruction Head สำหรับแบนด์ที่ขาดหาย
    """
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # 1. Patch Projection แยกรายแบนด์ (7 Optical + 2 SAR = 9 แบนด์)
        self.band_projs = nn.ModuleList([
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(9)
        ])
        
        # 2. Learnable Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.channel_embed = nn.Parameter(torch.zeros(9, embed_dim))  # 9 แบนด์
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Reconstruction Head (แมปกลับสู่พิกเซล P x P)
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size * patch_size)
        )
        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        mask_opt: torch.Tensor,
        mask_sar: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        x: (B, 9, H, W)
        mask_opt: (B, 7) True = Masked
        mask_sar: (B, 2) True = Masked
        """
        # กระบวนการประกอบ Token, Attention, และ Reconstruction
        ...
```

---

## 8. สรุปคำมั่นสัญญาและข้อตกลงในการดำเนินงาน (Commitment to Scientific Rigor)

1. **ไม่ใช้ Mock Data / Placeholder Metrics:** ทุกการทดสอบและตัวเลขในรายงานต้องคำนวณสดจาก Tensor และอาเรย์ข้อมูลจริงเท่านั้น
2. **Deterministic & Reproducible:** ทุกขั้นตอนการสุ่มต้องมี Seed ควบคุม และชุดข้อมูลทดสอบถูกล็อก Hash ป้องกัน Data Leakage
3. **การส่งมอบเป็นรูปธรรม:** ผลลัพธ์จากการรันทั้ง 127 รูปแบบจะถูกบันทึกเป็นไฟล์ `.parquet` พร้อมสคริปต์สร้างแผนภูมิที่พร้อมนำไปใช้ในรายงานและบทความวิจัยระดับนานาชาติทันที
