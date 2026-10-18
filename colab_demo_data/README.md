# 📁 ชุดข้อมูลตัวอย่างสำหรับการเรียนการสอนบน Google Colab (GISxKMUTNB Colab Demo Dataset)

โฟลเดอร์นี้รวบรวม **ชุดข้อมูลขนาดเล็ก (Lightweight ~108 MB)** และ **โมเดล Checkpoints สำเร็จรูป** สำหรับใช้คู่กับ Google Colab Notebook: [`notebooks/GISxKMUTNB_Tutorial_Colab.ipynb`](../notebooks/GISxKMUTNB_Tutorial_Colab.ipynb) เพื่อให้นักศึกษา อาจารย์ และผู้สนใจสามารถทดลองรันระบบ Deep Learning สำหรับดาวเทียมได้ทันทีโดยไม่ต้องโหลดข้อมูลดิบขนาดหลายสิบกิกะไบต์

---

## 🗂️ โครงสร้างภายในโฟลเดอร์ `colab_demo_data/`

```
colab_demo_data/
├── checkpoints/
│   ├── cloud_mae_best.pth            # Pretrained Cloud-MAE ViT (25.6 MB - 500 Epochs บน 5 จังหวัด)
│   └── vitpalm_latefusion_best.pth   # Fine-tuned ViTLateFusion Model (61.5 MB - Ban Bueng Oil Palm)
│
├── test_samples/
│   ├── sample_01_dense_palm_12band.npy     # แปลงปาล์มหนาแน่น (Palm coverage: 27.90%)
│   ├── sample_01_dense_palm_mask.npy       # Ground Truth Mask ปาล์ม
│   ├── sample_01_cloud_mae.npz             # ข้อมูล S1 + S2 สำหรับทดสอบ Cloud Inpainting
│   │
│   ├── sample_02_medium_palm_12band.npy    # แปลงปาล์มขนาดกลาง (Palm coverage: 25.04%)
│   ├── sample_02_medium_palm_mask.npy
│   ├── sample_02_cloud_mae.npz
│   │
│   ├── sample_03_sparse_palm_12band.npy    # สวนปาล์มผสมพืชชนิดอื่น (Palm coverage: 14.75%)
│   ├── sample_03_sparse_palm_mask.npy
│   ├── sample_03_cloud_mae.npz
│   │
│   ├── sample_04_low_palm_12band.npy       # บริเวณขอบแปลง / ถนน / ป่า (Palm coverage: 6.70%)
│   ├── sample_04_low_palm_mask.npy
│   └── sample_04_cloud_mae.npz
│
├── manifest.json                     # Metadata รายละเอียดพิกัดและสถิติของแต่ละ Sample
└── README.md                         # คำแนะนำการใช้งาน (ไฟล์นี้)
```

---

## 🛰️ ลำดับแบนด์ในไฟล์ `*_12band.npy` (Shape: 12 × 256 × 256, float32, [0, 1])

| Channel Index | ดาวเทียม / เซนเซอร์ | Band Code | หน้าที่หลักทาง Remote Sensing |
|:---:|:---:|:---:|:---|
| **0 – 1** | **Sentinel-1 (SAR)** | `VV`, `VH` | เรดาร์ C-band **ทะลุเมฆ 100%**, ตรวจจับความขรุขระและเรือนยอดปาล์ม |
| **2 – 8** | **Sentinel-2 (Optical)** | `B2`, `B3`, `B4`, `B8A`, `B8`, `B11`, `B12` | สเปกตรัมแสงและอินฟราเรด (ใช้คำนวณ NDVI สุขภาพพืช และถูกเมฆบดบังได้) |
| **9 – 11** | **GEO-Sat (High-Res)** | `Red`, `Green`, `Blue` | ภาพถ่ายทางอากาศความละเอียดสูง แสดงรูปร่าง Texture ต้นปาล์ม |

---

## ☁️ วิธีการอัปโหลดขึ้น Google Drive เพื่อใช้งานบน Colab

1. บีบอัดโฟลเดอร์นี้เป็น `colab_demo_data.zip` (หรืออัปโหลดทั้งโฟลเดอร์ `colab_demo_data`) ไปไว้ที่หน้าแรกของ Google Drive:
   ```
   My Drive/
   └── colab_demo_data/
       ├── checkpoints/
       └── test_samples/
   ```
2. ใน Google Colab ให้ Mount Drive ด้วยคำสั่ง:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DATA_DIR = '/content/drive/MyDrive/colab_demo_data'
   ```
3. Notebook จะสามารถโหลดข้อมูลตัวอย่างและโมเดลไปรันการทดลองได้ทันที!
