# 🛰️ คู่มือการใช้งาน Google Earth Engine (GEE) และชุดข้อมูลทดสอบพื้นที่ใหม่
**ระบบ True Miracle GeoAI Platform (GISxKMUTNB)**

คู่มือนี้จัดทำขึ้นเพื่อให้ผู้ใช้งานสามารถดึงข้อมูลดาวเทียมจริงจาก **Google Earth Engine (GEE)** ในพื้นที่ใหม่ใดๆ ก็ได้ทั่วประเทศไทย หรือใช้ **ชุดข้อมูลทดสอบสำเร็จรูป (Ready-to-use Test Dataset)** ที่ระบบจัดเตรียมไว้ให้ เพื่อนำเข้าสู่หน้าต่าง [GeoAI Data Import Studio](http://localhost:8080/import)

---

## 🔗 ลิงก์เข้าใช้งาน Google Earth Engine
เข้าสู่แพลตฟอร์ม GEE Code Editor ได้ที่ลิงก์ทางการ:
👉 **[https://code.earthengine.google.com/](https://code.earthengine.google.com/)**

---

## 📂 สคริปต์ GEE สำเร็จรูปในโปรเจกต์
ไฟล์สคริปต์ JavaScript บันทึกไว้ที่:
📁 [`scripts/gee_export_all_scenarios.js`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/scripts/gee_export_all_scenarios.js)

### วิธีการนำไปรันบน GEE:
1. เปิดเบราว์เซอร์ไปที่ **[https://code.earthengine.google.com/](https://code.earthengine.google.com/)** และล็อกอินด้วยบัญชี Google
2. เปิดไฟล์ [`scripts/gee_export_all_scenarios.js`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/scripts/gee_export_all_scenarios.js) แล้ว Copy โค้ดทั้งหมดไป Paste ลงในช่อง Editor ตรงกลาง
3. **ปรับแต่งพื้นที่ (ROI)**:
   - ค่าเริ่มต้นในโค้ดเป็นพื้นที่ **อ.กาญจนดิษฐ์ จ.สุราษฎร์ธานี** (แหล่งปลูกปาล์มน้ำมันหลักภาคใต้ ขนาด 5.3 × 5.3 กม.)
   - หากต้องการเลือกพื้นที่อื่น ให้ใช้เครื่องมือรูปสี่เหลี่ยมหรือโพลีกอน (Draw a shape) บนแผนที่ GEE เพื่อวาดกรอบพื้นที่ที่ต้องการ
4. กดปุ่ม **"Run"** ที่แถบเมนูด้านบน
5. สลับไปที่แถบ **"Tasks"** (มุมขวาบนของหน้าจอ GEE) จะเห็นรายการ Export 5 รายการ:
   - `S2_Clean_7Bands_EPSG32647`
   - `S1_SAR_DualPol_EPSG32647`
   - `S2_Cloudy_7Bands_EPSG32647`
   - `GSat_HighRes_RGB_EPSG32647`
   - `ROI_Boundary_Vector`
6. กดปุ่ม **"Run"** ของแต่ละ Task เพื่อบันทึกไฟล์ GeoTIFF ลงใน Google Drive โฟลเดอร์ `GISxKMUTNB_Uploads`

---

## 📦 ชุดข้อมูลสำเร็จรูปสำหรับทดสอบทันที (Ready-to-Test Datasets)
ระบบได้จัดเตรียมชุดข้อมูลทดสอบทั้ง **ชุดที่ผ่านเกณฑ์ 100% (Valid)** และ **ชุดที่เจตนาไม่ผ่านเงื่อนไข (Negative Edge Cases)** เพื่อทดสอบระบบดักจับข้อผิดพลาด (Strict Prerequisite Inspection) โดยจัดระเบียบแยกเป็น Subfolder ชัดเจนในไดเรกทอรี:
📁 [`data/test_samples/`](file:///c:/Users/kittisak/OneDrive/Documents/Project_GISxKmutnb/data/test_samples)

### รายการ Subfolder แยกตามจุดประสงค์การทดสอบ:

#### 1. สำหรับ Existing Zone (Update Satellite):
- `01_existingzone_valid/` : `s2_new_cycle_clean_7bands.tif` (**✓ ผ่าน 100%** • 7 แบนด์, EPSG:32647)
- `01_existingzone_no_crs/` : `s2_missing_crs.tif` (**✗ ขาดพิกัด CRS** • ทดสอบปฏิเสธไฟล์ที่ไม่มี Georeference)
- `01_existingzone_insufficient_bands/` : `s2_only_1band_grayscale.tif` (**✗ แบนด์ไม่ครบ** • ทดสอบปฏิเสธไฟล์ที่มีแค่ 1 แบนด์)
- `01_existingzone_corrupted_format/` : `corrupted_satellite_image.tif` (**✗ ไฟล์ชำรุด** • ทดสอบปฏิเสธไฟล์ที่ไม่ใช่ GeoTIFF)

#### 2. สำหรับ Cloud Repair (S1 SAR + S2 Optical):
- `02_cloudrepair_real_zone1/` : `s2_optical_cloudy_7bands.tif` + `s1_sar_radar_dualpol.tif` (**✓ ดาวเทียมจริง 100% จาก GEE** • มิติ 522 × 522 px โซน 1 ชลบุรี)
- `02_cloudrepair_valid/` : `s2_optical_cloudy_7bands.tif` + `s1_sar_radar_dualpol.tif` (**✓ ผ่าน 100%** • ชุดภาพทดสอบมาตรฐาน)
- `02_cloudrepair_s1_single_pol/` : `s1_sar_single_band_vv_only.tif` (**✗ S1 แบนด์ไม่ครบ** • ทดสอบเรดาร์มีแบนด์เดียว ขาด VH)
- `02_cloudrepair_s2_insufficient_bands/` : `s2_optical_only_1band.tif` (**✗ S2 แบนด์ไม่ครบ** • ทดสอบภาพ Optical มีแค่ 1 แบนด์)
- `02_cloudrepair_no_crs/` : `s2_cloudy_no_crs.tif` + `s1_sar_no_crs.tif` (**✗ ขาดพิกัด CRS** • ทดสอบคู่ไฟล์ไม่มีระบบพิกัด)

#### 3. สำหรับ Brand New Zone (S1 + S2 + GSat + Vector):
- `03_newzone_real_gee/` : ครบทั้ง S1, S2, GSat, และ `parcels_boundary_polygons.geojson` (**✓ ดาวเทียมจริง 100% จาก GEE** • 522 × 522 px พร้อมเวกเตอร์ ROI)
- `03_newzone_valid/` : ครบทั้ง S1, S2, GSat, และ `parcels_boundary_polygons.geojson` (**✓ ผ่าน 100%** • เวกเตอร์ 25 แปลง Polygon)
- `03_newzone_vector_points_only/` : `gps_waypoints_points_only.geojson` (**✗ เวกเตอร์ไม่ใช่แปลง** • ทดสอบปฏิเสธ Point)
- `03_newzone_vector_empty/` : `empty_parcels_zero_features.geojson` (**✗ เวกเตอร์ว่างเปล่า** • ทดสอบปฏิเสธ 0 แปลง)
- `03_newzone_vector_no_crs/` : `parcels_missing_crs.gpkg` (**✗ เวกเตอร์ไม่มี CRS** • ทดสอบปฏิเสธไฟล์ GPKG ที่ขาดพิกัด)
- `03_newzone_gsat_no_crs/` : `gsat_rgb_missing_crs.tif` (**✗ GSat ไม่มี CRS** • ทดสอบภาพ RGB ขาด Georeference)


---

## 🎯 ตารางจับคู่ไฟล์สำหรับการนำเข้าแต่ละรูปแบบ (Scenarios)

### รูปแบบที่ 1: อัปเดตดาวเทียมรอบใหม่ลงในโซนเดิม (Update Existing Zone)
- **URL**: [http://localhost:8080/import](http://localhost:8080/import) (เลือกตัวเลือก 1)
- **เป้าหมาย**: เลือก Zone 1 – 4 ที่ต้องการอัปเดต
- **ช่องแนบไฟล์**:
  - `ภาพถ่ายดาวเทียมรอบใหม่ (GeoTIFF)` ➔ แนบไฟล์ `surat_thani_s2_7bands_clean.tif`
- **การตรวจสอบ**: ตรวจพบ CRS: EPSG:32647, 7 แบนด์, มิติ $530 \times 530\text{ px}$ ➔ ผ่านเกณฑ์ ✓
- **ผลลัพธ์**: คำนวณ Zonal Statistics และค่า NDVI เฉลี่ยจริงจากพิกเซล พร้อมจัดกลุ่มสุขภาพแปลง

---

### รูปแบบที่ 2: ซ่อมแซมภาพเมฆด้วยเรดาร์ (Sentinel-2 Cloud Removal by Cloud-MAE)
- **URL**: [http://localhost:8080/import](http://localhost:8080/import) (เลือกตัวเลือก 2) หรือ [http://localhost:8080/cloudmae](http://localhost:8080/cloudmae)
- **ช่องแนบไฟล์**:
  - `ภาพ Optical Sentinel-2 ที่ติดเมฆ` ➔ แนบไฟล์ `surat_thani_s2_cloudy.tif`
  - `ภาพเรดาร์ Sentinel-1 SAR คู่กัน` ➔ แนบไฟล์ `surat_thani_s1_sar_dualpol.tif`
- **การตรวจสอบ**: S2 มี 7 แบนด์, S1 มี 2 แบนด์ (Dual-Pol VV, VH), ทั้งสองไฟล์มี CRS ตรงกัน ➔ ผ่านเกณฑ์ ✓
- **ผลลัพธ์**: รันโมเดล Cloud-MAE ViT บน GPU รายงานค่า PSNR (dB), MSE Loss, และภาพคืนสภาพ 7 แบนด์

---

### รูปแบบที่ 3: นำเข้าพื้นที่ศึกษาใหม่ทั้งหมด (Import Brand New Geographic Zone)
- **URL**: [http://localhost:8080/import](http://localhost:8080/import) (เลือกตัวเลือก 3)
- **ชื่อพื้นที่**: ระบุชื่อ เช่น `Zone 5: กาญจนดิษฐ์ สุราษฎร์ธานี (แหล่งปลูกปาล์มภาคใต้)`
- **ช่องแนบไฟล์ดาวเทียม**:
  - `1. ภาพเรดาร์ Sentinel-1 SAR` ➔ แนบไฟล์ `surat_thani_s1_sar_dualpol.tif`
  - `2. ภาพมัลติสเปกตรัม Sentinel-2` ➔ แนบไฟล์ `surat_thani_s2_7bands_clean.tif`
  - `3. ภาพสีจริงความละเอียดสูง (GSat RGB)` ➔ แนบไฟล์ `surat_thani_gsat_rgb_highres.tif`
- **การสกัดแปลง (Parcel Boundary)**:
  - **ตัวเลือก A (Custom Vector)**: แนบไฟล์ `surat_thani_parcels.geojson` (ตรวจพบ 38 แปลง)
  - **ตัวเลือก B (Auto SAM-1 Fallback)**: ติ๊กเลือกใช้โมเดล Meta Segment Anything (SAM-1) เพื่อสกัดแปลงอัตโนมัติจากภาพ GSat
- **การตรวจสอบ**: ตรวจสอบครบทุกไฟล์ ทั้ง CRS และจำนวนแบนด์ ➔ ปลดล็อกปุ่มเริ่มประมวลผล 🔓
- **ผลลัพธ์**: บันทึกโซนใหม่เข้า `zones_manifest.json` พร้อมฉายแปลงบนแผนที่ 3D ทันที
