/**
 * =========================================================================================
 * GOOGLE EARTH ENGINE (GEE) DATA EXPORT SUITE FOR GISxKMUTNB TRUE MIRACLE PLATFORM
 * =========================================================================================
 * ลิงก์เปิด Google Earth Engine Code Editor: https://code.earthengine.google.com/
 * 
 * สคริปต์นี้เขียนขึ้นเพื่อให้ผู้ใช้สามารถ Export ชุดข้อมูลดาวเทียมในพื้นที่ใหม่ (New Area)
 * สำหรับนำเข้าสู่ระบบ True Miracle GeoAI Platform (http://localhost:8080/import)
 * ครบทั้ง 3 รูปแบบการทำงาน (Scenarios):
 * 
 *   [รูปแบบที่ 1] Update Satellite GeoTIFF for Existing Zone:
 *                - Sentinel-2 Optical 7 Bands (B2, B3, B4, B8A, B8, B11, B12)
 *                - สำหรับคำนวณ Zonal NDVI ซ้อนทับแปลงเกษตรจริง
 * 
 *   [รูปแบบที่ 2] Cloud Repair with Cloud-MAE:
 *                - คู่ภาพ Sentinel-2 (ติดเมฆหนา) + Sentinel-1 SAR เรดาร์ (VV, VH)
 *                - ช่วงเวลาและขอบเขต ROI เดียวกันเป๊ะ สำหรับฟื้นฟู 7 แบนด์บน GPU
 * 
 *   [รูปแบบที่ 3] Import Brand New Geographic Zone:
 *                - ชุดข้อมูลครบวงจร: S1 SAR (2 Bands) + S2 Optical (7 Bands) + GSat High-Res RGB (3 Bands)
 *                - พร้อมขอบเขต ROI สำหรับสกัดแปลงอัตโนมัติด้วย SAM-1
 * =========================================================================================
 */

// -----------------------------------------------------------------------------------------
// 1. กำหนดขอบเขตพื้นที่ศึกษา (ROI) และช่วงเวลา (Date Range)
// -----------------------------------------------------------------------------------------
// [ตัวเลือกที่ 1 - แนะนำ] ขอบเขต Zone 1: บ้านบึง-ใต้ จ.ชลบุรี (ตรงกับภาพ Baseline ในระบบเป๊ะ 100%)
var roi = ee.Geometry.Rectangle([
  101.3679, // West Longitude
  13.0910,  // South Latitude
  101.4155, // East Longitude
  13.1377   // North Latitude
], null, false);

// [ตัวเลือกที่ 2] หากต้องการทดสอบพื้นที่ใหม่: อ.กาญจนดิษฐ์ จ.สุราษฎร์ธานี (เปิดคอมเมนต์บรรทัดล่าง)
// var roi = ee.Geometry.Rectangle([99.4500, 9.1200, 99.4980, 9.1680], null, false);

// จัดกึ่งกลางแผนที่ไปที่ ROI
Map.centerObject(roi, 13);
Map.addLayer(roi, {color: 'FF0000'}, 'Study Area ROI', false);

// กำหนดช่วงเวลา (Date Range)
// สำหรับภาพท้องฟ้าโปร่งรอบใหม่ (New Cycle): เพื่อเปรียบเทียบเชิงเวลากับภาพเดิม
var dateStartClean  = '2024-01-01'; 
var dateEndClean    = '2024-04-30';

// สำหรับภาพติดเมฆหนา & เรดาร์ (ช่วงฤดูฝน/มรสุม): เพื่อทดสอบโมเดลซ่อมแซมเมฆ Cloud-MAE
var dateStartCloudy = '2023-09-01'; 
var dateEndCloudy   = '2023-10-31';

// -----------------------------------------------------------------------------------------
// 2. ข้อมูลภาพถ่ายดาวเทียม SENTINEL-2 (SURFACE REFLECTANCE L2A)
// -----------------------------------------------------------------------------------------
// แบนด์ 7 แชนแนลมาตรฐานของโปรเจกต์:
// B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR/Red Edge), B8 (Broad NIR), B11 (SWIR-1), B12 (SWIR-2)
var s2Bands = ['B2', 'B3', 'B4', 'B8A', 'B8', 'B11', 'B12'];

// 2.1 ภาพท้องฟ้าโปร่ง (Clean Target / Ground Truth) สำหรับ รูปแบบที่ 1 & รูปแบบที่ 3
var s2CleanCol = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(dateStartClean, dateEndClean)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .sort('CLOUDY_PIXEL_PERCENTAGE');

var s2CleanImage = s2CleanCol.first().select(s2Bands).clip(roi).toInt16();

// 2.2 ภาพติดเมฆจริงบางส่วน (Cloudy Optical Input: 20% - 50%) สำหรับ รูปแบบที่ 2 (Cloud-MAE)
// แนะนำช่วง 20% - 50% เพื่อให้มีก้อนเมฆบดบังแปลงปาล์มบางส่วน และมีพื้นที่เปิดให้เห็นเปรียบเทียบชัดเจน
var s2CloudyCol = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(dateStartCloudy, dateEndCloudy)
  .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', 20))
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 50))
  .sort('CLOUDY_PIXEL_PERCENTAGE', false);

var s2CloudyImage = s2CloudyCol.first().select(s2Bands).clip(roi).toInt16();

// -----------------------------------------------------------------------------------------
// 3. ข้อมูลภาพถ่ายดาวเทียม SENTINEL-1 SAR (C-BAND GRD)
// -----------------------------------------------------------------------------------------
// เรดาร์คลื่นไมโครเวฟ ทะลุเมฆและหมอกควัน 100%
// แบนด์: VV (แนวราบ-ราบ) และ VH (แนวราบ-แนวดิ่ง)
var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate(dateStartCloudy, dateEndCloudy)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .sort('system:time_start', false);

var s1Image = s1Col.first().select(['VV', 'VH']).clip(roi).toFloat();

// -----------------------------------------------------------------------------------------
// 4. ข้อมูลภาพสีจริงความละเอียดสูง (HIGH-RES RGB BASEMAP / GSAT)
// -----------------------------------------------------------------------------------------
// สร้างภาพ RGB True Color (B4, B3, B2) ที่ปรับแต่งสเกลเป็น Byte [0-255]
// เหมาะสำหรับใช้เป็นภาพ RGB พื้นหลัง และใช้รันโมเดล SAM-1 สกัดแปลง
var rgbVisual = s2CleanImage.select(['B4', 'B3', 'B2'])
  .visualize({min: 200, max: 2500, bands: ['B4', 'B3', 'B2']})
  .clip(roi);

// -----------------------------------------------------------------------------------------
// 5. แสดงผลบนแผนที่อินเทอร์แอคทีฟ (Interactive Map Visualization)
// -----------------------------------------------------------------------------------------
Map.addLayer(s2CleanImage, {bands: ['B4', 'B3', 'B2'], min: 200, max: 2500}, 'Sentinel-2 Clean True Color (RGB)');
Map.addLayer(s2CloudyImage, {bands: ['B4', 'B3', 'B2'], min: 200, max: 8000}, 'Sentinel-2 Cloudy Image (Input)');
Map.addLayer(s1Image.select('VV'), {min: -25, max: 0}, 'Sentinel-1 SAR (VV Polarization)');

// คำนวณ NDVI สดบนหน้าจอ
var ndvi = s2CleanImage.normalizedDifference(['B8', 'B4']);
Map.addLayer(ndvi, {min: 0.1, max: 0.85, palette: ['red', 'yellow', 'green', 'darkgreen']}, 'Vegetation Health (NDVI)');

// -----------------------------------------------------------------------------------------
// 6. ส่งออกข้อมูล (EXPORT TASKS TO GOOGLE DRIVE)
// -----------------------------------------------------------------------------------------
// เมื่อกด "Run" ที่สคริปต์นี้ ในแถบ "Tasks" (ด้านขวาบนของหน้าจอ GEE) จะมีปุ่ม Run ให้คลิกเพื่อดาวน์โหลด
// ไฟล์ทั้งหมดจะถูกส่งออกเป็น GeoTIFF ที่มีพิกัด UTM Zone 47N (EPSG:32647) ขนาดตรงกับระบบ 100%

// [รูปแบบที่ 1 & รูปแบบที่ 3] Sentinel-2 Optical 7 Bands (Clean Sky)
Export.image.toDrive({
  image: s2CleanImage,
  description: 'S2_Clean_7Bands_EPSG32647',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 'new_zone_s2_clean_7bands',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// [รูปแบบที่ 2 & รูปแบบที่ 3] Sentinel-1 SAR Dual-Pol (VV, VH)
Export.image.toDrive({
  image: s1Image,
  description: 'S1_SAR_DualPol_EPSG32647',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 'new_zone_s1_sar_dualpol',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// [รูปแบบที่ 2] Sentinel-2 Optical 7 Bands (Cloudy Input สำหรับทดสอบซ่อมแซมเมฆ)
Export.image.toDrive({
  image: s2CloudyImage,
  description: 'S2_Cloudy_7Bands_EPSG32647',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 'new_zone_s2_cloudy_7bands',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// [รูปแบบที่ 3] GSat High-Res RGB (สำหรับ SAM-1 Parcel Segmentation)
Export.image.toDrive({
  image: rgbVisual,
  description: 'GSat_HighRes_RGB_EPSG32647',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 'new_zone_gsat_rgb_highres',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// [ตัวเลือกเสริม] ส่งออกพิกัดกรอบสี่เหลี่ยม ROI เป็นไฟล์ GeoJSON
Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(roi, {name: 'New_Study_Zone_Surat'})]),
  description: 'ROI_Boundary_Vector',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 'new_zone_roi_boundary',
  fileFormat: 'GeoJSON'
});

print('===========================================================');
print('✅ สคริปต์เตรียม Export สำเร็จเรียบร้อย!');
print('กรุณาไปที่แถบ "Tasks" (มุมขวาบนของหน้าจอ GEE) แล้วกดปุ่ม Run เพื่อดาวน์โหลดไฟล์เข้าสู่ Google Drive ของท่าน');
print('===========================================================');
