/**
 * =========================================================================================
 * GOOGLE EARTH ENGINE (GEE) - SENTINEL-2 & SENTINEL-1 CLOUD BENCHMARK EXPORTER
 * =========================================================================================
 * แพลตฟอร์ม: Google Earth Engine Code Editor (https://code.earthengine.google.com/)
 * 
 * วัตถุประสงค์:
 *   ดึงภาพถ่ายดาวเทียม Sentinel-2 (7 แบนด์) + Sentinel-1 SAR (2 แบนด์) คู่กัน
 *   โดยสามารถกำหนดระดับปริมาณเมฆได้ 3 ระดับ เพื่อเปรียบเทียบประสิทธิภาพโมเดล Cloud-MAE:
 *     1. เมฆต่ำมาก (< 10%)  : สำหรับท้องฟ้าโปร่ง (Clear Sky Ground Truth)
 *     2. เมฆบางส่วน (< 25%) : สำหรับทดสอบการซ่อมแซมเมฆระดับสมบูรณ์แบบ (Sweet Spot of MAE)
 *     3. เมฆปานกลาง (< 50%) : สำหรับทดสอบสภาวะท้าทาย (Challenging Benchmark)
 * =========================================================================================
 */

// -----------------------------------------------------------------------------------------
// 1. กำหนดขอบเขตพื้นที่ศึกษา (ROI)
// -----------------------------------------------------------------------------------------
// พื้นที่ Zone 1: อ.บ้านบึง จ.ชลบุรี (ขนาด 5.22 × 5.22 กม. ระบบพิกัด UTM Zone 47N)
var roi = ee.Geometry.Rectangle([
  101.3674, // West Longitude
  13.0905,  // South Latitude
  101.4160, // East Longitude
  13.1382   // North Latitude
], null, false);

Map.centerObject(roi, 13);
Map.addLayer(roi, {color: 'FF0000'}, 'Study Area ROI', false);

// แบนด์ 7 ช่องสัญญาณมาตรฐานโปรเจกต์ GISxKMUTNB
var s2Bands = ['B2', 'B3', 'B4', 'B8A', 'B8', 'B11', 'B12'];

// -----------------------------------------------------------------------------------------
// 2. ฟังก์ชันช่วยค้นหาภาพ Sentinel-2 ตามช่วงปริมาณเมฆ
// -----------------------------------------------------------------------------------------
function getSentinel2ByCloud(minCloud, maxCloud, startDate, endDate, name) {
  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', minCloud))
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', maxCloud))
    .sort('CLOUDY_PIXEL_PERCENTAGE');
  
  var img = ee.Image(col.first());
  return img;
}

// -----------------------------------------------------------------------------------------
// 3. ค้นหาภาพทั้ง 3 ระดับเมฆสำหรับ Zone 1 (ชลบุรี)
// -----------------------------------------------------------------------------------------

// ระดับ 1: เมฆต่ำกว่า 10% (< 10% Cloud Cover) — ฟ้าโปร่งจัด
var s2_under_10 = getSentinel2ByCloud(0, 10, '2023-11-01', '2024-03-31', 'Under_10');

// ระดับ 2: เมฆต่ำกว่า 25% (10% - 25% Cloud Cover) — เมฆปุยเล็กน้อย เหมาะที่สุดสำหรับ Cloud-MAE
var s2_under_25 = getSentinel2ByCloud(10, 25, '2023-08-01', '2023-12-31', 'Under_25');

// ระดับ 3: เมฆต่ำกว่า 50% (25% - 50% Cloud Cover) — เมฆก้อนขนาดกลาง ปกคลุมบางแปลง
var s2_under_50 = getSentinel2ByCloud(25, 50, '2023-06-01', '2023-10-31', 'Under_50');

// -----------------------------------------------------------------------------------------
// 4. ค้นหาภาพเรดาร์ SENTINEL-1 SAR (C-Band Dual-Pol VV, VH) ในวันเวลาที่ตรงกัน
// -----------------------------------------------------------------------------------------
function getPairedS1(s2Image) {
  var date = ee.Date(s2Image.get('system:time_start'));
  var s1Col = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(roi)
    .filterDate(date.advance(-4, 'day'), date.advance(4, 'day'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .sort('system:time_start');
  
  return ee.Image(s1Col.first());
}

var s1_paired_25 = getPairedS1(s2_under_25);
var s1_paired_50 = getPairedS1(s2_under_50);

// -----------------------------------------------------------------------------------------
// 5. แสดงผลบนแผนที่อินเทอร์แอคทีฟ (Interactive Visualizations)
// -----------------------------------------------------------------------------------------
var visRgb = {min: 300, max: 2800, bands: ['B4', 'B3', 'B2']};
var visSar = {min: -25, max: -3, bands: ['VV']};

Map.addLayer(s2_under_10.clip(roi), visRgb, '1. S2 Cloud < 10% (Clear Reference)', false);
Map.addLayer(s2_under_25.clip(roi), visRgb, '2. S2 Cloud < 25% (Sweet Spot - แนะนำ)', true);
Map.addLayer(s2_under_50.clip(roi), visRgb, '3. S2 Cloud < 50% (Medium Cloud)', false);
Map.addLayer(s1_paired_25.select(['VV', 'VH']).clip(roi), visSar, '4. S1 SAR Radar (Paired with <25%)', false);

// พิมพ์ข้อมูลสถิติของภาพออกทางแถบ Console
print('===========================================================');
print('📊 รายละเอียดภาพที่ตรวจพบในระบบ GEE:');
print('1. ภาพเมฆ < 10% วันที่:', s2_under_10.get('system:index'), 'เมฆ (%):', s2_under_10.get('CLOUDY_PIXEL_PERCENTAGE'));
print('2. ภาพเมฆ < 25% วันที่:', s2_under_25.get('system:index'), 'เมฆ (%):', s2_under_25.get('CLOUDY_PIXEL_PERCENTAGE'));
print('3. ภาพเมฆ < 50% วันที่:', s2_under_50.get('system:index'), 'เมฆ (%):', s2_under_50.get('CLOUDY_PIXEL_PERCENTAGE'));
print('===========================================================');

// -----------------------------------------------------------------------------------------
// 6. ส่งออกเป็น GeoTIFF เข้าสู่ Google Drive (โฟลเดอร์ GISxKMUTNB_Uploads)
// -----------------------------------------------------------------------------------------

// ส่งออกระดับที่ 1: เมฆ < 10% (7 แบนด์ 10m UTM Zone 47N)
Export.image.toDrive({
  image: s2_under_10.select(s2Bands).clip(roi).toInt16(),
  description: 'S2_Cloud_Under_10pct_Zone1',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 's2_zone1_cloud_under_10pct',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// ส่งออกระดับที่ 2 (แนะนำที่สุด): เมฆ < 25% (7 แบนด์ 10m)
Export.image.toDrive({
  image: s2_under_25.select(s2Bands).clip(roi).toInt16(),
  description: 'S2_Cloud_Under_25pct_Zone1',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 's2_zone1_cloud_under_25pct',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// ส่งออกเรดาร์ Sentinel-1 SAR คู่กัน (2 แบนด์ VV, VH)
Export.image.toDrive({
  image: s1_paired_25.select(['VV', 'VH']).clip(roi).toFloat(),
  description: 'S1_SAR_Paired_Under_25pct_Zone1',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 's1_zone1_sar_paired_25pct',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

// ส่งออกระดับที่ 3: เมฆ < 50% (7 แบนด์ 10m)
Export.image.toDrive({
  image: s2_under_50.select(s2Bands).clip(roi).toInt16(),
  description: 'S2_Cloud_Under_50pct_Zone1',
  folder: 'GISxKMUTNB_Uploads',
  fileNamePrefix: 's2_zone1_cloud_under_50pct',
  region: roi,
  scale: 10,
  crs: 'EPSG:32647',
  maxPixels: 1e9
});

print('👉 วิธีการดาวน์โหลด: ไปที่แถบ "Tasks" (มุมขวาบนของจอ GEE) แล้วคลิกปุ่ม "Run" ของแต่ละรายการ');
