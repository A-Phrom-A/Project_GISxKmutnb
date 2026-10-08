"""
ตัวอย่างการใช้งาน Library lib3_geoai ที่ออกแบบมาให้กระชับและใช้งานง่ายที่สุด
"""

from lib3_geoai import GeoAI, GeoAIConfig

# =====================================================================
# 🌴 Use Case 1: ตรวจจับสวนปาล์มในพื้นที่ชลบุรี (Palm Pipeline)
# =====================================================================

print("--- 1. Palm Detection ---")
# 1.1 สร้างโมเดลแบบบรรทัดเดียว จบ!
palm = GeoAI.palm(
    data_dir="data/processed/split",
    fusion="late",            # เลือก fusion: early, late, cross
    decoder="seg",            # อนาคตสามารถเปลี่ยนเป็น "unet" หรือ "linear" ได้
)

# 1.2 สั่งเทรนโมเดล
# palm.fit(epochs=50, lr=1e-4)

# 1.3 ประเมินผล
# palm.evaluate()

# 1.4 นำไปทายผลภาพใหม่
# pred = palm.predict("data/processed/new_tile.npy")


# =====================================================================
# 🌍 Use Case 1.1: เพิ่มข้อมูลพื้นที่อื่นเข้ามา เพื่อให้โมเดลเก่งขึ้น
# =====================================================================
print("\n--- 1.1 Add New Region Data ---")
# เพิ่มข้อมูลจากจังหวัดอื่นเข้ามา แล้วสั่ง fine_tune
# palm.add_data("data/new_surat_thani/")
# palm.fine_tune(epochs=20, lr=1e-5)


# =====================================================================
# 🚀 Use Case 2: เทรนโมเดลงานด้านดาวเทียมใหม่ (ไม่ต้องเป็นปาล์ม)
# =====================================================================
print("\n--- 2. Custom Satellite Task ---")
# ใช้คลาส .custom() รับค่า Config อิสระ เช่น สมมติว่ามีแค่ข้อมูล S2 (7 bands)
custom_task = GeoAI.custom(
    data_dir="data/other_task/split",
    in_channels=7,             # มีแค่ S2 (7 bands)
    pos_weight=2.0,            # ปรับ weight ลดลงถ้า class ไม่ balance มาก
    output_mode="binary"       # สั่งให้คืนค่า 0, 1 ตอน predict ออกมาเลย
)
# custom_task.fit(epochs=30)


# =====================================================================
# 🏆 Use Case 3: รัน Benchmark หาสิ่งที่ดีที่สุดอัตโนมัติ
# =====================================================================
print("\n--- 3. Automated Benchmark ---")
# ให้ Library รันเปรียบเทียบทุก Fusion ให้เลย ว่าแบบไหนดีสุด
"""
results = GeoAI.benchmark(
    data_dir="data/processed/split",
    fusions=["early", "late", "cross"],
    epochs=50
)
results.compare()
"""

# =====================================================================
# 🌥️ Use Case 4: Self-Supervised Pretraining (รองรับเมฆ)
# =====================================================================
print("\n--- 4. Cloud MAE Pretraining ---")
# สั่ง MAE Pretrain โดยระบุ Strategy = 'cloud' 
"""
mae = GeoAI.pretrain(
    data_dir="data/processed/tiles", 
    mask_strategy="cloud"      # ให้ mask บริเวณที่มีเมฆบัง
)
mae.fit()

# แล้วเอา weights ไปสร้าง Palm Pipeline
palm_pre = GeoAI.palm(pretrained=mae.checkpoint_path)
"""
