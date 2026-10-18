import json
import requests
from datetime import datetime, timedelta

STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'

# 30 Comprehensive Agricultural ROIs across Thailand
NEW_ROIS = {
    # ภาคใต้ (Southern Palm, Rubber, Fruit)
    'Prachuap_BangSaphan': {'bbox': [99.40, 11.10, 99.75, 11.40], 'region': 'Southern', 'crop': 'Oil Palm / Coconut'},
    'Prachuap_KuiBuri': {'bbox': [99.70, 12.00, 100.05, 12.30], 'region': 'Southern', 'crop': 'Pineapple / Rubber'},
    'Prachuap_ThapSakae': {'bbox': [99.50, 11.40, 99.85, 11.70], 'region': 'Southern', 'crop': 'Oil Palm / Coconut'},
    'Chumphon_LangSuan': {'bbox': [99.00, 9.85, 99.35, 10.15], 'region': 'Southern', 'crop': 'Oil Palm / Durian / Rubber'},
    'Surat_Chaiya': {'bbox': [99.10, 9.30, 99.45, 9.60], 'region': 'Southern', 'crop': 'Oil Palm / Rice'},
    'Surat_WiangSa': {'bbox': [99.25, 8.55, 99.60, 8.85], 'region': 'Southern', 'crop': 'Oil Palm / Rubber'},
    'Krabi_AoLuek': {'bbox': [98.65, 8.35, 99.00, 8.65], 'region': 'Southern', 'crop': 'Oil Palm / Rubber'},
    'Krabi_PlaiPhraya': {'bbox': [98.75, 8.50, 99.10, 8.80], 'region': 'Southern', 'crop': 'Oil Palm / Rubber'},
    'Trang_HuaiYot': {'bbox': [99.50, 7.70, 99.85, 8.00], 'region': 'Southern', 'crop': 'Oil Palm / Rubber'},
    'Songkhla_Sadao': {'bbox': [100.35, 6.55, 100.70, 6.85], 'region': 'Southern', 'crop': 'Rubber (Hub)'},
    'Phatthalung_KhuanKhanun': {'bbox': [99.90, 7.65, 100.25, 7.95], 'region': 'Southern', 'crop': 'Rubber / Rice'},

    # ภาคกลางและตะวันตก (Central / Western Sugarcane, Cassava, Fruit)
    'Phetchaburi_ThaYang': {'bbox': [99.80, 12.80, 100.10, 13.10], 'region': 'Western', 'crop': 'Crops / Fruit'},
    'Ratchaburi_PakTho': {'bbox': [99.65, 13.25, 99.95, 13.55], 'region': 'Western', 'crop': 'Sugarcane / Cassava'},
    'Ratchaburi_ChomBueng': {'bbox': [99.50, 13.50, 99.85, 13.80], 'region': 'Western', 'crop': 'Sugarcane / Cassava'},
    'Kanchanaburi_BoPhloi': {'bbox': [99.45, 14.15, 99.80, 14.45], 'region': 'Western', 'crop': 'Sugarcane / Cassava'},
    'Suphanburi_DanChang': {'bbox': [99.55, 14.75, 99.90, 15.05], 'region': 'Central', 'crop': 'Sugarcane / Cassava'},
    'Nakhon_Sawan_Takhli': {'bbox': [100.25, 15.20, 100.60, 15.50], 'region': 'Central', 'crop': 'Sugarcane / Corn'},
    'Nakhon_Sawan_PhaiSali': {'bbox': [100.60, 15.50, 100.95, 15.80], 'region': 'Central', 'crop': 'Sugarcane / Cassava'},
    'Uthai_Thani_LanSak': {'bbox': [99.45, 15.35, 99.80, 15.65], 'region': 'Central', 'crop': 'Sugarcane / Cassava'},
    'ChaiNat_Sankhaburi': {'bbox': [100.05, 14.95, 100.40, 15.25], 'region': 'Central', 'crop': 'Rice / Sugarcane'},
    'Saraburi_MuakLek': {'bbox': [101.10, 14.60, 101.45, 14.90], 'region': 'Central', 'crop': 'Corn / Cassava'},
    'Lopburi_ChaiBadan': {'bbox': [101.05, 15.10, 101.40, 15.40], 'region': 'Central', 'crop': 'Cassava / Sugarcane'},

    # ภาคอีสาน (Northeastern Cassava, Sugarcane, Rubber)
    'Korat_PakChong': {'bbox': [101.30, 14.55, 101.65, 14.85], 'region': 'Northeastern', 'crop': 'Corn / Cassava'},
    'Korat_ChokChai': {'bbox': [102.00, 14.65, 102.35, 14.95], 'region': 'Northeastern', 'crop': 'Cassava / Sugarcane'},
    'Korat_DanKhunThot': {'bbox': [101.65, 15.10, 102.00, 15.40], 'region': 'Northeastern', 'crop': 'Cassava (Hub)'},
    'Buriram_NangRong': {'bbox': [102.70, 14.70, 103.05, 15.00], 'region': 'Northeastern', 'crop': 'Cassava / Sugarcane'},
    'Chaiyaphum_KhonSan': {'bbox': [101.80, 16.50, 102.15, 16.80], 'region': 'Northeastern', 'crop': 'Sugarcane / Cassava'},
    'KhonKaen_ChumPhae': {'bbox': [102.00, 16.45, 102.35, 16.75], 'region': 'Northeastern', 'crop': 'Sugarcane / Rice'},
    'BuengKan_Seka': {'bbox': [103.85, 17.85, 104.20, 18.15], 'region': 'Northeastern', 'crop': 'Rubber (Hub)'},

    # ภาคเหนือตอนล่าง (Lower Northern Crops)
    'Phetchabun_Wichian': {'bbox': [101.00, 15.55, 101.35, 15.85], 'region': 'Lower Northern', 'crop': 'Corn / Sugarcane'},
    'Phetchabun_BuengSamPhan': {'bbox': [101.00, 15.75, 101.35, 16.05], 'region': 'Lower Northern', 'crop': 'Sugarcane / Cassava'},
    'Phitsanulok_NoenMaprang': {'bbox': [100.50, 16.45, 100.85, 16.75], 'region': 'Lower Northern', 'crop': 'Corn / Fruit'}
}

# Scan Dry Season 2024 and Dry Season 2023
SEASONS = [
    '2024-01-01/2024-03-31',
    '2023-01-01/2023-03-31'
]

discovered_candidates = []

print(f"Scanning STAC for clear-sky pairs across {len(NEW_ROIS)} ROIs...", flush=True)

for roi_name, info in NEW_ROIS.items():
    bbox = info['bbox']
    found_for_roi = 0
    for season in SEASONS:
        if found_for_roi >= 1: # 1 best pair per ROI is plenty to get 20+ pairs
            break
        try:
            r2 = requests.post(STAC_URL, json={
                'collections': ['sentinel-2-l2a'],
                'bbox': bbox,
                'datetime': season,
                'query': {'eo:cloud_cover': {'lt': 5.0}},
                'limit': 3
            }, timeout=20).json()
            
            s2_items = r2.get('features', [])
            if not s2_items:
                continue
                
            best_pair = None
            min_dt = 999.0
            
            for s2_it in s2_items:
                s2_time = datetime.fromisoformat(s2_it['properties']['datetime'].replace('Z', '+00:00'))
                t0 = (s2_time - timedelta(days=3.0)).strftime('%Y-%m-%dT%H:%M:%SZ')
                t1 = (s2_time + timedelta(days=3.0)).strftime('%Y-%m-%dT%H:%M:%SZ')
                
                r1 = requests.post(STAC_URL, json={
                    'collections': ['sentinel-1-rtc'],
                    'bbox': bbox,
                    'datetime': f'{t0}/{t1}',
                    'limit': 2
                }, timeout=20).json()
                
                s1_items = r1.get('features', [])
                for s1_it in s1_items:
                    s1_time = datetime.fromisoformat(s1_it['properties']['datetime'].replace('Z', '+00:00'))
                    dt = abs((s2_time - s1_time).total_seconds()) / 86400.0
                    if dt < min_dt and dt <= 3.0:
                        min_dt = dt
                        best_pair = {
                            'roi_id': roi_name,
                            'region': info['region'],
                            'crop': info['crop'],
                            'bbox': bbox,
                            's2_id': s2_it['id'],
                            's1_id': s1_it['id'],
                            's2_date': s2_it['properties']['datetime'][:10],
                            's1_date': s1_it['properties']['datetime'][:10],
                            'cloud_pct': round(s2_it['properties']['eo:cloud_cover'], 2),
                            'dt_days': round(dt, 2)
                        }
            if best_pair is not None:
                discovered_candidates.append(best_pair)
                found_for_roi += 1
                print(f"FOUND: {roi_name:25s} | Cloud: {best_pair['cloud_pct']:4.2f}% | dt: {best_pair['dt_days']:4.2f}d | Date: {best_pair['s2_date']}", flush=True)
        except Exception as e:
            pass

print(f"\nTotal high-quality candidates discovered: {len(discovered_candidates)}", flush=True)

# Save to catalog
out_path = 'projects/cross_band_recovery/data/manifests/stac_expansion_catalog.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(discovered_candidates, f, indent=2, ensure_ascii=False)
print(f"Catalog saved to {out_path}", flush=True)
