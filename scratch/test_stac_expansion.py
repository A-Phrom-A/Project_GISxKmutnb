import requests
from datetime import datetime, timedelta

STAC_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'

test_rois = {
    'Prachuap_BangSaphan': {'bbox': [99.40, 11.10, 99.75, 11.40], 'region': 'Southern', 'crop': 'Oil Palm / Coconut'},
    'Prachuap_KuiBuri': {'bbox': [99.70, 12.00, 100.05, 12.30], 'region': 'Southern', 'crop': 'Pineapple / Rubber'},
    'Phetchaburi_ThaYang': {'bbox': [99.80, 12.80, 100.10, 13.10], 'region': 'Western', 'crop': 'Crops / Fruit'},
    'Ratchaburi_PakTho': {'bbox': [99.65, 13.25, 99.95, 13.55], 'region': 'Western', 'crop': 'Sugarcane / Cassava'},
    'Nakhon_Sawan': {'bbox': [100.20, 15.50, 100.55, 15.80], 'region': 'Central', 'crop': 'Sugarcane / Rice'},
    'Phetchabun_Wichian': {'bbox': [101.00, 15.55, 101.35, 15.85], 'region': 'Central/North', 'crop': 'Corn / Sugarcane'},
    'Korat_PakChong': {'bbox': [101.30, 14.55, 101.65, 14.85], 'region': 'Northeastern', 'crop': 'Corn / Cassava'},
    'Korat_ChokChai': {'bbox': [102.00, 14.65, 102.35, 14.95], 'region': 'Northeastern', 'crop': 'Cassava / Sugarcane'},
    'Buriram_NangRong': {'bbox': [102.70, 14.70, 103.05, 15.00], 'region': 'Northeastern', 'crop': 'Cassava / Sugarcane'},
    'Trang_HuaiYot': {'bbox': [99.50, 7.70, 99.85, 8.00], 'region': 'Southern', 'crop': 'Oil Palm / Rubber'},
    'Surat_Kanchanadit': {'bbox': [99.45, 9.10, 99.80, 9.40], 'region': 'Southern', 'crop': 'Oil Palm (Hub)'},
    'Krabi_AoLuek': {'bbox': [98.65, 8.35, 99.00, 8.65], 'region': 'Southern', 'crop': 'Oil Palm / Rubber'},
    'Saraburi_MuakLek': {'bbox': [101.10, 14.60, 101.45, 14.90], 'region': 'Central', 'crop': 'Cassava / Corn'},
    'BuengKan_Seka': {'bbox': [103.85, 17.85, 104.20, 18.15], 'region': 'Northeastern', 'crop': 'Rubber (Hub)'}
}

found = 0
for name, info in test_rois.items():
    bbox = info['bbox']
    r2 = requests.post(STAC_URL, json={
        'collections': ['sentinel-2-l2a'],
        'bbox': bbox,
        'datetime': '2024-01-01/2024-03-31',
        'query': {'eo:cloud_cover': {'lt': 5.0}},
        'limit': 3
    }, timeout=20).json()
    items2 = r2.get('features', [])
    if items2:
        s2 = items2[0]
        s2_t = datetime.fromisoformat(s2['properties']['datetime'].replace('Z', '+00:00'))
        t0 = (s2_t - timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
        t1 = (s2_t + timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
        r1 = requests.post(STAC_URL, json={
            'collections': ['sentinel-1-rtc'],
            'bbox': bbox,
            'datetime': f'{t0}/{t1}',
            'limit': 1
        }, timeout=20).json()
        items1 = r1.get('features', [])
        if items1:
            s1 = items1[0]
            s1_t = datetime.fromisoformat(s1['properties']['datetime'].replace('Z', '+00:00'))
            dt = abs((s2_t - s1_t).total_seconds()) / 86400.0
            cloud = s2['properties']['eo:cloud_cover']
            s2_date = s2['properties']['datetime'][:10]
            print(f"MATCH: {name:22s} | Cloud: {cloud:4.2f}% | dt: {dt:4.2f}d | Date: {s2_date}")
            found += 1
print(f"\nTotal matched in quick sample: {found} / {len(test_rois)}")
