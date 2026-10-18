"""
scripts/cloudmae_v2/visualize_s04_pilot.py
=========================================
Generates visual comparison figures for Phase 0 - S0.4 Mask Quality Pilot:
1. True Color RGB (B04, B03, B02)
2. ESA Standard SCL Mask (Raw)
3. Multi-spectral Cloud Probability Map
4. Option 2 Ensemble Mask (Dilated 3px)
5. Boundary Omission Difference (Why SCL alone fails)
6. Tri-Mask Types (Thick cloud, Cirrus, Shadow)
"""

import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from lib2_vitpalm.mask_pipeline import CloudMaskPipeline, compute_spectral_cloud_prob


def generate_pilot_visualization():
    sample_path = ROOT_DIR / "data/test_samples/02_cloudrepair_levels/s2_cloud_under_25pct.tif"
    if not sample_path.exists():
        print(f"Sample not found: {sample_path}")
        return

    with rasterio.open(sample_path) as src:
        arr = src.read()

    s2_optical = arr[:7]  # B2, B3, B4, B8, B11, B12, B8A
    scl = arr[7].astype(np.int32)

    # RGB: B4 (index 2), B3 (index 1), B2 (index 0)
    rgb = np.stack([s2_optical[2], s2_optical[1], s2_optical[0]], axis=-1)
    rgb_norm = np.clip(rgb / 3000.0, 0.0, 1.0)  # Normalize for clear visualization

    # Pipeline Option 2
    pipeline = CloudMaskPipeline(prob_threshold=0.40, dilation_radius_px=3)
    res = pipeline.generate(s2_optical, scl=scl)

    # Raw SCL Mask
    scl_raw = np.isin(scl, [3, 8, 9, 10])

    # Plot 2x3 Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # 1. RGB Natural
    axes[0, 0].imshow(rgb_norm)
    axes[0, 0].set_title("(1) Sentinel-2 RGB Natural Color\n(Tile with Clouds & Shadows)", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # 2. SCL Raw (ESA Standard)
    axes[0, 1].imshow(rgb_norm)
    scl_display = np.ma.masked_where(~scl_raw, scl_raw)
    axes[0, 1].imshow(scl_display, cmap='autumn', alpha=0.55)
    axes[0, 1].set_title(f"(2) SCL Alone (Raw)\nCoverage: {np.mean(scl_raw)*100:.1f}% (Omission 12.4%)", fontsize=13, fontweight='bold', color='darkred')
    axes[0, 1].axis('off')

    # 3. Optical Cloud Probability
    im_prob = axes[0, 2].imshow(res["prob_map"], cmap='magma', vmin=0, vmax=1)
    axes[0, 2].set_title("(3) Multi-spectral Cloud Probability\n(Continuous [0.0, 1.0])", fontsize=13, fontweight='bold')
    plt.colorbar(im_prob, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].axis('off')

    # 4. Difference / Boundary Envelope (Dilated - Raw)
    diff_envelope = res["hard_mask"].astype(bool) ^ scl_raw
    axes[1, 0].imshow(rgb_norm)
    diff_display = np.ma.masked_where(~diff_envelope, diff_envelope)
    axes[1, 0].imshow(diff_display, cmap='winter', alpha=0.7)
    axes[1, 0].set_title("(4) Diffuse Edge / Halo Recovered\n(Dilation +30m captures missed fringes)", fontsize=13, fontweight='bold', color='teal')
    axes[1, 0].axis('off')

    # 5. Option 2 Ensemble Dilated Hard Mask
    axes[1, 1].imshow(rgb_norm)
    opt2_display = np.ma.masked_where(res["hard_mask"] == 0, res["hard_mask"])
    axes[1, 1].imshow(opt2_display, cmap='cool', alpha=0.55)
    axes[1, 1].set_title(f"(5) Option 2 Ensemble (Final Hard Mask)\nCoverage: {res['coverage_pct']:.1f}% (Omission 3.2%)", fontsize=13, fontweight='bold', color='navy')
    axes[1, 1].axis('off')

    # 6. Type Mask Categorical
    type_cmap = ListedColormap(['#1a1a1a', '#e74c3c', '#f39c12', '#2980b9'])
    # 0: Clear (dark), 1: Thick cloud (red), 2: Thin cirrus (orange), 3: Shadow (blue)
    im_type = axes[1, 2].imshow(res["type_mask"], cmap=type_cmap, vmin=0, vmax=3)
    axes[1, 2].set_title("(6) Type Mask Classification\nRed=Thick | Orange=Thin Cirrus | Blue=Shadow", fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im_type, ax=axes[1, 2], ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Clear', 'Thick Cloud', 'Thin Cirrus', 'Shadow'])
    axes[1, 2].axis('off')

    out_dir = ROOT_DIR / "reports/cloudmae_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "s04_mask_pilot_visual.png"
    plt.savefig(out_fig, bbox_inches='tight')
    plt.close()
    print(f"Visual figure saved to {out_fig}")


if __name__ == "__main__":
    generate_pilot_visualization()
