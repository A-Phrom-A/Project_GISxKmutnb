"""
poc/split_data.py
===================
Splits the generated tiles and masks into Train (70%), Val (15%), Test (15%) 
using a Spatial Block Split strategy.

Instead of randomly shuffling (which causes spatial leakage due to overlap),
this sorts the tiles by their Row and Col indices, effectively slicing the 
map horizontally.
"""

import os
import shutil
import glob
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = PROCESSED_DIR / "split"

def main():
    print("=" * 60)
    print("  GISxKMUTNB - Spatial Data Split")
    print("=" * 60)

    tiles_dir = PROCESSED_DIR / "tiles"
    masks_dir = PROCESSED_DIR / "masks"
    
    if not tiles_dir.exists() or not masks_dir.exists():
        print("Error: data/processed/tiles or masks not found. Run data_prep.py first.")
        return

    # Find all tiles and sort them by filename (which naturally sorts by r then c)
    # Filename format: tile_r00000_c00000.npy
    tile_files = sorted([f.name for f in tiles_dir.glob("tile_r*.npy")])
    total_tiles = len(tile_files)
    
    if total_tiles == 0:
        print("Error: No tiles found in", tiles_dir)
        return
        
    print(f"Total tiles found: {total_tiles}")
    
    # Calculate splits (70/15/15)
    train_end = int(total_tiles * 0.7)
    val_end = int(total_tiles * 0.85)
    
    splits = {
        "train": tile_files[:train_end],
        "val"  : tile_files[train_end:val_end],
        "test" : tile_files[val_end:]
    }
    
    print(f"\nPlanned Split:")
    print(f"  Train : {len(splits['train'])} tiles (Top 70% of the map)")
    print(f"  Val   : {len(splits['val'])} tiles (Next 15%)")
    print(f"  Test  : {len(splits['test'])} tiles (Bottom 15%)")
    
    # Create directories and move files
    for split_name, files in splits.items():
        split_tile_dir = SPLIT_DIR / split_name / "tiles"
        split_mask_dir = SPLIT_DIR / split_name / "masks"
        
        split_tile_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for fname in files:
            # File names
            src_tile = tiles_dir / fname
            # Extract r and c from tile_r00128_c00256.npy
            r_str = fname.split('_')[1]
            c_str = fname.split('_')[2]
            src_mask = masks_dir / f"mask_{r_str}_{c_str}"
            
            dst_tile = split_tile_dir / fname
            dst_mask = split_mask_dir / f"mask_{r_str}_{c_str}"
            
            # Copy (not move, to allow rerunning without wiping original)
            if src_tile.exists() and src_mask.exists():
                shutil.copy2(src_tile, dst_tile)
                shutil.copy2(src_mask, dst_mask)
                copied += 1
            else:
                print(f"  Warning: Missing tile or mask for {fname}")
                
        print(f"  [OK] Copied {copied} pairs to {split_name}/")

    print("\n" + "=" * 60)
    print("  Spatial Split Complete!")
    print(f"  Output location: {SPLIT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
