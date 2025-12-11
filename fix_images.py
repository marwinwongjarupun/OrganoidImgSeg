import os
import tifffile
import numpy as np
from pathlib import Path

# --- UPDATE THESE PATHS ---
paths_to_fix = [
    r"C:\Users\Marwin\Desktop\organoids\251204\training-day-5-bulk",
    r"C:\Users\Marwin\Desktop\organoids\251204\training-day-5"
]

def fix_tiff_files(directory):
    print(f"Processing directory: {directory}")
    files = list(Path(directory).glob("*.tif")) + list(Path(directory).glob("*.tiff"))
    
    if not files:
        print("  No TIFF files found.")
        return

    for file_path in files:
        try:
            # Read the image; tifffile.imread often handles the mismatch by ignoring the bad metadata
            img = tifffile.imread(str(file_path))
            
            # Check if it loaded correctly
            if img is None:
                print(f"  Failed to read: {file_path.name}")
                continue

            # Check dimensions: Cellpose prefers (Z, Y, X) or (Y, X)
            # If your image is (Y, X, C) (e.g. 1024, 1440, 3), Cellpose usually handles it,
            # but it is safer to ensure it is saved cleanly.
            print(f"  Fixed {file_path.name} (Shape: {img.shape})")

            # Overwrite the file with a clean standard TIFF
            # We assume the data loaded into 'img' is the correct pixel data
            tifffile.imwrite(str(file_path), img, photometric='rgb' if img.ndim==3 and img.shape[-1]==3 else None)
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

for folder in paths_to_fix:
    fix_tiff_files(folder)

print("\nDone! Try running your Cellpose command again.")