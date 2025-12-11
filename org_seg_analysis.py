import numpy as np
import pandas as pd
import skimage.io       
from skimage import measure
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import seaborn as sns
import re
import os
import glob
import tifffile
from pathlib import Path

# ================= CONFIGURATION =================
# 1. PATHS FOR THE TWO DATASETS
# Please enter the path to your "Droplet" images
DROPLET_FOLDER = r"C:\Users\Marwin\Desktop\organoids\251204\training-day-5" 

# Please enter the path to your "Bulk" images
BULK_FOLDER = r"C:\Users\Marwin\Desktop\organoids\251204\training-day-5-bulk" 

# 2. OUTPUT LOCATION
# Where should the comparison folder be created? (Defaults to the parent of the droplet folder)
BASE_OUTPUT_DIR = os.path.dirname(DROPLET_FOLDER)

# 3. FILE SETTINGS
IMG_EXTENSION = ".tif"
MASK_SUFFIX = "_seg.npy" 

# 4. SCALE SETTINGS
MICRONS_PER_PIXEL = 0.85  # <--- CHANGE THIS to your actual scale
UNIT_NAME = "µm"          
# =================================================

def fix_tiff_files(directory):
    """
    Scans a directory for TIFF files and rewrites them using tifffile
    to fix potential metadata/header issues before processing.
    """
    print(f"\n[Pre-Check] Scanning and fixing TIFFs in: {directory}")
    
    if not os.path.exists(directory):
        print(f"  Warning: Directory not found: {directory}")
        return

    files = list(Path(directory).glob("*.tif")) + list(Path(directory).glob("*.tiff"))
    
    if not files:
        print("  No TIFF files found to fix.")
        return

    count = 0
    for file_path in files:
        if file_path.name.startswith("._"):
            continue

        try:
            img = tifffile.imread(str(file_path))
            
            if img is None:
                print(f"  Failed to read: {file_path.name}")
                continue

            tifffile.imwrite(
                str(file_path), 
                img, 
                photometric='rgb' if img.ndim==3 and img.shape[-1]==3 else None
            )
            count += 1
            
        except Exception as e:
            print(f"  Error fixing {file_path.name}: {e}")
    
    print(f"  [Pre-Check] Successfully sanitized {count} files.")

def plot_mask_overlay(save_dir, img, masks, filename_prefix, alpha=1.):
    mask_overlay = label2rgb(
        masks, image=img, bg_label=0, bg_color=None, alpha=alpha, kind="overlay"
    )
    plt.figure(figsize=(8, 8))
    plt.imshow(mask_overlay)
    plt.axis("off")
    plt.title(f"Mask overlay: {filename_prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_overlay.png"))
    plt.close()

def calculate_organoid_metrics(masks, filename, scale):
    props = measure.regionprops(masks)
    
    metrics = {
        "source_image": [],
        "label": [],
        "area": [],           
        "eccentricity": [],   
        "circularity": [],    
        "perimeter": [],      
        "major_axis": [],     
        "minor_axis": [],     
        "average_axis": []    
    }

    for prop in props:
        if prop.area < 10: 
            continue

        metrics["source_image"].append(filename)
        metrics["label"].append(prop.label)
        
        real_area = prop.area * (scale ** 2)
        metrics["area"].append(real_area)

        real_perimeter = prop.perimeter_crofton * scale
        real_major = prop.major_axis_length * scale
        real_minor = prop.minor_axis_length * scale
        
        metrics["perimeter"].append(real_perimeter)
        metrics["major_axis"].append(real_major)
        metrics["minor_axis"].append(real_minor)
        metrics["average_axis"].append((real_major + real_minor) / 2)

        metrics["eccentricity"].append(prop.eccentricity)

        circularity = (
            (4 * np.pi * real_area) / (real_perimeter**2)
            if real_perimeter > 0
            else 0
        )
        metrics["circularity"].append(min(circularity, 1.0))

    return pd.DataFrame(metrics)

def plot_metric_statistics(save_dir, metrics_df, title_suffix):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    sns.histplot(metrics_df["area"], bins=30, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title(f"Area ({UNIT_NAME}²)")

    sns.histplot(metrics_df["circularity"], bins=30, kde=True, ax=axes[0, 1], color="orange")
    axes[0, 1].set_title("Circularity (0-1)")

    sns.histplot(metrics_df["average_axis"], bins=30, kde=True, ax=axes[0, 2], color="green")
    axes[0, 2].set_title(f"Average Axis Length ({UNIT_NAME})")

    sns.histplot(metrics_df["major_axis"], bins=30, kde=True, ax=axes[1, 0], color="purple")
    axes[1, 0].set_title(f"Major Axis ({UNIT_NAME})")

    sns.histplot(metrics_df["minor_axis"], bins=30, kde=True, ax=axes[1, 1], color="pink")
    axes[1, 1].set_title(f"Minor Axis ({UNIT_NAME})")

    sns.scatterplot(x="major_axis", y="minor_axis", data=metrics_df, ax=axes[1, 2], alpha=0.6)
    max_val = max(metrics_df["major_axis"].max(), metrics_df["minor_axis"].max())
    axes[1, 2].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    axes[1, 2].set_title("Major vs. Minor Axis")

    plt.suptitle(f"{title_suffix} Statistics", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"statistics_{title_suffix}.png"))
    plt.close(fig)

def plot_metric_heatmap_on_image(save_dir, img, masks, metrics_df, metric, filename_prefix, alpha=0.6):
    norm = Normalize(vmin=metrics_df[metric].min(), vmax=metrics_df[metric].max())
    cmap = viridis
    
    overlay = np.zeros((*masks.shape, 4), dtype=float)
    label_to_metric = dict(zip(metrics_df["label"], metrics_df[metric]))

    for label_val, metric_val in label_to_metric.items():
        color = cmap(norm(metric_val)) 
        mask = masks == label_val
        overlay[mask, :3] = color[:3]
        overlay[mask, 3] = alpha 

    if img.ndim == 2:
        background = np.stack([img] * 3, axis=-1)
    else:
        background = img.copy()
    background = background.astype(float)
    if background.max() > 1.0:
        background /= 255.0

    combined = background.copy()
    alpha_mask = overlay[..., 3]
    for c in range(3):
        combined[..., c] = (overlay[..., c] * alpha_mask) + (
            background[..., c] * (1 - alpha_mask)
        )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(combined)
    ax.axis("off")
    ax.set_title(f"{metric} - {filename_prefix}")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_{metric}_heatmap.png"))
    plt.close(fig)

def process_batch(input_folder, output_folder_name, condition_label):
    print(f"\n--- Starting processing for: {condition_label} ---")
    
    save_dir = os.path.join(input_folder, output_folder_name)
    individual_dir = os.path.join(save_dir, "individual_images")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(individual_dir, exist_ok=True)

    batch_metrics_list = []
    image_files = glob.glob(os.path.join(input_folder, f"*{IMG_EXTENSION}"))
    
    print(f"Directory: {input_folder}")
    print(f"Found {len(image_files)} files.")

    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(IMG_EXTENSION, "")
        
        if base_name.endswith("flows"):
            continue

        mask_path = os.path.join(input_folder, base_name + MASK_SUFFIX)
        
        if not os.path.exists(mask_path):
            print(f"Skipping {base_name}: Mask missing.")
            continue

        print(f"Processing: {base_name}...")
        try:
            img = skimage.io.imread(img_path)
            seg_data = np.load(mask_path, allow_pickle=True).item()
            masks = seg_data["masks"] 
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
            continue

        plot_mask_overlay(individual_dir, img, masks, filename_prefix=base_name)
        current_df = calculate_organoid_metrics(masks, base_name, MICRONS_PER_PIXEL)
        current_df["Condition"] = condition_label
        batch_metrics_list.append(current_df)

        metrics_to_plot = ['area', 'average_axis'] 
        for metric in metrics_to_plot:
            plot_metric_heatmap_on_image(
                individual_dir, img, masks, current_df, metric, 
                filename_prefix=base_name, alpha=0.6
            )

    if batch_metrics_list:
        master_df = pd.concat(batch_metrics_list, ignore_index=True)
        csv_path = os.path.join(save_dir, f"{condition_label}_metrics.csv")
        master_df.to_csv(csv_path, index=False)
        plot_metric_statistics(save_dir, master_df, title_suffix=condition_label)
        print(f"Finished {condition_label}. Saved to {save_dir}")
        return master_df
    else:
        print(f"No valid data found for {condition_label}")
        return pd.DataFrame()

def plot_comparison(droplet_df, bulk_df, output_dir):
    """
    Generate comparison plots between Droplet and Bulk.
    Updated to resolve Seaborn FutureWarning.
    """
    print("\n--- Generating Comparison Graphs ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if droplet_df.empty or bulk_df.empty:
        print("Cannot compare: One or both datasets are empty.")
        return

    combined_df = pd.concat([droplet_df, bulk_df], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, "combined_comparison_data.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Average Axis Comparison
    sns.boxplot(
        data=combined_df, 
        x="Condition", 
        y="average_axis", 
        hue="Condition", # Added hue
        legend=False,    # Disabled legend
        ax=axes[0], 
        palette="Set2"
    )
    axes[0].set_title(f"Average Axis Length ({UNIT_NAME})")
    axes[0].set_ylabel(f"Length ({UNIT_NAME})")

    # 2. Area Comparison
    sns.boxplot(
        data=combined_df, 
        x="Condition", 
        y="area", 
        hue="Condition", # Added hue
        legend=False,    # Disabled legend
        ax=axes[1], 
        palette="Set2"
    )
    axes[1].set_title(f"Organoid Area ({UNIT_NAME}²)")
    axes[1].set_ylabel(f"Area ({UNIT_NAME}²)")
    
    # 3. Circularity Comparison
    sns.boxplot(
        data=combined_df, 
        x="Condition", 
        y="circularity", 
        hue="Condition", # Added hue
        legend=False,    # Disabled legend
        ax=axes[2], 
        palette="Set2"
    )
    axes[2].set_title("Circularity (0-1)")
    axes[2].set_ylabel("Circularity Index")
    axes[2].set_ylim(0, 1.05) 

    plt.suptitle("Droplet vs. Bulk Comparison", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "comparison_droplet_vs_bulk.png")
    plt.savefig(save_path)
    print(f"Comparison graph saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    
    print("====================================================")
    print("STEP 0: Sanitize TIFF files (Fix Bad Metadata)")
    print("====================================================")
    
    fix_tiff_files(DROPLET_FOLDER)
    fix_tiff_files(BULK_FOLDER)
    
    print("\n====================================================")
    print("STEP 1: Analyze Images")
    print("====================================================")

    droplet_df = process_batch(
        input_folder=DROPLET_FOLDER, 
        output_folder_name="droplet_consolidated", 
        condition_label="Droplet"
    )

    bulk_df = process_batch(
        input_folder=BULK_FOLDER, 
        output_folder_name="bulk_consolidated", 
        condition_label="Bulk"
    )

    # 3. Compare
    full_folder_name = os.path.basename(os.path.normpath(DROPLET_FOLDER))
    match = re.search(r"day-\d+", full_folder_name)
    
    if match:
        day_suffix = match.group()
    else:
        day_suffix = full_folder_name

    comparison_folder_name = f"comparison_analysis_{day_suffix}"
    comparison_dir = os.path.join(BASE_OUTPUT_DIR, comparison_folder_name)
    
    plot_comparison(droplet_df, bulk_df, comparison_dir)