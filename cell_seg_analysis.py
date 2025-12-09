import numpy as np
import pandas as pd
import skimage.io       
from skimage import measure
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import seaborn as sns
import os
import glob

# ================= CONFIGURATION =================
# Path to the folder containing your images and .npy files
INPUT_FOLDER = r"C:\Users\Marwin\Desktop\organoids\251204\training-day-4" 

# File suffixes (to match image to mask)
IMG_EXTENSION = ".tif"
MASK_SUFFIX = "_seg.npy"  # e.g., if image is "day4.tif", mask must be "day4_seg.npy"
# =================================================

def plot_mask_overlay(save_dir, img, masks, filename_prefix, alpha=1.):
    """Overlay labeled mask on original image."""
    mask_overlay = label2rgb(
        masks, image=img, bg_label=0, bg_color=None, alpha=alpha, kind="overlay"
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_overlay)
    plt.axis("off")
    plt.title(f"Mask overlay: {filename_prefix}")
    plt.tight_layout()
    # Save with unique filename
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_overlay.png"))
    plt.close()

def calculate_organoid_metrics(masks, filename):
    """Calculate organoid metrics from labeled masks."""
    props = measure.regionprops(masks)
    
    metrics = {
        "source_image": [], # Added to track which image the organoid came from
        "label": [],
        "area": [],
        "eccentricity": [],
        "circularity": [],
        "perimeter": [],
    }

    for prop in props:
        if prop.area < 10: 
            continue

        metrics["source_image"].append(filename)
        metrics["label"].append(prop.label)
        metrics["area"].append(prop.area)
        metrics["eccentricity"].append(prop.eccentricity)
        metrics["perimeter"].append(prop.perimeter_crofton)

        circularity = (
            (4 * np.pi * prop.area) / (prop.perimeter_crofton**2)
            if prop.perimeter_crofton > 0
            else 0
        )
        metrics["circularity"].append(min(circularity, 1.0))

    return pd.DataFrame(metrics)

def plot_metric_statistics(save_dir, metrics_df, suffix="global"):
    """Plot distribution statistics for organoid metrics (Consolidated)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Area
    sns.histplot(metrics_df["area"], bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title(f"Area Distribution (N={len(metrics_df)})")

    # Circularity
    sns.histplot(metrics_df["circularity"], bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Circularity Distribution")

    # Eccentricity
    sns.histplot(metrics_df["eccentricity"], bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Eccentricity Distribution")

    # Scatter: Area vs Circularity
    # Optional: Color by source image if there aren't too many images
    if len(metrics_df['source_image'].unique()) <= 10:
        sns.scatterplot(x="area", y="circularity", hue="source_image", data=metrics_df, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    else:
        sns.scatterplot(x="area", y="circularity", data=metrics_df, ax=axes[1, 1], alpha=0.5)
    
    axes[1, 1].set_title("Area vs Circularity")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"statistics_{suffix}.png"))
    plt.close(fig)

def plot_metric_heatmap_on_image(save_dir, img, masks, metrics_df, metric, filename_prefix, alpha=0.6):
    """Overlay a metric heatmap on the original image."""
    norm = Normalize(vmin=metrics_df[metric].min(), vmax=metrics_df[metric].max())
    cmap = viridis
    
    # Filter metrics for just this image (in case df is global, though here we pass local)
    # The logic below assumes 'metrics_df' contains only the labels present in 'masks'
    
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
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_{metric}_heatmap.png"))
    plt.close(fig)

if __name__ == "__main__":
    # Setup directories
    results_dir = os.path.join(INPUT_FOLDER, "consolidated_analysis")
    individual_dir = os.path.join(results_dir, "individual_images")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(individual_dir, exist_ok=True)

    # List to hold dataframes from all images
    all_metrics_list = []

    # Find all .tif files
    image_files = glob.glob(os.path.join(INPUT_FOLDER, f"*{IMG_EXTENSION}"))
    print(f"Found {len(image_files)} images to process.")

    for img_path in image_files:
        base_name = os.path.basename(img_path).replace(IMG_EXTENSION, "")
        
        # Construct expected mask path
        mask_path = os.path.join(INPUT_FOLDER, base_name + MASK_SUFFIX)
        
        if not os.path.exists(mask_path):
            print(f"Skipping {base_name}: Mask not found at {mask_path}")
            continue

        print(f"Processing: {base_name}...")

        # 1. Load Data
        img = skimage.io.imread(img_path)
        seg_data = np.load(mask_path, allow_pickle=True).item()
        
        # Handle 'masks' key or 'outlines' if that's what you labeled, usually 'masks'
        masks = seg_data["masks"] 
        
        # 2. Plot Overlay (Individual)
        plot_mask_overlay(individual_dir, img, masks, filename_prefix=base_name)

        # 3. Calculate Metrics
        # Pass base_name so we know which image these cells belong to
        current_df = calculate_organoid_metrics(masks, base_name)
        
        # Add to global list
        all_metrics_list.append(current_df)

        # 4. Plot Heatmaps (Individual)
        for metric in ['area', 'circularity', 'eccentricity']:
            plot_metric_heatmap_on_image(
                individual_dir, img, masks, current_df, metric, 
                filename_prefix=base_name, alpha=0.6
            )

    # ================= CONSOLIDATION =================
    if all_metrics_list:
        print("Consolidating data and generating global statistics...")
        
        # Combine all individual dataframes into one giant dataframe
        master_df = pd.concat(all_metrics_list, ignore_index=True)
        
        # Save raw data to CSV
        csv_path = os.path.join(results_dir, "all_organoid_metrics.csv")
        master_df.to_csv(csv_path, index=False)
        print(f"Consolidated data saved to: {csv_path}")

        # Plot Global Statistics (All images combined)
        plot_metric_statistics(results_dir, master_df, suffix="CONSOLIDATED")
        
        print(f"Analysis complete. Results saved in {results_dir}")
    else:
        print("No valid image-mask pairs were processed.")