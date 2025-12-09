MASK_PATH = r"E:\mw\day 4 train\day4_seg.npy"
IMAGE_PATH = r"E:\mw\day 4 train\day4.tif"

import numpy as np
import pandas as pd
import skimage.io       
from skimage import measure
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, viridis
import seaborn as sns
from matplotlib.ticker import LogLocator
import os


def plot_mask_overlay(save_dir, img, masks, alpha=1.):
    """
    Overlay labeled mask on original image with different colors per ROI.

    Parameters:
    - img: 2D grayscale or 3D RGB image (numpy array)
    - masks: 2D integer array with labeled ROIs (0=background)
    - alpha: float, transparency of mask overlay (0=transparent, 1=opaque)

    Returns:
    - None (displays the plot)
    """
    # If img is grayscale 2D, convert to RGB for color overlay
    # if img.ndim == 2:
    #     img_rgb = np.stack([img] * 3, axis=-1)
    # else:
    #     img_rgb = img.copy()

    # Generate colored label overlay with background transparent
    mask_overlay = label2rgb(
        masks, image=img, bg_label=0, bg_color = None, alpha=alpha, kind="overlay"
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_overlay)
    plt.axis("off")
    plt.title("Mask overlay on original image")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mask_overlay.png"))
    plt.close()


def calculate_organoid_metrics(masks):
    """Calculate organoid metrics from labeled masks"""
    # Calculate region properties
    props = measure.regionprops(masks)

    # Initialize metrics dictionary
    metrics = {
        "label": [],
        "area": [],
        "eccentricity": [],
        "circularity": [],
        "perimeter": [],
    }

    for prop in props:
        if prop.area < 10:  # Skip small artifacts
            continue

        metrics["label"].append(prop.label)
        metrics["area"].append(prop.area)
        metrics["eccentricity"].append(prop.eccentricity)
        metrics["perimeter"].append(prop.perimeter_crofton)

        # Calculate circularity: 1 = perfect circle
        circularity = (
            (4 * np.pi * prop.area) / (prop.perimeter_crofton**2)
            if prop.perimeter_crofton > 0
            else 0
        )
        metrics["circularity"].append(min(circularity, 1.0))  # Cap at 1.0
    return pd.DataFrame(metrics)


def plot_metric_statistics(save_dir, metrics_df):
    """Plot distribution statistics for organoid metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(metrics_df["area"], bins=30, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Area Distribution")

    sns.histplot(metrics_df["circularity"], bins=30, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Circularity Distribution")

    sns.histplot(metrics_df["eccentricity"], bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Eccentricity Distribution")

    sns.scatterplot(x="area", y="circularity", data=metrics_df, ax=axes[1, 1])
    # axes[1, 1].set_xscale("log")
    # axes[1, 1].set_yscale("log")

    axes[1, 1].set_title("Area vs Circularity")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metric_stats.png"))
    plt.close(fig)


def plot_metric_heatmap_on_image(save_dir, img, masks, metrics_df, metric, alpha=0.6):
    """
    Overlay a metric heatmap on the original image as background.
    Only the segmented ROIs are colored by the metric; background is transparent.

    Parameters:
    - img: 2D grayscale or 3D RGB image (numpy array)
    - masks: 2D labeled mask array (integers, 0=background)
    - metrics_df: DataFrame with columns ['label', metric]
    - metric: string, metric column to visualize
    - alpha: float, transparency of the heatmap overlay on ROIs

    Displays the plot with original image as background and heatmap overlay on ROIs.
    """
    # Normalize metric values to [0,1]
    norm = Normalize(vmin=metrics_df[metric].min(), vmax=metrics_df[metric].max())
    cmap = viridis

    # Create empty RGBA overlay image
    overlay = np.zeros((*masks.shape, 4), dtype=float)  # RGBA

    # Map each ROI label to a color based on its metric value
    label_to_metric = dict(zip(metrics_df["label"], metrics_df[metric]))

    for label_val, metric_val in label_to_metric.items():
        color = cmap(norm(metric_val))  # RGBA tuple
        mask = masks == label_val
        overlay[mask, :3] = color[:3]  # RGB
        overlay[mask, 3] = alpha  # Alpha channel only on ROI pixels

    # Prepare background image as RGB float in [0,1]
    if img.ndim == 2:
        background = np.stack([img] * 3, axis=-1)
    else:
        background = img.copy()
    background = background.astype(float)
    if background.max() > 1.0:
        background /= 255.0  # scale if needed

    # Alpha blending: out = alpha*overlay + (1-alpha)*background on ROI pixels
    combined = background.copy()
    alpha_mask = overlay[..., 3]
    for c in range(3):
        combined[..., c] = (overlay[..., c] * alpha_mask) + (
            background[..., c] * (1 - alpha_mask)
        )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(combined)
    ax.axis("off")
    ax.set_title(f"Heatmap overlay of {metric} on original image")

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{metric}_heatmap.png"))
    plt.close(fig)

# data_dir = os.path.dirname(IMAGE_PATH)
# results_dir = os.path.join(data_dir, "segmentation_analysis")
# os.makedirs(results_dir, exist_ok=True)

# img = skimage.io.imread(IMAGE_PATH)
# seg_data = np.load(MASK_PATH, allow_pickle=True).item()
# plot_mask_overlay(results_dir, img, seg_data["outlines"])

# masks = seg_data["masks"]  # Labeled masks

# # Calculate metrics
# metrics_df = calculate_organoid_metrics(masks)

# # Visualize statistics
# plot_metric_statistics(results_dir, metrics_df)

# # Create heatmap overlay (choose metric: 'area', 'circularity', 'eccentricity')
# for metric in ['area', 'circularity', 'eccentricity']:
#     plot_metric_heatmap_on_image(results_dir, img, masks, metrics_df, metric, alpha=0.6)

if __name__ == "__main__":
    data_dir = os.path.dirname(IMAGE_PATH)
    results_dir = os.path.join(data_dir, "segmentation_analysis")
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    img = skimage.io.imread(IMAGE_PATH)
    seg_data = np.load(MASK_PATH, allow_pickle=True).item()
    
    # Extract masks (Ensure you are grabbing the right key, usually 'masks' for analysis)
    masks = seg_data["masks"] 

    # 1. Overlay Plot
    # Changed to plt.show() inside the function or just call it here if you modify the function
    plot_mask_overlay(results_dir, img, masks) 
    print("Overlay saved.")

    # 2. Calculate Metrics
    metrics_df = calculate_organoid_metrics(masks)
    print("Metrics calculated.")

    # 3. Statistics Plot
    plot_metric_statistics(results_dir, metrics_df)
    print("Stats saved.")

    # 4. Heatmaps
    for metric in ['area', 'circularity', 'eccentricity']:
        plot_metric_heatmap_on_image(results_dir, img, masks, metrics_df, metric, alpha=0.6)
        print(f"{metric} heatmap saved.")

    # If you changed the functions to use plt.show(), plots will appear now.
    # Otherwise, check the 'segmentation_analysis' folder for the .png files.
