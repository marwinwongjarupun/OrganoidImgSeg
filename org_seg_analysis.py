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
from scipy import stats  # Added for statistical analysis

# ================= CONFIGURATION =================
# 1. PATHS FOR THE TWO DATASETS
TRAINING_SET_1 = r"D:\Organoid images\confocal\260330\training\training-collection"
TRAINING_SET_2 = r"D:\Organoid images\confocal\260330\training\training-waste" 

# 2. OUTPUT LOCATION
BASE_OUTPUT_DIR = os.path.dirname(TRAINING_SET_1)

# 3. FILE SETTINGS
MASK_SUFFIX = "_seg.npy" 

# 4. SCALE SETTINGS
MICRONS_PER_PIXEL = 0.85 
UNIT_NAME = "µm"          
# =================================================

def get_condition_name_from_path(folder_path):
    """Extracts label from folder name."""
    folder_name = os.path.basename(os.path.normpath(folder_path))
    if "-" in folder_name:
        label = folder_name.split("-")[-1]
    else:
        label = folder_name
    return label.capitalize()

def fix_tiff_files(directory):
    """Scans and fixes TIFF metadata."""
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
            if img is None: continue
            
            # Save nicely
            tifffile.imwrite(
                str(file_path), 
                img, 
                photometric='rgb' if img.ndim==3 and img.shape[-1]==3 else None
            )
            count += 1
        except Exception:
            pass
    print(f"  [Pre-Check] Successfully sanitized {count} files.")

def normalize_image_for_plot(img):
    """
    Normalizes raw image data (e.g., 0-4095 12-bit) to 0.0-1.0 float range
    to prevent matplotlib 'Clipping' warnings and slowness.
    """
    img = img.astype(float)
    if img.size == 0:
        return img
    
    val_min = img.min()
    val_max = img.max()
    
    # If the image is not already 0-1 and has data
    if val_max > 1.0:
        if val_max > val_min:
            img = (img - val_min) / (val_max - val_min)
        else:
            # If the image is a solid color (min == max), make it all zeros
            img = np.zeros_like(img)
            
    return img

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
    props = measure.regionprops_table(masks, properties=(
        'label', 'area', 'perimeter_crofton', 
        'major_axis_length', 'minor_axis_length', 'eccentricity'
    ))
    
    df = pd.DataFrame(props)
    
    if df.empty:
        return pd.DataFrame()
        
    df = df[df['area'] >= 10].copy()
    
    if df.empty:
        return pd.DataFrame()

    df["source_image"] = filename
    
    df["area_micron"] = df["area"] * (scale ** 2)
    df["perimeter_micron"] = df["perimeter_crofton"] * scale
    df["major_axis"] = df["major_axis_length"] * scale
    df["minor_axis"] = df["minor_axis_length"] * scale
    df["average_axis"] = (df["major_axis"] + df["minor_axis"]) / 2
    
    df["circularity"] = (4 * np.pi * df["area_micron"]) / (df["perimeter_micron"] ** 2)
    df["circularity"] = df["circularity"].clip(upper=1.0).fillna(0)
    
    df = df.drop(columns=['area', 'perimeter_crofton', 'major_axis_length', 'minor_axis_length'])
    
    df = df.rename(columns={
        "area_micron": "area", 
        "perimeter_micron": "perimeter"
    })
    
    return df[[
        "source_image", "label", "area", "eccentricity", 
        "circularity", "perimeter", "major_axis", "minor_axis", "average_axis"
    ]]

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
    if not metrics_df.empty:
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
    
    max_label = masks.max()
    metric_map_lookup = np.zeros(max_label + 1)
    
    indices = metrics_df["label"].values
    values = metrics_df[metric].values
    valid_mask = indices <= max_label
    metric_map_lookup[indices[valid_mask]] = values[valid_mask]

    mapped_image = metric_map_lookup[masks]
    overlay_rgba = cmap(norm(mapped_image))
    
    background_mask = (masks == 0)
    overlay_rgba[background_mask, 3] = 0.0  
    overlay_rgba[~background_mask, 3] = alpha 

    if img.ndim == 2:
        background = np.stack([img] * 3, axis=-1)
    else:
        background = img.copy()

    combined = background.copy()
    mask_indices = ~background_mask
    
    fg = overlay_rgba[mask_indices, :3]
    bg = background[mask_indices, :3]
    a = overlay_rgba[mask_indices, 3][:, None] 
    
    combined[mask_indices] = (fg * a) + (bg * (1.0 - a))

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

    batch_metrics_list =[]
    
    image_files = glob.glob(os.path.join(input_folder, "*.tif")) + \
                  glob.glob(os.path.join(input_folder, "*.tiff"))
    
    print(f"Directory: {input_folder}")
    print(f"Found {len(image_files)} files.")

    for img_path in image_files:
        _, ext = os.path.splitext(img_path)
        base_name = os.path.basename(img_path).replace(ext, "")
        
        if base_name.endswith("flows"):
            continue

        mask_path = os.path.join(input_folder, base_name + MASK_SUFFIX)
        
        if not os.path.exists(mask_path):
            print(f"Skipping {base_name}: Mask missing.")
            continue

        print(f"Processing: {base_name}...")
        try:
            img = skimage.io.imread(img_path)
            img = normalize_image_for_plot(img)
            
            seg_data = np.load(mask_path, allow_pickle=True).item()
            masks = seg_data["masks"] 
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
            continue

        plot_mask_overlay(individual_dir, img, masks, filename_prefix=base_name)
        current_df = calculate_organoid_metrics(masks, base_name, MICRONS_PER_PIXEL)
        
        if current_df.empty:
            print(f"Skipping plots for {base_name} (no valid organoids).")
            continue
        
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

def add_stat_annotation(ax, data1, data2, metric):
    """
    Calculates Mann-Whitney U p-value between two datasets and adds
    a significance bracket with asterisks and the p-value to the plot.
    """
    val1 = data1[metric].dropna()
    val2 = data2[metric].dropna()
    
    if len(val1) == 0 or len(val2) == 0:
        return
        
    # Non-parametric independent test (appropriate for image properties)
    stat, p = stats.mannwhitneyu(val1, val2, alternative='two-sided')
    
    # Significance logic
    if p > 0.05: sig = "ns"
    elif p <= 0.0001: sig = "****"
    elif p <= 0.001: sig = "***"
    elif p <= 0.01: sig = "**"
    else: sig = "*"
        
    # Get y-axis limits to position the bracket correctly
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    
    # Position mappings
    y_bracket = ymax + yrange * 0.02
    y_text = y_bracket + yrange * 0.01
    
    # Draw bracket (Assuming boxplots are at x=0 and x=1)
    x1, x2 = 0, 1 
    ax.plot([x1, x1, x2, x2],[y_bracket - yrange*0.02, y_bracket, y_bracket, y_bracket - yrange*0.02], 
            lw=1.5, c='black')
    
    # Add text
    p_text = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
    ax.text((x1 + x2) * 0.5, y_text, f"{sig}\n({p_text})", 
            ha='center', va='bottom', color='black', fontsize=10)
    
    # Adjust y-limits to make room for the annotation
    ax.set_ylim(ymin, y_text + yrange * 0.18)

def plot_comparison(df1, df2, output_dir):
    print("\n--- Generating Comparison Graphs ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if df1.empty or df2.empty:
        print("Cannot compare: One or both datasets are empty.")
        return

    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Get exact conditions for ordering 
    cond_1 = df1["Condition"].iloc[0]
    cond_2 = df2["Condition"].iloc[0]
    
    title_str = f"{cond_1} vs. {cond_2}"
    filename_str = f"comparison_{cond_1}_vs_{cond_2}.png"
    csv_name = f"combined_comparison_data_{cond_1}_vs_{cond_2}.csv"

    combined_df.to_csv(os.path.join(output_dir, csv_name), index=False)

    # Increased figure height slightly to leave room for the statistical annotations
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Setting an explicit order guarantees cond_1 is at x=0 and cond_2 is at x=1
    plot_order = [cond_1, cond_2]
    
    # Plot 1: Average Axis
    sns.boxplot(data=combined_df, x="Condition", y="average_axis", hue="Condition", 
                legend=False, ax=axes[0], palette="Set2", order=plot_order)
    axes[0].set_title(f"Average Axis Length ({UNIT_NAME})")
    axes[0].set_ylabel(f"Length ({UNIT_NAME})")
    add_stat_annotation(axes[0], df1, df2, "average_axis")

    # Plot 2: Area
    sns.boxplot(data=combined_df, x="Condition", y="area", hue="Condition", 
                legend=False, ax=axes[1], palette="Set2", order=plot_order)
    axes[1].set_title(f"Organoid Area ({UNIT_NAME}²)")
    axes[1].set_ylabel(f"Area ({UNIT_NAME}²)")
    add_stat_annotation(axes[1], df1, df2, "area")
    
    # Plot 3: Circularity
    sns.boxplot(data=combined_df, x="Condition", y="circularity", hue="Condition", 
                legend=False, ax=axes[2], palette="Set2", order=plot_order)
    axes[2].set_title("Circularity (0-1)")
    axes[2].set_ylabel("Circularity Index")
    axes[2].set_ylim(0, 1.05) # Initialize bounds first before calculating text placement
    add_stat_annotation(axes[2], df1, df2, "circularity")

    plt.suptitle(f"{title_str} Comparison", fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename_str)
    plt.savefig(save_path)
    print(f"Comparison graph saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    
    print("====================================================")
    print("STEP 0: Sanitize TIFF files (Fix Bad Metadata)")
    print("====================================================")
    
    fix_tiff_files(TRAINING_SET_1)
    fix_tiff_files(TRAINING_SET_2)
    
    label_1 = get_condition_name_from_path(TRAINING_SET_1)
    label_2 = get_condition_name_from_path(TRAINING_SET_2)
    
    print(f"\nDetected Conditions:")
    print(f"  Set 1: {label_1}")
    print(f"  Set 2: {label_2}")
    
    print("\n====================================================")
    print("STEP 1: Analyze Images")
    print("====================================================")

    df_1 = process_batch(TRAINING_SET_1, f"{label_1.lower()}_consolidated", label_1)
    df_2 = process_batch(TRAINING_SET_2, f"{label_2.lower()}_consolidated", label_2)

    full_folder_name = os.path.basename(os.path.normpath(TRAINING_SET_1))
    match = re.search(r"day-\d+", full_folder_name)
    day_suffix = match.group() if match else full_folder_name

    comparison_folder_name = f"comparison_analysis_{day_suffix}"
    comparison_dir = os.path.join(BASE_OUTPUT_DIR, comparison_folder_name)
    
    plot_comparison(df_1, df_2, comparison_dir)