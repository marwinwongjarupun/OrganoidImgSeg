import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import re
import numpy as np
from scipy.stats import ttest_ind

# ================= CONFIGURATION =================
# Path to the folder containing all your "comparison_analysis_day-X" folders
PARENT_FOLDER = r"D:\Organoid images\260126"

# Where to save the final timeline graphs (Sub-folders will be created here)
OUTPUT_DIR = os.path.join(PARENT_FOLDER, "longitudinal_analysis_results")

# Metrics to analyze
METRICS = {
    "area": "Area (µm²)",
    "average_axis": "Average Diameter (µm)",
    "circularity": "Circularity (0-1)"
}
# =================================================

def get_unique_comparison_filenames(parent_dir):
    """
    Scans all day folders to find unique comparison CSV filenames.
    Returns a set of filenames (e.g., {'combined_comparison_data_A_vs_B.csv', ...})
    """
    unique_files = set()
    search_path = os.path.join(parent_dir, "comparison_analysis_day-*")
    day_folders = glob.glob(search_path)
    
    for folder in day_folders:
        # Look for any CSV starting with combined_comparison_data
        csv_files = glob.glob(os.path.join(folder, "combined_comparison_data*.csv"))
        for f in csv_files:
            unique_files.add(os.path.basename(f))
            
    return sorted(list(unique_files))

def collect_longitudinal_data(parent_dir, target_filename):
    """
    Scans for comparison folders and reads specifically the 'target_filename'.
    """
    print(f"Scanning for file: {target_filename}")
    
    search_path = os.path.join(parent_dir, "comparison_analysis_day-*")
    folders = glob.glob(search_path)
    
    if not folders:
        print("No 'comparison_analysis_day-X' folders found!")
        return pd.DataFrame()

    all_data = []

    for folder in folders:
        folder_name = os.path.basename(folder)
        match = re.search(r"day-(\d+)", folder_name)
        
        if not match:
            continue
            
        day_num = int(match.group(1))
        
        # Look for the specific filename passed to the function
        csv_path = os.path.join(folder, target_filename)
        
        if not os.path.exists(csv_path):
            # It is possible distinct comparisons don't exist for every single day
            continue
            
        try:
            df = pd.read_csv(csv_path)
            df["Day"] = day_num
            all_data.append(df)
            # print(f"  -> Found in Day {day_num}")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    master_df = pd.concat(all_data, ignore_index=True)
    master_df = master_df.sort_values(by="Day")
    return master_df

def get_p_value_annotation(p_val):
    """Convert p-value to star notation."""
    if p_val > 0.05:
        return "ns"
    elif p_val <= 0.0001:
        return "****"
    elif p_val <= 0.001:
        return "***"
    elif p_val <= 0.01:
        return "**"
    else:
        return "*"

def add_stat_annotation(ax, df, metric_col, days):
    """
    Performs T-test between the two conditions for each day and adds stars.
    """
    conditions = df["Condition"].unique()
    
    if len(conditions) != 2:
        return

    cond1, cond2 = conditions[0], conditions[1]
    
    # Calculate Y-axis limit to place stars
    y_max = df[metric_col].max()
    y_range = y_max - df[metric_col].min()
    text_y_pos = y_max + (y_range * 0.05) 
    
    for i, day in enumerate(days):
        day_data = df[df["Day"] == day]
        
        group1 = day_data[day_data["Condition"] == cond1][metric_col]
        group2 = day_data[day_data["Condition"] == cond2][metric_col]
        
        if len(group1) > 1 and len(group2) > 1:
            # Welch's t-test
            stat, p_val = ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
            label = get_p_value_annotation(p_val)
            
            ax.text(i, text_y_pos, label, ha='center', va='bottom', 
                    color='black', fontsize=12, fontweight='bold')
            
    ax.set_ylim(top=text_y_pos + (y_range * 0.1))

def plot_timelines(df, output_folder, comparison_name):
    """
    Generates Line plots and Box plots.
    """
    if df.empty:
        print("No data to plot.")
        return

    days = sorted(df["Day"].unique())
    print(f"Plotting '{comparison_name}' for Days: {days}")

    sns.set_theme(style="whitegrid")

    for col, nice_name in METRICS.items():
        if col not in df.columns:
            continue

        # --- PLOT 1: LINE PLOT ---
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, 
            x="Day", 
            y=col, 
            hue="Condition", 
            style="Condition", 
            markers=True, 
            dashes=False,
            palette="Set2",
            linewidth=2.5,
            markersize=8
        )
        
        plt.xticks(days)
        plt.title(f"Timeline ({comparison_name}): {nice_name}", fontsize=15)
        plt.ylabel(nice_name)
        plt.xlabel("Day")
        
        save_path = os.path.join(output_folder, f"Timeline_Trend_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        # --- PLOT 2: BOX PLOT ---
        plt.figure(figsize=(10, 7))
        ax_box = sns.boxplot(
            data=df, 
            x="Day", 
            y=col, 
            hue="Condition", 
            palette="Set2",
            gap=0.1
        )
        
        try:
            add_stat_annotation(ax_box, df, col, days)
        except Exception as e:
            print(f"Could not add stats for {col}: {e}")
        
        plt.title(f"Distribution ({comparison_name}): {nice_name}", fontsize=15)
        plt.ylabel(nice_name)
        plt.xlabel("Day")
        
        save_path = os.path.join(output_folder, f"Timeline_Distribution_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

def save_stats_summary(df, output_folder):
    """
    Saves a CSV table of Mean/Std per Day and P-values.
    """
    summary = df.groupby(["Day", "Condition"])[list(METRICS.keys())].agg(['mean', 'std', 'count'])
    
    p_values = []
    days = sorted(df["Day"].unique())
    conditions = df["Condition"].unique()
    
    if len(conditions) == 2:
        cond1, cond2 = conditions[0], conditions[1]
        for day in days:
            row = {"Day": day}
            day_data = df[df["Day"] == day]
            
            for metric in METRICS.keys():
                if metric not in df.columns: continue
                g1 = day_data[day_data["Condition"] == cond1][metric]
                g2 = day_data[day_data["Condition"] == cond2][metric]
                
                if len(g1) > 1 and len(g2) > 1:
                    _, p = ttest_ind(g1, g2, equal_var=False, nan_policy='omit')
                    row[f"{metric}_p_value"] = p
                    row[f"{metric}_significance"] = get_p_value_annotation(p)
                else:
                    row[f"{metric}_p_value"] = None
            
            p_values.append(row)
            
        p_df = pd.DataFrame(p_values)
        p_path = os.path.join(output_folder, "longitudinal_p_values.csv")
        p_df.to_csv(p_path, index=False)

    csv_path = os.path.join(output_folder, "longitudinal_summary_stats.csv")
    summary.to_csv(csv_path)

if __name__ == "__main__":
    
    print("--- Starting Multi-Timeline Longitudinal Analysis ---")
    
    # 1. Find all distinct comparison files (e.g., A vs B, C vs D)
    comparison_files = get_unique_comparison_filenames(PARENT_FOLDER)
    
    if not comparison_files:
        print("No 'combined_comparison_data*.csv' files found in day folders.")
    else:
        print(f"Found {len(comparison_files)} unique comparisons to analyze: {comparison_files}")

    # 2. Loop through each distinct comparison type found
    for target_file in comparison_files:
        
        # Create a clean name for the output folder (remove extension and prefix)
        clean_name = target_file.replace("combined_comparison_data", "").replace(".csv", "").strip("_")
        if not clean_name: 
            clean_name = "Main_Comparison" # Fallback if file is just combined_comparison_data.csv
            
        print(f"\nProcessing: {clean_name} ({target_file})")
        
        # Create a specific sub-folder for this comparison
        current_output_dir = os.path.join(OUTPUT_DIR, f"Analysis_{clean_name}")
        os.makedirs(current_output_dir, exist_ok=True)
        
        # Collect data specifically for this file pattern
        master_df = collect_longitudinal_data(PARENT_FOLDER, target_file)
        
        if not master_df.empty:
            # Save raw data
            master_df.to_csv(os.path.join(current_output_dir, "compiled_data.csv"), index=False)
            
            # Run Plots and Stats
            plot_timelines(master_df, current_output_dir, clean_name)
            save_stats_summary(master_df, current_output_dir)
            print(f"-> Done. Results in: {current_output_dir}")
        else:
            print(f"-> Skipped. No data found for {target_file}.")

    print("\nAll analyses complete.")