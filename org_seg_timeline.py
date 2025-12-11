import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import re

# ================= CONFIGURATION =================
# Path to the folder containing all your "comparison_analysis_day-X" folders
# If this script is inside that folder, you can use os.getcwd()
PARENT_FOLDER = r"C:\Users\Marwin\Desktop\organoids\251204"

# Where to save the final timeline graphs
OUTPUT_DIR = os.path.join(PARENT_FOLDER, "Longitudinal_Analysis_Results")

# Metrics to analyze
METRICS = {
    "area": "Area (µm²)",
    "average_axis": "Average Diameter (µm)",
    "circularity": "Circularity (0-1)"
}
# =================================================

def collect_longitudinal_data(parent_dir):
    """
    Scans for comparison folders, reads the CSVs, and tags them with the Day.
    """
    print(f"Scanning directory: {parent_dir}")
    
    # Find all comparison folders
    search_path = os.path.join(parent_dir, "comparison_analysis_day-*")
    folders = glob.glob(search_path)
    
    if not folders:
        print("No 'comparison_analysis_day-X' folders found!")
        return pd.DataFrame()

    all_data = []

    for folder in folders:
        # 1. Extract Day Number using Regex
        folder_name = os.path.basename(folder)
        match = re.search(r"day-(\d+)", folder_name)
        
        if not match:
            print(f"Skipping folder (no day number found): {folder_name}")
            continue
            
        day_num = int(match.group(1)) # Convert "5" to integer 5
        
        # 2. Load the CSV
        csv_path = os.path.join(folder, "combined_comparison_data.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {folder_name}: CSV not found.")
            continue
            
        try:
            df = pd.read_csv(csv_path)
            # Add the Day column
            df["Day"] = day_num
            all_data.append(df)
            print(f"Loaded Day {day_num} from {folder_name}")
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if not all_data:
        return pd.DataFrame()

    # Combine and sort by Day
    master_df = pd.concat(all_data, ignore_index=True)
    master_df = master_df.sort_values(by="Day")
    return master_df

def plot_timelines(df, output_folder):
    """
    Generates Line plots (Trends) and Box plots (Distributions) over time.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Ensure Day is treated as a number for sorting, but categorical for plotting if needed
    days = sorted(df["Day"].unique())
    print(f"Plotting data for Days: {days}")

    sns.set_theme(style="whitegrid")

    for col, nice_name in METRICS.items():
        if col not in df.columns:
            continue

        # --- PLOT 1: LINE PLOT (Trend of Means with Confidence Interval) ---
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
        
        # Force X-axis to show integer ticks only
        plt.xticks(days)
        plt.title(f"Timeline: {nice_name} (Mean ± 95% CI)", fontsize=15)
        plt.ylabel(nice_name)
        plt.xlabel("Day")
        
        save_path = os.path.join(output_folder, f"Timeline_Trend_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        # --- PLOT 2: BOX PLOT (Distribution Comparison) ---
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df, 
            x="Day", 
            y=col, 
            hue="Condition", 
            palette="Set2",
            gap=0.1 # Adds gap between grouped boxes (seaborn > v0.13)
        )
        
        plt.title(f"Distribution Timeline: {nice_name}", fontsize=15)
        plt.ylabel(nice_name)
        plt.xlabel("Day")
        
        save_path = os.path.join(output_folder, f"Timeline_Distribution_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

def save_stats_summary(df, output_folder):
    """
    Saves a CSV table of Mean/Std per Day per Condition.
    """
    summary = df.groupby(["Day", "Condition"])[list(METRICS.keys())].agg(['mean', 'std', 'count'])
    csv_path = os.path.join(output_folder, "longitudinal_summary_stats.csv")
    summary.to_csv(csv_path)
    print(f"Summary stats saved to: {csv_path}")

if __name__ == "__main__":
    
    # 1. Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Collect Data
    print("--- Starting Longitudinal Analysis ---")
    master_df = collect_longitudinal_data(PARENT_FOLDER)
    
    if not master_df.empty:
        # Save the master compiled CSV
        master_df.to_csv(os.path.join(OUTPUT_DIR, "all_days_compiled_data.csv"), index=False)
        print(f"Compiled data saved. Total organoids analyzed: {len(master_df)}")
        
        # 3. Plot
        plot_timelines(master_df, OUTPUT_DIR)
        
        # 4. Stats
        save_stats_summary(master_df, OUTPUT_DIR)
        
        print("\nAnalysis Complete! Check the folder: 'Longitudinal_Analysis_Results'")
    else:
        print("Could not generate plots due to missing data.")