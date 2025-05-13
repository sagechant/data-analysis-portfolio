import pandas as pd
import numpy as np
# Plotting - Use Plotly for interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
# Seaborn is still useful for the heatmap color palette
import seaborn as sns
import matplotlib.pyplot as plt # Keep for color map access if needed

import os
import sys
import glob # For finding files
import tkinter as tk
from tkinter import filedialog

# --- Configuration ---

# Assumed frame interval (seconds) - based on previous metadata
# TODO: Consider making this interactive or reading from params file later
assumed_frame_interval = 5.0
# Assumed pixel size (micrometers per pixel) - based on previous metadata/setting
# TODO: Consider making this interactive or reading from params file later
pixel_size_um = 0.08
area_per_pixel_um2 = pixel_size_um * pixel_size_um if pixel_size_um else None

# Minimum track length filter requested by user
min_track_length_filter = 20

# Columns to analyze and plot
# Ensure column names exactly match those in the CSV files
# <<< Include integrated intensities for normalization >>>
metrics_to_analyze = [
    'vimentin_intensity_mean',
    'vimentin_intensity_integrated', # Base integrated intensity
    'fak_intensity_mean',
    'fak_intensity_integrated',    # Base integrated intensity
    'pcc',
    'm1_coeff',
    'm2_coeff',
    'icq',
    'area_pixels' # Base area measurement
    # Normalized metrics will be added later
]
# area_um2 will be added if possible

# Number of sample trajectories to plot
num_trajectory_samples = 50

# --- Helper Functions ---

def get_paths():
    """Uses Tkinter dialogs to get input CSV directory and output plot directory."""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window

    print("--- File Path Setup ---")
    print("Please select the directory containing the analysis CSV files...")
    csv_dir = filedialog.askdirectory(title="Select Directory Containing CSV Files")
    if not csv_dir:
        print("No input directory selected. Exiting.")
        sys.exit()
    print(f"Selected input directory: {csv_dir}")

    print("Please select the directory where output plots should be saved...")
    output_plot_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_plot_dir:
        print("No output directory selected. Exiting.")
        sys.exit()
    print(f"Selected output directory: {output_plot_dir}")
    print("-----------------------")
    os.makedirs(output_plot_dir, exist_ok=True) # Ensure output dir exists
    return csv_dir, output_plot_dir

def find_csv_files(csv_dir):
    """Finds relevant CSV files in the specified directory."""
    print(f"\nSearching for CSV files in: {csv_dir}")
    # Look for files ending with _D3D.csv or _tracks_filtered.csv
    pattern1 = os.path.join(csv_dir, "*_D3D.csv")
    pattern2 = os.path.join(csv_dir, "*_Analysis_tracks_filtered.csv") # From main script output
    csv_files_found = glob.glob(pattern1) + glob.glob(pattern2)
    # Filter out any potential temporary excel files if pattern1 is too broad
    csv_files_found = [f for f in csv_files_found if not os.path.basename(f).startswith('~$')]
    csv_files_found = sorted(list(set(csv_files_found))) # Remove duplicates and sort

    if not csv_files_found:
        print(f"Error: No CSV files matching patterns '..._D3D.csv' or '..._Analysis_tracks_filtered.csv' found in '{csv_dir}'.")
        sys.exit()

    print(f"Found {len(csv_files_found)} CSV files to process.")
    return csv_files_found

def load_and_combine_data(csv_files):
    """Loads data from a list of CSV paths, adds identifiers, and combines."""
    all_tracks_list = []
    print("\nLoading and combining CSV files...")
    loaded_file_count = 0
    for i, f_path in enumerate(csv_files):
        f_name = os.path.basename(f_path)
        print(f" Reading {f_name}...")
        try:
            # Handle potential delimiter issues based on filename pattern
            # Use a more robust check for the specific filename
            # Example: Use ';' only if filename *exactly* matches the known problematic one
            delimiter = ';' if f_name == "150318_mefko_gfpfakmcherryvim_002_visit_3_visit_2_D3D.csv" else ','
            if delimiter == ';': print(f"  --> Using delimiter ';'")

            df_single = pd.read_csv(f_path, delimiter=delimiter)

            # Check for essential columns
            if 'particle' not in df_single.columns:
                 print(f"  Warning: 'particle' column not found in {f_name}. Skipping this file.")
                 continue
            if 'frame' not in df_single.columns:
                 print(f"  Warning: 'frame' column not found in {f_name}. Skipping this file.")
                 continue

            # Add identifiers
            df_single['source_file'] = f_name
            # Ensure particle IDs are integers before converting to string
            df_single['particle'] = df_single['particle'].astype(int)
            df_single['global_track_id'] = f"{f_name}_" + df_single['particle'].astype(str)
            all_tracks_list.append(df_single)
            loaded_file_count += 1

        except pd.errors.EmptyDataError:
             print(f"  Warning: File {f_name} is empty. Skipping.")
        except Exception as e:
            print(f"  Error reading file {f_name}: {e}. Skipping.")

    if not all_tracks_list:
        print("Error: No valid data loaded from any CSV file. Exiting.")
        sys.exit()

    # Combine into a single dataframe
    combined_df = pd.concat(all_tracks_list, ignore_index=True)
    print(f"\nSuccessfully combined data from {loaded_file_count} files.")
    return combined_df

def filter_tracks_by_length(df, min_len):
    """Filters the combined dataframe to keep tracks >= min_len."""
    print(f"\nFiltering tracks: Keeping tracks with length >= {min_len} frames...")
    if 'global_track_id' not in df.columns:
        print("Error: 'global_track_id' column missing. Cannot filter.")
        return pd.DataFrame(), 0 # Return empty and 0 tracks

    n_total_tracks = df['global_track_id'].nunique()
    print(f"Total tracks found before length filtering: {n_total_tracks}")
    print(f"Total data points before length filtering: {len(df)}")

    track_lengths = df.groupby('global_track_id')['frame'].transform('count')
    filtered_df = df[track_lengths >= min_len].copy() # Use .copy()

    n_filtered_tracks = filtered_df['global_track_id'].nunique()

    if n_filtered_tracks == 0:
         print(f"Warning: No tracks remained after filtering (min length = {min_len}).")
    else:
        print(f"Tracks remaining after filtering: {n_filtered_tracks} (out of {n_total_tracks})")
        print(f"Data points remaining: {len(filtered_df)}")

    return filtered_df, n_filtered_tracks

def prepare_data(df, interval, px_size_um):
    """Adds time_sec and area_um2 columns."""
    print("\nPreparing data columns...")
    df['time_sec'] = df['frame'] * interval
    area_col = 'area_pixels' # Default
    area_label = 'Area (pixels)'
    area_px_sq = None
    if px_size_um:
        area_px_sq = px_size_um * px_size_um

    if area_px_sq is not None and 'area_pixels' in df.columns:
        print("Calculating area in um^2...")
        # Convert area_pixels to numeric first, coercing errors
        df['area_pixels'] = pd.to_numeric(df['area_pixels'], errors='coerce')
        df['area_um2'] = df['area_pixels'] * area_px_sq
        area_col = 'area_um2'
        area_label = 'Area (um^2)'
    elif 'area_pixels' not in df.columns:
        print("Warning: 'area_pixels' column not found.")
        area_col = None
        area_label = None
    # If px_size_um is None or area_pixels conversion failed, area_col remains 'area_pixels'
    return df, area_col, area_label

def normalize_intensity_by_first_frame(df, intensity_col, norm_col_name):
    """Normalizes intensity within each track to its first frame value."""
    print(f"Normalizing '{intensity_col}' to first frame value...")
    if intensity_col not in df.columns:
        print(f"  Warning: Column '{intensity_col}' not found. Cannot normalize.")
        df[norm_col_name] = np.nan
        return df

    # Ensure intensity column is numeric
    df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')

    # Get the intensity value of the first frame for each track
    # Need to handle tracks that might start with NaN intensity
    df_sorted = df.sort_values(by=['global_track_id', 'frame'])
    # Find the first NON-NaN value for each track
    first_valid_intensity = df_sorted.groupby('global_track_id')[intensity_col].transform(lambda x: x.loc[x.first_valid_index()] if x.first_valid_index() is not None else np.nan)

    # Normalize: Divide by first valid frame intensity. Handle potential zero division.
    # Replace 0 or NaN in first_intensity with NaN to avoid division errors or infinite results
    first_intensity_safe = first_valid_intensity.replace(0, np.nan)
    df[norm_col_name] = df[intensity_col] / first_intensity_safe

    # Report issues
    num_nan_norm = df[norm_col_name].isnull().sum()
    num_nan_orig = df[intensity_col].isnull().sum()
    num_zero_first = (first_valid_intensity == 0).sum() # Count where first valid was zero
    if num_nan_norm > num_nan_orig:
         print(f"  Note: {num_nan_norm - num_nan_orig} NaN values created/kept during normalization (likely due to NaN or zero initial intensity).")
         if num_zero_first > 0:
              print(f"    ({num_zero_first} instances where first valid intensity was zero).")

    return df


def get_final_metrics(df, requested_metrics, area_col_actual):
    """Returns list of metrics that actually exist in the dataframe."""
    metrics_final = []
    # Start with non-area metrics
    for m in requested_metrics:
        if 'area' not in m: # Avoid double-adding area if area_col_actual is area_pixels
            if m in df.columns:
                metrics_final.append(m)
            else:
                 print(f"Warning: Requested metric '{m}' not found in data. Skipping.")
    # Add the correct area column if it exists
    if area_col_actual and area_col_actual in df.columns:
        # Avoid adding area_pixels if area_um2 is the main one
        if not (area_col_actual == 'area_um2' and 'area_pixels' in metrics_final):
             if area_col_actual not in metrics_final: # Ensure no duplicates
                metrics_final.append(area_col_actual)
    elif 'area_pixels' in df.columns and 'area_pixels' not in metrics_final:
         # Fallback if area_um2 calculation failed but pixels exist
         metrics_final.append('area_pixels')

    return metrics_final

def calculate_time_course_stats(df, metrics):
    """Calculates mean and SEM per time point for given metrics."""
    print("Calculating average time courses...")
    if not metrics or df.empty:
        print("Warning: No metrics or data available for time course calculation.")
        return pd.DataFrame()

    # Define aggregation functions
    agg_funcs = ['mean', 'sem'] # Use standard strings
    # Apply aggregation
    time_stats = df.groupby('time_sec')[metrics].agg(agg_funcs)
    # Flatten the multi-index columns
    time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
    time_stats = time_stats.reset_index()
    return time_stats

def plot_interactive_time_courses(stats_df, metrics, area_col, area_label, n_tracks, min_len, outfile_html, outfile_png):
    """Generates and saves an interactive Plotly time course plot AND a static PNG."""
    print("Generating interactive time course plots...")
    if stats_df.empty or not metrics:
        print("Skipping time course plot: No data or metrics available.")
        return

    num_metrics = len(metrics)
    # Adjust subplot titles for normalized metrics
    subplot_titles_list = []
    for m in metrics:
        title = m.replace('_', ' ').title()
        if '_norm' in m:
            title = title.replace(' Norm', ' (Normalized to t0)')
        subplot_titles_list.append(title)

    fig = make_subplots(rows=num_metrics, cols=1, shared_xaxes=True,
                        subplot_titles=subplot_titles_list) # Use adjusted titles

    for i, metric in enumerate(metrics):
        mean_col = f"{metric}_mean"; sem_col = f"{metric}_sem"
        row_num = i + 1

        # Check if columns exist
        if mean_col not in stats_df.columns or sem_col not in stats_df.columns:
            print(f"  Skipping time course for {metric}: Mean/SEM columns not found.")
            fig.add_annotation(text=f"Data N/A for {metric}", xref="paper", yref="paper",
                               x=0.5, y=(num_metrics - row_num + 0.5) / num_metrics,
                               showarrow=False, row=row_num, col=1)
            continue

        mean_data = stats_df[mean_col]; sem_data = stats_df[sem_col].fillna(0)
        time_data = stats_df['time_sec']
        upper_bound = mean_data + sem_data; lower_bound = mean_data - sem_data

        # Plot SEM band
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_data, time_data[::-1]]), y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=False, name='SEM'
        ), row=row_num, col=1)
        # Plot Mean line
        fig.add_trace(go.Scatter(
            x=time_data, y=mean_data, line=dict(color='rgb(0,100,80)'), mode='lines', name='Mean'
        ), row=row_num, col=1)

        # Update y-axis title and ranges
        ylabel = metric.replace('_', ' ').title()
        if area_col and metric == area_col: ylabel = area_label
        elif metric == 'pcc': ylabel = 'PCC (Vim/FAK)'; fig.update_yaxes(range=[-1.1, 1.1], row=row_num, col=1)
        elif metric == 'm1_coeff': ylabel = 'Manders M1'; fig.update_yaxes(range=[-0.1, 1.1], row=row_num, col=1)
        elif metric == 'm2_coeff': ylabel = 'Manders M2'; fig.update_yaxes(range=[-0.1, 1.1], row=row_num, col=1)
        elif metric == 'icq': ylabel = 'ICQ (Vim/FAK)'; fig.update_yaxes(range=[-0.6, 0.6], row=row_num, col=1)
        # Add label for normalized intensity
        elif '_norm' in metric: ylabel = ylabel.replace(' Norm', ' (Norm. to t0)')
        fig.update_yaxes(title_text=ylabel, row=row_num, col=1)

    fig.update_layout(
        title=f"Average FA Dynamics Across {n_tracks} Tracks (Mean ± SEM, Length >= {min_len})",
        hovermode="x unified", height=250 * num_metrics, showlegend=False # Adjusted height per subplot
    )
    # Find last valid axis to set x-label
    last_valid_ax_idx = -1
    for i, metric in enumerate(metrics):
        if f"{metric}_mean" in stats_df.columns: last_valid_ax_idx = i
    if last_valid_ax_idx != -1:
         fig.update_xaxes(title_text="Time (seconds)", row=last_valid_ax_idx + 1, col=1)


    # Save as HTML
    try:
        pio.write_html(fig, outfile_html, auto_open=False)
        print(f"✅ Interactive time course plot saved to {outfile_html}")
    except Exception as e: print(f"Error saving Plotly time course HTML plot: {e}")

    # Save as PNG
    try:
        # Ensure kaleido is installed: pip install -U kaleido
        pio.write_image(fig, outfile_png, scale=2) # Use scale for better resolution
        print(f"✅ Static time course plot saved to {outfile_png}")
    except ValueError as ve:
         if "kaleido" in str(ve).lower(): # Check case-insensitively
              print("\n--- Kaleido Error ---")
              print("Saving static PNG plots requires the 'kaleido' package.")
              print("Please install it: pip install -U kaleido")
              print("Skipping PNG export for this plot.")
              print("---------------------\n")
         else: print(f"Error saving Plotly time course PNG plot: {ve}")
    except Exception as e: print(f"Error saving Plotly time course PNG plot: {e}")


def calculate_per_track_stats(df, metrics):
    """Calculates mean value per track for given metrics."""
    print("\nCalculating per-track averages...")
    if not metrics or df.empty:
        print("Warning: No metrics or data available for per-track calculation.")
        return pd.DataFrame()
    avg_stats = df.groupby('global_track_id')[metrics].mean(numeric_only=True).reset_index()
    return avg_stats

def plot_interactive_distributions(avg_df, metrics, area_col, area_label, n_tracks, min_len, outfile_html, outfile_png):
    """Generates and saves interactive Plotly distribution plots (violin/histogram) AND static PNG."""
    print("Generating interactive distribution plots...")
    if avg_df.empty or not metrics:
        print("Skipping distribution plot: No data or metrics available.")
        return

    num_metrics = len(metrics)
    n_cols = 3
    n_rows = (num_metrics + n_cols - 1) // n_cols
    # Create subplot titles dynamically
    subplot_titles = []
    valid_metrics_for_plot = []
    for metric in metrics:
         if metric in avg_df.columns and pd.notna(avg_df[metric]).any():
              title_label = metric.replace('_', ' ').title()
              if area_col and metric == area_col: title_label = f"Average {area_label}"
              elif metric == 'pcc': title_label = 'Average PCC (Vim/FAK)'
              elif metric == 'm1_coeff': title_label = 'Average Manders M1'
              elif metric == 'm2_coeff': title_label = 'Average Manders M2'
              elif metric == 'icq': title_label = 'Average ICQ (Vim/FAK)'
              # Add title for normalized
              elif '_norm' in metric: title_label = title_label.replace(' Norm', ' (Norm. to t0)')
              else: title_label = f"Average {title_label}"
              subplot_titles.append(title_label)
              valid_metrics_for_plot.append(metric)
         else:
              print(f"  Skipping distribution plot for '{metric}' (missing or all NaN after averaging).")

    if not valid_metrics_for_plot:
        print("Skipping distribution plot: No valid metrics to plot.")
        return

    # Recalculate grid size based on valid metrics
    num_metrics = len(valid_metrics_for_plot)
    n_rows = (num_metrics + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)

    current_row = 1
    current_col = 1
    for i, metric in enumerate(valid_metrics_for_plot):
        # Add Violin plot
        fig.add_trace(go.Violin(y=avg_df[metric], name=metric,
                                box_visible=True, meanline_visible=True,
                                points='all', jitter=0.3),
                      row=current_row, col=current_col)

        # Set specific ranges if needed
        if metric == 'pcc': fig.update_yaxes(range=[-1.1, 1.1], row=current_row, col=current_col)
        elif metric == 'm1_coeff': fig.update_yaxes(range=[-0.1, 1.1], row=current_row, col=current_col)
        elif metric == 'm2_coeff': fig.update_yaxes(range=[-0.1, 1.1], row=current_row, col=current_col)
        elif metric == 'icq': fig.update_yaxes(range=[-0.6, 0.6], row=current_row, col=current_col)
        # No specific range needed for normalized intensity by default

        # Move to next subplot position
        current_col += 1
        if current_col > n_cols:
            current_col = 1
            current_row += 1

    fig.update_layout(
        title=f"Distribution of Average Metrics per Track ({n_tracks} Tracks, Length >= {min_len})",
        height=350 * n_rows, # Adjust height
        showlegend=False
    )

     # Save as HTML
    try:
        pio.write_html(fig, outfile_html, auto_open=False)
        print(f"✅ Interactive distribution plot saved to {outfile_html}")
    except Exception as e: print(f"Error saving Plotly distribution HTML plot: {e}")

    # Save as PNG
    try:
        # Ensure kaleido is installed: pip install -U kaleido
        pio.write_image(fig, outfile_png, scale=2) # Use scale for better resolution
        print(f"✅ Static distribution plot saved to {outfile_png}")
    except ValueError as ve:
         if "kaleido" in str(ve).lower(): # Check case-insensitively
              print("\n--- Kaleido Error ---")
              print("Saving static PNG plots requires the 'kaleido' package.")
              print("Please install it: pip install -U kaleido")
              print("Skipping PNG export for this plot.")
              print("---------------------\n")
         else: print(f"Error saving Plotly distribution PNG plot: {ve}")
    except Exception as e: print(f"Error saving Plotly distribution PNG plot: {e}")


def plot_correlation_heatmap(avg_df, metrics, outfile):
    """Calculates and plots correlation heatmap for per-track averages."""
    print("\nGenerating correlation heatmap...")
    if avg_df.empty or len(metrics) < 2:
        print("Skipping correlation heatmap: Not enough data or metrics.")
        return

    # Select only the numeric metric columns for correlation
    # Ensure columns used for correlation actually exist in avg_df
    metrics_for_corr = [m for m in metrics if m in avg_df.columns]
    if len(metrics_for_corr) < 2:
         print("Skipping correlation heatmap: Not enough valid metric columns found.")
         return

    corr_df = avg_df[metrics_for_corr].corr()

    plt.figure(figsize=(12, 10)) # Increased size for more metrics
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8}) # Smaller annotation font
    plt.title('Correlation Matrix of Average Per-Track Metrics')
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"✅ Correlation heatmap saved to {outfile}")
    except Exception as e: print(f"Error saving correlation heatmap: {e}")
    plt.close() # Close the matplotlib figure

def plot_trajectories(df, n_samples, outfile_html, outfile_png):
    """Plots X/Y trajectories for a sample of tracks using Plotly AND saves PNG."""
    print(f"\nGenerating interactive trajectory plot for up to {n_samples} sample tracks...")
    if df.empty or 'global_track_id' not in df.columns or 'x' not in df.columns or 'y' not in df.columns:
        print("Skipping trajectory plot: Missing required columns (global_track_id, x, y) or no data.")
        return
    # Ensure x and y are numeric before calculating max
    if not pd.api.types.is_numeric_dtype(df['x']) or not pd.api.types.is_numeric_dtype(df['y']):
         print("Warning: x or y columns are not numeric. Cannot plot trajectories accurately.")
         return

    unique_tracks = df['global_track_id'].unique()
    # Ensure n_samples is not larger than the number of unique tracks
    n_samples = min(n_samples, len(unique_tracks))
    if n_samples == 0:
         print("Skipping trajectory plot: No tracks to sample.")
         return

    if len(unique_tracks) > n_samples:
        sampled_tracks = np.random.choice(unique_tracks, n_samples, replace=False)
        plot_title = f"Sample Trajectories (N={n_samples} of {len(unique_tracks)})"
    else:
        sampled_tracks = unique_tracks
        plot_title = f"All Trajectories (N={len(unique_tracks)})"

    sampled_df = df[df['global_track_id'].isin(sampled_tracks)]

    fig = go.Figure()

    # Add each track as a separate trace
    for track_id in sampled_tracks:
        track_data = sampled_df[sampled_df['global_track_id'] == track_id].sort_values('frame')
        # Extract just the base filename part for the legend name
        short_name = str(os.path.basename(track_id))
        fig.add_trace(go.Scatter(
            x=track_data['x'], y=track_data['y'], mode='lines+markers',
            marker=dict(size=3), name=short_name, # Shorter name for legend/hover
            hoverinfo='name+x+y+text', text=[f"Frame: {f}" for f in track_data['frame']]
        ))

    # Get image dimensions for setting axis limits
    # Calculate max only on the filtered data to avoid errors from potential NaNs
    max_x = df['x'].max(skipna=True)
    max_y = df['y'].max(skipna=True)
    if pd.isna(max_x) or pd.isna(max_y):
         print("Warning: Could not determine max X/Y from data, using fallback.")
         max_x = max_x if pd.notna(max_x) else 1024 # Fallback
         max_y = max_y if pd.notna(max_y) else 1024 # Fallback


    fig.update_layout(
        title=plot_title, xaxis_title="X coordinate (pixels)", yaxis_title="Y coordinate (pixels)",
        yaxis_autorange='reversed', xaxis_range=[0, max_x], yaxis_range=[max_y, 0],
        width=800, height=700, legend_title="Track ID", hovermode='closest'
    )
    # fig.update_yaxes(scaleanchor = "x", scaleratio = 1) # Equal aspect ratio if needed

    # Save as HTML
    try:
        pio.write_html(fig, outfile_html, auto_open=False)
        print(f"✅ Interactive trajectory plot saved to {outfile_html}")
    except Exception as e: print(f"Error saving Plotly trajectory HTML plot: {e}")

    # Save as PNG
    try:
        # Ensure kaleido is installed: pip install -U kaleido
        pio.write_image(fig, outfile_png, scale=2) # Use scale for better resolution
        print(f"✅ Static trajectory plot saved to {outfile_png}")
    except ValueError as ve:
         if "kaleido" in str(ve).lower(): # Check case-insensitively
              print("\n--- Kaleido Error ---")
              print("Saving static PNG plots requires the 'kaleido' package.")
              print("Please install it: pip install -U kaleido")
              print("Skipping PNG export for this plot.")
              print("---------------------\n")
         else: print(f"Error saving Plotly trajectory PNG plot: {ve}")
    except Exception as e: print(f"Error saving Plotly trajectory PNG plot: {e}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Get Paths
    csv_directory, output_directory = get_paths()

    # 2. Find Files
    csv_files_to_process = find_csv_files(csv_directory)

    # 3. Load and Combine
    combined_data = load_and_combine_data(csv_files_to_process)

    if not combined_data.empty:
        # 4. Filter Tracks
        filtered_data, num_filt_tracks = filter_tracks_by_length(combined_data, min_track_length_filter)

        if not filtered_data.empty and num_filt_tracks > 0: # Check if tracks remain
            # 5. Prepare Data (Add time, area_um2)
            prepared_data, area_col, area_label_str = prepare_data(filtered_data, assumed_frame_interval, pixel_size_um)

            # 6. Normalize Intensities (Mean and Integrated)
            if 'fak_intensity_mean' in prepared_data.columns:
                prepared_data = normalize_intensity_by_first_frame(prepared_data, 'fak_intensity_mean', 'fak_intensity_mean_norm')
            if 'vimentin_intensity_mean' in prepared_data.columns:
                 prepared_data = normalize_intensity_by_first_frame(prepared_data, 'vimentin_intensity_mean', 'vimentin_intensity_mean_norm')
            if 'fak_intensity_integrated' in prepared_data.columns: # <<< Normalize integrated
                prepared_data = normalize_intensity_by_first_frame(prepared_data, 'fak_intensity_integrated', 'fak_intensity_integrated_norm')
            if 'vimentin_intensity_integrated' in prepared_data.columns: # <<< Normalize integrated
                 prepared_data = normalize_intensity_by_first_frame(prepared_data, 'vimentin_intensity_integrated', 'vimentin_intensity_integrated_norm')

            # 7. Define Final Metrics List (including normalized)
            metrics_to_analyze_with_norm = metrics_to_analyze.copy()
            if 'fak_intensity_mean_norm' in prepared_data.columns:
                metrics_to_analyze_with_norm.append('fak_intensity_mean_norm')
            if 'vimentin_intensity_mean_norm' in prepared_data.columns:
                metrics_to_analyze_with_norm.append('vimentin_intensity_mean_norm')
            if 'fak_intensity_integrated_norm' in prepared_data.columns: # <<< Add normalized integrated
                metrics_to_analyze_with_norm.append('fak_intensity_integrated_norm')
            if 'vimentin_intensity_integrated_norm' in prepared_data.columns: # <<< Add normalized integrated
                metrics_to_analyze_with_norm.append('vimentin_intensity_integrated_norm')

            final_metrics = get_final_metrics(prepared_data, metrics_to_analyze_with_norm, area_col)

            if final_metrics:
                # 8. Convert relevant columns to numeric (including coordinates)
                print("\nConverting relevant columns to numeric...")
                cols_to_convert = final_metrics + [col for col in ['x', 'y'] if col in prepared_data.columns and col not in final_metrics]
                cols_successfully_converted = []
                for col in cols_to_convert:
                    if col in prepared_data.columns:
                        # Check if already numeric before converting
                        if not pd.api.types.is_numeric_dtype(prepared_data[col]):
                            print(f"  Converting column '{col}' to numeric (errors='coerce')...")
                            original_dtype = prepared_data[col].dtype
                            prepared_data[col] = pd.to_numeric(prepared_data[col], errors='coerce')
                            if original_dtype != prepared_data[col].dtype:
                                print(f"    Data type changed for '{col}' from {original_dtype} to {prepared_data[col].dtype}")
                            # Report how many values were coerced to NaN
                            num_nan = prepared_data[col].isnull().sum()
                            # Only print warning if there were actually NaNs introduced by coercion
                            if num_nan > 0 and not pd.api.types.is_numeric_dtype(original_dtype):
                                 print(f"    Warning: Found {num_nan} non-numeric values in '{col}', converted to NaN.")
                        # else: # Optional: print confirmation it's already numeric
                        #     print(f"  Column '{col}' is already numeric ({prepared_data[col].dtype}).")
                        cols_successfully_converted.append(col) # Keep track of cols that are now numeric (or were already)
                    else:
                         print(f"  Skipping conversion for '{col}': Column not found.")

                # Update final_metrics based on successful conversion and existence
                final_metrics = [m for m in final_metrics if m in cols_successfully_converted]

                # Drop rows where essential numeric columns became NaN after coercion
                essential_cols = [col for col in ['x', 'y'] if col in prepared_data.columns] # Check existence first
                if area_col and area_col in prepared_data.columns:
                     essential_cols.append(area_col)

                if essential_cols: # Only drop if essential columns exist
                    initial_rows = len(prepared_data)
                    prepared_data.dropna(subset=essential_cols, inplace=True)
                    rows_dropped = initial_rows - len(prepared_data)
                    if rows_dropped > 0:
                        print(f"Warning: Dropped {rows_dropped} rows containing NaN in essential columns ({', '.join(essential_cols)}) after numeric conversion.")

                # Recalculate number of tracks after potential row drops
                if 'global_track_id' in prepared_data.columns:
                     num_filt_tracks = prepared_data['global_track_id'].nunique()
                     if num_filt_tracks == 0:
                          print("Error: All tracks were removed after dropping rows with NaN in essential columns. Exiting.")
                          sys.exit()
                else: # Should not happen if loaded correctly
                     num_filt_tracks = 0
                     print("Error: 'global_track_id' column missing after processing. Exiting.")
                     sys.exit()

                print(f"Tracks remaining after NaN drop: {num_filt_tracks}")


                # 9. Calculate Time Course Stats
                time_stats_df = calculate_time_course_stats(prepared_data, final_metrics)

                # 10. Plot Time Courses (HTML and PNG)
                ts_plot_file_html = os.path.join(output_directory, f"Combined_TimeSeries_Interactive_Filt{min_track_length_filter}.html")
                ts_plot_file_png = os.path.join(output_directory, f"Combined_TimeSeries_Static_Filt{min_track_length_filter}.png")
                plot_interactive_time_courses(time_stats_df, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, ts_plot_file_html, ts_plot_file_png)

                # 11. Calculate Per-Track Stats
                avg_track_stats_df = calculate_per_track_stats(prepared_data, final_metrics)

                # 12. Plot Distributions (HTML and PNG)
                dist_plot_file_html = os.path.join(output_directory, f"Combined_Distribution_Interactive_Filt{min_track_length_filter}.html")
                dist_plot_file_png = os.path.join(output_directory, f"Combined_Distribution_Static_Filt{min_track_length_filter}.png")
                plot_interactive_distributions(avg_track_stats_df, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, dist_plot_file_html, dist_plot_file_png)

                # 13. Plot Correlation Heatmap (PNG only)
                heatmap_plot_file = os.path.join(output_directory, f"Combined_Correlation_Heatmap_Filt{min_track_length_filter}.png")
                plot_correlation_heatmap(avg_track_stats_df, final_metrics, heatmap_plot_file)

                # 14. Plot Trajectories (HTML and PNG)
                traj_plot_file_html = os.path.join(output_directory, f"Combined_Trajectories_Sample_Interactive_Filt{min_track_length_filter}.html")
                traj_plot_file_png = os.path.join(output_directory, f"Combined_Trajectories_Sample_Static_Filt{min_track_length_filter}.png")
                plot_trajectories(prepared_data, num_trajectory_samples, traj_plot_file_html, traj_plot_file_png)

                # 15. Print Head of Filtered Data
                print("\nFirst 5 rows of combined & filtered data:")
                print(prepared_data.head())

            else:
                 print("No valid metrics left to analyze after checking columns.")
        # No explicit else needed here, message printed in filter_tracks function

    else:
        print("Exiting: No data was loaded.")

    print("\nScript finished.")
