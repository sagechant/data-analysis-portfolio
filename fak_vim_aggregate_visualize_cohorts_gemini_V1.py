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
# Clustering & Scaling
# <<< Need to install scikit-learn: pip install -U scikit-learn >>>
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer # To handle potential NaNs before scaling/clustering
    from sklearn.decomposition import PCA # Optional: for visualizing clusters
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found. Clustering analysis will be skipped.")
    print("         Install it using: pip install -U scikit-learn")
    SKLEARN_AVAILABLE = False

import logging # Added for logger definition


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

# Columns to analyze and plot initially (more added later)
metrics_to_analyze = [
    'vimentin_intensity_mean', 'vimentin_intensity_integrated',
    'fak_intensity_mean', 'fak_intensity_integrated',
    'pcc', 'm1_coeff', 'm2_coeff', 'icq',
    'area_pixels'
]
# area_um2 will be added if possible

# Number of sample trajectories to plot
num_trajectory_samples = 50
# Number of clusters for K-Means (can be adjusted based on Elbow plot)
# Set to None to generate Elbow plot first, then re-run with chosen k
n_clusters_kmeans = 3 # Example: Start with 3 clusters

# --- Helper Functions ---

def setup_logging(level=logging.INFO):
    """Sets up basic logging."""
    # Remove previous handlers if any exist to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout # Log to console
    )
    # Disable matplotlib font manager logs which can be verbose
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


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
    logger.info(f"Searching for CSV files in: {csv_dir}")
    # Look for files ending with _D3D.csv or _tracks_filtered.csv
    pattern1 = os.path.join(csv_dir, "*_D3D.csv")
    pattern2 = os.path.join(csv_dir, "*_Analysis_tracks_filtered.csv") # From main script output
    csv_files_found = glob.glob(pattern1) + glob.glob(pattern2)
    # Filter out any potential temporary excel files if pattern1 is too broad
    csv_files_found = [f for f in csv_files_found if not os.path.basename(f).startswith('~$')]
    csv_files_found = sorted(list(set(csv_files_found))) # Remove duplicates and sort

    if not csv_files_found:
        logger.error(f"No CSV files matching patterns '..._D3D.csv' or '..._Analysis_tracks_filtered.csv' found in '{csv_dir}'. Exiting.")
        sys.exit()

    logger.info(f"Found {len(csv_files_found)} CSV files to process.")
    return csv_files_found

def load_and_combine_data(csv_files):
    """Loads data from a list of CSV paths, adds identifiers, and combines."""
    all_tracks_list = []
    logger.info("Loading and combining CSV files...")
    loaded_file_count = 0
    for i, f_path in enumerate(csv_files):
        f_name = os.path.basename(f_path)
        logger.info(f" Reading {f_name}...")
        try:
            # Handle potential delimiter issues based on filename pattern
            # Use a more robust check for the specific filename
            # Example: Use ';' only if filename *exactly* matches the known problematic one
            delimiter = ';' if f_name == "150318_mefko_gfpfakmcherryvim_002_visit_3_visit_2_D3D.csv" else ','
            if delimiter == ';': logger.info(f"  --> Using delimiter ';'")

            df_single = pd.read_csv(f_path, delimiter=delimiter)

            # Check for essential columns
            if 'particle' not in df_single.columns:
                 logger.warning(f"  Skipping {f_name}: missing 'particle' column.")
                 continue
            if 'frame' not in df_single.columns:
                 logger.warning(f"  Skipping {f_name}: missing 'frame' column.")
                 continue

            # Add identifiers
            df_single['source_file'] = f_name
            # Ensure particle IDs are integers before converting to string
            df_single['particle'] = df_single['particle'].astype(int)
            df_single['global_track_id'] = f"{f_name}_" + df_single['particle'].astype(str)
            all_tracks_list.append(df_single)
            loaded_file_count += 1

        except pd.errors.EmptyDataError:
             logger.warning(f"  File {f_name} is empty. Skipping.")
        except Exception as e:
            logger.error(f"  Error reading file {f_name}: {e}. Skipping.")

    if not all_tracks_list:
        logger.error("No valid data loaded from any CSV file. Exiting.")
        sys.exit()

    # Combine into a single dataframe
    combined_df = pd.concat(all_tracks_list, ignore_index=True)
    logger.info(f"Successfully combined data from {loaded_file_count} files.")
    return combined_df

def filter_tracks_by_length(df, min_len):
    """Filters the combined dataframe to keep tracks >= min_len."""
    logger.info(f"Filtering tracks: Keeping tracks with length >= {min_len} frames...")
    if 'global_track_id' not in df.columns:
        logger.error(" 'global_track_id' column missing. Cannot filter.")
        return pd.DataFrame(), 0 # Return empty and 0 tracks

    n_total_tracks = df['global_track_id'].nunique()
    logger.info(f"Total tracks found before length filtering: {n_total_tracks}")
    logger.info(f"Total data points before length filtering: {len(df)}")

    track_lengths = df.groupby('global_track_id')['frame'].transform('count')
    df['track_length'] = track_lengths # Add length column
    filtered_df = df[df['track_length'] >= min_len].copy() # Use .copy()

    n_filtered_tracks = filtered_df['global_track_id'].nunique()

    if n_filtered_tracks == 0:
         logger.warning(f"No tracks remained after filtering (min length = {min_len}).")
    else:
        logger.info(f"Tracks remaining after filtering: {n_filtered_tracks} (out of {n_total_tracks})")
        logger.info(f"Data points remaining: {len(filtered_df)}")

    return filtered_df, n_filtered_tracks

def prepare_data(df, interval, px_size_um):
    """Adds time_sec and area_um2 columns."""
    logger.info("Preparing data columns...")
    df['time_sec'] = df['frame'] * interval
    area_col = 'area_pixels' # Default
    area_label = 'Area (pixels)'
    area_px_sq = None
    if px_size_um:
        area_px_sq = px_size_um * px_size_um

    if area_px_sq is not None and 'area_pixels' in df.columns:
        logger.info("Calculating area in um^2...")
        # Convert area_pixels to numeric first, coercing errors
        df['area_pixels'] = pd.to_numeric(df['area_pixels'], errors='coerce')
        df['area_um2'] = df['area_pixels'] * area_px_sq
        area_col = 'area_um2'
        area_label = 'Area (um^2)'
    elif 'area_pixels' not in df.columns:
        logger.warning(" 'area_pixels' column not found.")
        area_col = None
        area_label = None
    # If px_size_um is None or area_pixels conversion failed, area_col remains 'area_pixels'
    return df, area_col, area_label

def normalize_intensity_by_first_frame(df, intensity_col, norm_col_name):
    """Normalizes intensity within each track to its first frame value."""
    logger.info(f"Normalizing '{intensity_col}' to first frame value...")
    if intensity_col not in df.columns:
        logger.warning(f"  Column '{intensity_col}' not found. Cannot normalize.")
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
         logger.info(f"  Note: {num_nan_norm - num_nan_orig} NaN values created/kept during normalization (likely due to NaN or zero initial intensity).")
         if num_zero_first > 0:
              logger.info(f"    ({num_zero_first} instances where first valid intensity was zero).")

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
                 logger.warning(f"Requested metric '{m}' not found in data. Skipping.")
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

def convert_and_clean_numeric(df, cols_to_check):
     """Converts specified columns to numeric and drops rows with NaNs in essential cols."""
     logger.info("Converting relevant columns to numeric and cleaning NaNs...")
     essential_cols = [col for col in ['x', 'y'] if col in df.columns] # Start with coords
     if 'area_pixels' in df.columns: essential_cols.append('area_pixels')
     if 'area_um2' in df.columns: essential_cols.append('area_um2')

     cols_to_convert = list(set(cols_to_check + essential_cols)) # Ensure essential cols are checked

     for col in cols_to_convert:
         if col in df.columns:
             # Check if already numeric before converting
             if not pd.api.types.is_numeric_dtype(df[col]):
                 logger.info(f"  Converting column '{col}' to numeric (errors='coerce')...")
                 original_dtype = df[col].dtype
                 df[col] = pd.to_numeric(df[col], errors='coerce')
                 num_nan = df[col].isnull().sum()
                 # Only print warning if there were actually NaNs introduced by coercion
                 if num_nan > 0 and not pd.api.types.is_numeric_dtype(original_dtype):
                      logger.warning(f"    Found {num_nan} non-numeric values in '{col}', converted to NaN.")
             # else: # Optional: print confirmation it's already numeric
             #     logger.debug(f"  Column '{col}' is already numeric ({df[col].dtype}).")
             # cols_successfully_converted.append(col) # Keep track of cols that are now numeric (or were already)
         else:
              logger.warning(f"  Skipping conversion for '{col}': Column not found.")

     # Drop rows where essential numeric columns became NaN after coercion
     essential_cols_exist = [c for c in essential_cols if c in df.columns]
     if essential_cols_exist:
         initial_rows = len(df)
         df.dropna(subset=essential_cols_exist, inplace=True)
         rows_dropped = initial_rows - len(df)
         if rows_dropped > 0:
             logger.warning(f"Dropped {rows_dropped} rows containing NaN in essential columns ({', '.join(essential_cols_exist)}) after numeric conversion.")
     return df


def calculate_time_course_stats(df, metrics, group_col=None):
    """Calculates mean and SEM per time point, optionally grouped."""
    logger.info(f"Calculating average time courses{f' grouped by {group_col}' if group_col else ''}...")
    if not metrics or df.empty: logger.warning("No metrics/data for time course."); return pd.DataFrame()
    agg_funcs = ['mean', 'sem']
    grouping_cols = ['time_sec']
    if group_col and group_col in df.columns: grouping_cols.append(group_col)
    else: group_col = None # Ensure group_col is None if not valid
    time_stats = df.groupby(grouping_cols)[metrics].agg(agg_funcs)
    time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
    return time_stats.reset_index()


def save_plotly(fig, html_path, png_path, logger):
    """Saves a Plotly figure as HTML and PNG."""
    # Save as HTML
    try:
        pio.write_html(fig, html_path, auto_open=False)
        logger.info(f"✅ Saved: {html_path}")
    except Exception as e: logger.error(f"Error saving Plotly HTML plot: {e}")
    # Save as PNG
    try:
        # Ensure kaleido is installed: pip install -U kaleido
        pio.write_image(fig, png_path, scale=2) # Use scale for better resolution
        logger.info(f"✅ Saved: {png_path}")
    except ValueError as ve:
         if "kaleido" in str(ve).lower(): # Check case-insensitively
              logger.warning("\n--- Kaleido Error ---\nSaving static PNG plots requires the 'kaleido' package.\nPlease install it: pip install -U kaleido\nSkipping PNG export for this plot.\n---------------------\n")
         else: logger.error(f"Error saving Plotly PNG plot: {ve}")
    except Exception as e: logger.error(f"Error saving Plotly PNG plot: {e}")


def plot_interactive_time_courses(stats_df, metrics, area_col, area_label, n_tracks, min_len, outfile_html, outfile_png, group_col=None, title_suffix=""):
    """Generates interactive Plotly time course plot (optionally grouped) and static PNG."""
    logger.info(f"Generating interactive time course plots{f' grouped by {group_col}' if group_col else ''}...")
    if stats_df.empty or not metrics: logger.warning("Skipping: No data/metrics."); return

    num_metrics = len(metrics)
    subplot_titles_list = []
    valid_metrics_plot = [] # Keep track of metrics actually plotted
    for m in metrics:
        mean_col = f"{m}_mean"; sem_col = f"{m}_sem"
        if mean_col in stats_df.columns and sem_col in stats_df.columns:
             title = m.replace('_', ' ').title()
             if '_norm' in m: title = title.replace(' Norm', ' (Norm. to t0)')
             subplot_titles_list.append(title)
             valid_metrics_plot.append(m)
        else:
             logger.warning(f"  Skipping time course for {m}: Mean/SEM columns not found.")

    if not valid_metrics_plot: logger.warning("Skipping: No valid metrics found for time course plot."); return
    num_metrics = len(valid_metrics_plot) # Update count based on valid metrics

    fig = make_subplots(rows=num_metrics, cols=1, shared_xaxes=True, subplot_titles=subplot_titles_list)
    groups = stats_df[group_col].unique() if group_col and group_col in stats_df else [None]
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

    for i, metric in enumerate(valid_metrics_plot): # Iterate through valid metrics only
        row_num = i + 1
        for j, group_val in enumerate(groups):
            group_name = str(group_val) if group_val is not None else 'Overall'
            color_rgba = f'rgba({int(colors[j][0]*255)}, {int(colors[j][1]*255)}, {int(colors[j][2]*255)}, 1)'
            fill_color_rgba = f'rgba({int(colors[j][0]*255)}, {int(colors[j][1]*255)}, {int(colors[j][2]*255)}, 0.2)'

            group_stats = stats_df[stats_df[group_col] == group_val] if group_val is not None else stats_df
            mean_col, sem_col = f"{metric}_mean", f"{metric}_sem"
            mean_data = group_stats[mean_col]; sem_data = group_stats[sem_col].fillna(0)
            time_data = group_stats['time_sec']
            upper = mean_data + sem_data; lower = mean_data - sem_data

            fig.add_trace(go.Scatter(x=np.concatenate([time_data, time_data[::-1]]), y=np.concatenate([upper, lower[::-1]]),
                                     fill='toself', fillcolor=fill_color_rgba, line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip", showlegend=(i==0), name=f'{group_name}_SEM'), row=row_num, col=1) # Show legend once per group
            fig.add_trace(go.Scatter(x=time_data, y=mean_data, line=dict(color=color_rgba), mode='lines', name=group_name, showlegend=(i==0)), row=row_num, col=1)

        ylabel = metric.replace('_', ' ').title()
        if area_col and metric == area_col: ylabel = area_label
        elif metric == 'pcc': ylabel = 'PCC (Vim/FAK)'; fig.update_yaxes(range=[-1.1, 1.1], row=row_num, col=1)
        elif metric == 'm1_coeff': ylabel = 'Manders M1'; fig.update_yaxes(range=[-0.1, 1.1], row=row_num, col=1)
        elif metric == 'm2_coeff': ylabel = 'Manders M2'; fig.update_yaxes(range=[-0.1, 1.1], row=row_num, col=1)
        elif metric == 'icq': ylabel = 'ICQ (Vim/FAK)'; fig.update_yaxes(range=[-0.6, 0.6], row=row_num, col=1)
        elif '_norm' in metric: ylabel = ylabel.replace(' Norm', ' (Norm. t0)')
        fig.update_yaxes(title_text=ylabel, row=row_num, col=1)

    fig.update_layout(title=f"Average FA Dynamics (Mean ± SEM, Length >= {min_len}){title_suffix}",
                      hovermode="x unified", height=250 * num_metrics, legend_title_text=group_col if group_col else "Trace")
    fig.update_xaxes(title_text="Time (seconds)", row=num_metrics, col=1) # Set on last row

    save_plotly(fig, outfile_html, outfile_png, logger) # Use helper


def calculate_per_track_stats(df, metrics):
    """Calculates mean value per track for given metrics."""
    logger.info("Calculating per-track averages...")
    if not metrics or df.empty: logger.warning("No metrics/data for per-track calc."); return pd.DataFrame()
    return df.groupby('global_track_id')[metrics].mean(numeric_only=True).reset_index()

def plot_interactive_distributions(avg_df, metrics, area_col, area_label, n_tracks, min_len, outfile_html, outfile_png, group_col=None, title_suffix=""):
    """Generates interactive Plotly distribution plots (violin) and static PNG."""
    logger.info(f"Generating interactive distribution plots{f' grouped by {group_col}' if group_col else ''}...")
    if avg_df.empty or not metrics: logger.warning("Skipping: No data/metrics."); return

    valid_metrics = [m for m in metrics if m in avg_df.columns and pd.notna(avg_df[m]).any()]
    if not valid_metrics: logger.warning("Skipping: No valid metrics."); return

    num_metrics = len(valid_metrics)
    n_cols = 3; n_rows = (num_metrics + n_cols - 1) // n_cols
    subplot_titles = []
    for m in valid_metrics:
        title = m.replace('_', ' ').title()
        if area_col and m == area_col: title = f"Average {area_label}"
        elif m == 'pcc': title = 'Avg PCC (Vim/FAK)'
        elif m == 'm1_coeff': title = 'Avg Manders M1'
        elif m == 'm2_coeff': title = 'Avg Manders M2'
        elif m == 'icq': title = 'Avg ICQ (Vim/FAK)'
        elif '_norm' in m: title = title.replace(' Norm', ' (Norm. t0)')
        else: title = f"Average {title}"
        subplot_titles.append(title)

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
    groups = avg_df[group_col].unique() if group_col and group_col in avg_df else [None]
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))

    r = c = 1
    for i, metric in enumerate(valid_metrics): # Use index i for legend logic
        for j, group_val in enumerate(groups):
            group_name = str(group_val) if group_val is not None else metric
            color_hex = colors[j][:3]
            color_plotly = f'rgb({int(color_hex[0]*255)}, {int(color_hex[1]*255)}, {int(color_hex[2]*255)})'
            data_to_plot = avg_df[avg_df[group_col] == group_val][metric] if group_val is not None else avg_df[metric]

            fig.add_trace(go.Violin(y=data_to_plot, name=group_name,
                                    box_visible=True, meanline_visible=True,
                                    points='all', jitter=0.2, pointpos=0,
                                    line_color=color_plotly, showlegend=(i==0 and group_col is not None)), # Show legend only if grouping
                          row=r, col=c)

        if metric == 'pcc': fig.update_yaxes(range=[-1.1, 1.1], row=r, col=c)
        elif metric == 'm1_coeff': fig.update_yaxes(range=[-0.1, 1.1], row=r, col=c)
        elif metric == 'm2_coeff': fig.update_yaxes(range=[-0.1, 1.1], row=r, col=c)
        elif metric == 'icq': fig.update_yaxes(range=[-0.6, 0.6], row=r, col=c)

        c += 1;
        if c > n_cols: c = 1; r += 1

    fig.update_layout(title=f"Distribution of Average Metrics per Track ({n_tracks} Tracks, Length >= {min_len}){title_suffix}",
                      height=350 * n_rows, legend_title_text=group_col if group_col else "Metric",
                      showlegend=(group_col is not None)) # Only show legend if grouping

    save_plotly(fig, outfile_html, outfile_png, logger) # Use helper


def plot_correlation_heatmap(avg_df, metrics, outfile):
    """Calculates and plots correlation heatmap for per-track averages."""
    logger.info("Generating correlation heatmap...")
    if avg_df.empty or len(metrics) < 2: logger.warning("Skipping heatmap: Not enough data/metrics."); return
    metrics_for_corr = [m for m in metrics if m in avg_df.columns and pd.api.types.is_numeric_dtype(avg_df[m])]
    if len(metrics_for_corr) < 2: logger.warning("Skipping heatmap: Not enough valid numeric metrics."); return
    corr_df = avg_df[metrics_for_corr].corr()
    plt.figure(figsize=(max(8, len(metrics_for_corr)*0.8), max(6, len(metrics_for_corr)*0.7)))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    plt.title('Correlation Matrix of Average Per-Track Metrics')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    try: plt.savefig(outfile, dpi=150, bbox_inches='tight'); logger.info(f"✅ Saved: {outfile}")
    except Exception as e: logger.error(f"Error saving heatmap: {e}")
    plt.close()

def plot_trajectories(df, n_samples, outfile_html, outfile_png):
    """Plots X/Y trajectories for a sample of tracks using Plotly AND saves PNG."""
    logger.info(f"Generating interactive trajectory plot for up to {n_samples} sample tracks...")
    if df.empty or 'global_track_id' not in df.columns or 'x' not in df.columns or 'y' not in df.columns: logger.warning("Skipping trajectories: Missing cols/no data."); return
    if not pd.api.types.is_numeric_dtype(df['x']) or not pd.api.types.is_numeric_dtype(df['y']): logger.warning("Warning: x/y not numeric."); return
    unique_tracks = df['global_track_id'].unique(); n_samples = min(n_samples, len(unique_tracks))
    if n_samples == 0: logger.warning("Skipping trajectories: No tracks."); return
    sampled_tracks = np.random.choice(unique_tracks, n_samples, replace=False) if len(unique_tracks) > n_samples else unique_tracks
    plot_title = f"Sample Trajectories (N={n_samples})" if len(unique_tracks) > n_samples else f"All Trajectories (N={len(unique_tracks)})"
    sampled_df = df[df['global_track_id'].isin(sampled_tracks)]
    fig = go.Figure()
    for track_id in sampled_tracks:
        track_data = sampled_df[sampled_df['global_track_id'] == track_id].sort_values('frame')
        short_name = str(os.path.basename(track_id))
        fig.add_trace(go.Scatter(x=track_data['x'], y=track_data['y'], mode='lines+markers', marker=dict(size=3), name=short_name, hoverinfo='name+x+y+text', text=[f"Frame: {f}" for f in track_data['frame']]))
    max_x = df['x'].max(skipna=True); max_y = df['y'].max(skipna=True)
    if pd.isna(max_x) or pd.isna(max_y): logger.warning("Could not determine max X/Y, using fallback."); max_x=1024; max_y=1024
    fig.update_layout(title=plot_title, xaxis_title="X (pix)", yaxis_title="Y (pix)", yaxis_autorange='reversed', xaxis_range=[0, max_x], yaxis_range=[max_y, 0], width=800, height=700, legend_title="Track ID", hovermode='closest')
    save_plotly(fig, outfile_html, outfile_png, logger) # Use helper


def define_lifetime_cohorts(df, quantiles=[0, 0.33, 0.66, 1.0], labels=['Short', 'Medium', 'Long']):
    """Adds lifetime and cohort columns based on track length quantiles."""
    logger.info("Defining lifetime cohorts...")
    if 'track_length' not in df.columns: logger.error("'track_length' column required."); return df
    if len(labels) != len(quantiles) - 1: logger.warning("Labels/Quantiles mismatch."); labels = [f"Cohort_{i+1}" for i in range(len(quantiles)-1)]
    unique_lengths = df[['global_track_id', 'track_length']].drop_duplicates()['track_length']
    try:
        cohort_bins = pd.qcut(unique_lengths, q=quantiles, labels=labels, retbins=True, duplicates='drop')[1]
        logger.info(f"  Cohort bins (frames): {cohort_bins}")
        df['lifetime_cohort'] = pd.cut(df['track_length'], bins=cohort_bins, labels=labels, right=True, include_lowest=True)
        logger.info("  Assigned tracks to cohorts:\n" + str(df['lifetime_cohort'].value_counts()))
    except Exception as e: logger.error(f"Cohort creation failed: {e}. Skipping."); df['lifetime_cohort'] = 'All'
    return df

def perform_kmeans_clustering(avg_df, metrics_for_clustering, k=None, elbow_k_max=10, outfile_elbow=None):
    """Performs K-Means clustering on average track stats."""
    logger.info("Performing K-Means clustering on average track metrics...")
    if not SKLEARN_AVAILABLE: logger.warning("Skipping clustering: scikit-learn not installed."); return avg_df, None
    if avg_df.empty or not metrics_for_clustering: logger.warning("Skipping clustering: No data/metrics."); return avg_df, None

    # Ensure metrics exist
    metrics_exist = [m for m in metrics_for_clustering if m in avg_df.columns]
    if len(metrics_exist) < 1: logger.warning("Skipping clustering: No valid metrics found."); return avg_df, None
    if len(metrics_exist) < len(metrics_for_clustering): logger.warning(f"Clustering using subset: {metrics_exist}")

    cluster_features = avg_df[metrics_exist].copy()
    imputer = SimpleImputer(strategy='median'); features_imputed = imputer.fit_transform(cluster_features)
    if np.isnan(features_imputed).any(): logger.warning("NaNs remain after imputation.")
    scaler = StandardScaler(); features_scaled = scaler.fit_transform(features_imputed)

    if k is None: # Elbow Method
        logger.info(f"  Calculating Elbow method WCSS for K=1 to {elbow_k_max}...")
        wcss = [KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(features_scaled).inertia_ for i in range(1, elbow_k_max + 1)]
        plt.figure(figsize=(8, 5)); plt.plot(range(1, elbow_k_max + 1), wcss, marker='o'); plt.title('Elbow Method'); plt.xlabel('K'); plt.ylabel('WCSS')
        if outfile_elbow:
            try: plt.savefig(outfile_elbow, dpi=150, bbox_inches='tight'); logger.info(f"✅ Elbow plot saved: {outfile_elbow}")
            except Exception as e: logger.error(f"Error saving Elbow plot: {e}")
        plt.close()
        logger.info(f"  -> Elbow plot saved. Re-run script with chosen 'n_clusters_kmeans'.")
        return avg_df, None

    # Perform K-Means
    logger.info(f"  Running K-Means with K={k}...")
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    avg_df['cluster'] = kmeans.fit_predict(features_scaled)
    logger.info("  Assigned tracks to clusters:\n" + str(avg_df['cluster'].value_counts().sort_index()))
    return avg_df, features_scaled


# --- Main Execution Block ---
if __name__ == "__main__":
    logger = setup_logging() # Setup logging

    # 1. Get Paths
    csv_directory, output_directory = get_paths()

    # 2. Find Files
    csv_files_to_process = find_csv_files(csv_directory)

    # 3. Load and Combine
    combined_data = load_and_combine_data(csv_files_to_process)

    if not combined_data.empty:
        # 4. Filter Tracks
        filtered_data, num_filt_tracks = filter_tracks_by_length(combined_data, min_track_length_filter)

        if not filtered_data.empty and num_filt_tracks > 0:
            # 5. Prepare Data (Add time, area_um2)
            prepared_data, area_col, area_label_str = prepare_data(filtered_data, assumed_frame_interval, pixel_size_um)

            # 6. Normalize Intensities
            if 'fak_intensity_mean' in prepared_data.columns: prepared_data = normalize_intensity_by_first_frame(prepared_data, 'fak_intensity_mean', 'fak_intensity_mean_norm')
            if 'vimentin_intensity_mean' in prepared_data.columns: prepared_data = normalize_intensity_by_first_frame(prepared_data, 'vimentin_intensity_mean', 'vimentin_intensity_mean_norm')
            if 'fak_intensity_integrated' in prepared_data.columns: prepared_data = normalize_intensity_by_first_frame(prepared_data, 'fak_intensity_integrated', 'fak_intensity_integrated_norm')
            if 'vimentin_intensity_integrated' in prepared_data.columns: prepared_data = normalize_intensity_by_first_frame(prepared_data, 'vimentin_intensity_integrated', 'vimentin_intensity_integrated_norm')

            # 7. Define Final Metrics List (including normalized)
            metrics_all = metrics_to_analyze.copy()
            if 'fak_intensity_mean_norm' in prepared_data.columns: metrics_all.append('fak_intensity_mean_norm')
            if 'vimentin_intensity_mean_norm' in prepared_data.columns: metrics_all.append('vimentin_intensity_mean_norm')
            if 'fak_intensity_integrated_norm' in prepared_data.columns: metrics_all.append('fak_intensity_integrated_norm')
            if 'vimentin_intensity_integrated_norm' in prepared_data.columns: metrics_all.append('vimentin_intensity_integrated_norm')
            final_metrics = get_final_metrics(prepared_data, metrics_all, area_col)

            if final_metrics:
                # 8. Convert relevant columns to numeric
                prepared_data = convert_and_clean_numeric(prepared_data, final_metrics + ['x', 'y'])
                num_filt_tracks = prepared_data['global_track_id'].nunique() # Update count after cleaning
                if num_filt_tracks == 0: logger.error("All tracks removed after NaN drop. Exiting."); sys.exit()
                logger.info(f"Tracks remaining after NaN drop: {num_filt_tracks}")
                final_metrics = [m for m in final_metrics if m in prepared_data.columns] # Update metrics list


                # --- Overall Population Analysis ---
                # 9. Calculate Overall Time Course Stats & Plot
                time_stats_df = calculate_time_course_stats(prepared_data, final_metrics)
                ts_html = os.path.join(output_directory, f"Combined_TimeSeries_Interactive_Filt{min_track_length_filter}.html")
                ts_png = os.path.join(output_directory, f"Combined_TimeSeries_Static_Filt{min_track_length_filter}.png")
                # <<< Removed logger=logger from call >>>
                plot_interactive_time_courses(time_stats_df, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, ts_html, ts_png)
                # 10. Calculate Overall Per-Track Stats & Plot Distributions
                avg_track_stats_df = calculate_per_track_stats(prepared_data, final_metrics)
                dist_html = os.path.join(output_directory, f"Combined_Distribution_Interactive_Filt{min_track_length_filter}.html")
                dist_png = os.path.join(output_directory, f"Combined_Distribution_Static_Filt{min_track_length_filter}.png")
                # <<< Removed logger=logger from call >>>
                plot_interactive_distributions(avg_track_stats_df, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, dist_html, dist_png)
                # 11. Plot Overall Correlation Heatmap
                heatmap_file = os.path.join(output_directory, f"Combined_Correlation_Heatmap_Filt{min_track_length_filter}.png")
                # <<< Removed logger=logger from call (not needed) >>>
                plot_correlation_heatmap(avg_track_stats_df, final_metrics, heatmap_file)
                # 12. Plot Sample Trajectories
                traj_html = os.path.join(output_directory, f"Combined_Trajectories_Sample_Interactive_Filt{min_track_length_filter}.html")
                traj_png = os.path.join(output_directory, f"Combined_Trajectories_Sample_Static_Filt{min_track_length_filter}.png")
                # <<< Removed logger=logger from call >>>
                plot_trajectories(prepared_data, num_trajectory_samples, traj_html, traj_png)


                # --- Lifetime Cohort Analysis ---
                prepared_data_cohorts = define_lifetime_cohorts(prepared_data.copy()) # Use copy to avoid modifying original df
                if 'lifetime_cohort' in prepared_data_cohorts.columns and prepared_data_cohorts['lifetime_cohort'].nunique() > 1:
                    cohort_time_stats = calculate_time_course_stats(prepared_data_cohorts, final_metrics, group_col='lifetime_cohort')
                    cohort_ts_html = os.path.join(output_directory, f"Combined_TimeSeries_ByCohort_Filt{min_track_length_filter}.html")
                    cohort_ts_png = os.path.join(output_directory, f"Combined_TimeSeries_ByCohort_Filt{min_track_length_filter}.png")
                    # <<< Removed logger=logger from call >>>
                    plot_interactive_time_courses(cohort_time_stats, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, cohort_ts_html, cohort_ts_png, group_col='lifetime_cohort', title_suffix=" by Lifetime Cohort")
                else:
                    logger.info("Skipping cohort time course plots (only one cohort or column missing).")


                # --- Clustering Analysis ---
                if SKLEARN_AVAILABLE:
                    cluster_feature_cols = [m for m in [area_col, 'fak_intensity_mean_norm', 'vimentin_intensity_mean_norm', 'pcc', 'icq'] if m and m in avg_track_stats_df.columns]
                    if len(cluster_feature_cols) > 1:
                        elbow_plot_file = os.path.join(output_directory, f"Clustering_ElbowPlot_Filt{min_track_length_filter}.png")
                        avg_track_stats_clustered, _ = perform_kmeans_clustering(avg_track_stats_df.copy(), cluster_feature_cols, k=n_clusters_kmeans, outfile_elbow=elbow_plot_file)

                        if n_clusters_kmeans is not None and 'cluster' in avg_track_stats_clustered.columns:
                            prepared_data_clustered = pd.merge(prepared_data, avg_track_stats_clustered[['global_track_id', 'cluster']], on='global_track_id', how='left')
                            # Plot time courses per cluster
                            cluster_time_stats = calculate_time_course_stats(prepared_data_clustered, final_metrics, group_col='cluster')
                            clus_ts_html = os.path.join(output_directory, f"Combined_TimeSeries_ByCluster_Filt{min_track_length_filter}.html")
                            clus_ts_png = os.path.join(output_directory, f"Combined_TimeSeries_ByCluster_Filt{min_track_length_filter}.png")
                            # <<< Removed logger=logger from call >>>
                            plot_interactive_time_courses(cluster_time_stats, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, clus_ts_html, clus_ts_png, group_col='cluster', title_suffix=f" by Cluster (K={n_clusters_kmeans})")
                            # Plot distributions per cluster
                            clus_dist_html = os.path.join(output_directory, f"Combined_Distribution_ByCluster_Filt{min_track_length_filter}.html")
                            clus_dist_png = os.path.join(output_directory, f"Combined_Distribution_ByCluster_Filt{min_track_length_filter}.png")
                            # <<< Removed logger=logger from call >>>
                            plot_interactive_distributions(avg_track_stats_clustered, final_metrics, area_col, area_label_str, num_filt_tracks, min_track_length_filter, clus_dist_html, clus_dist_png, group_col='cluster', title_suffix=f" by Cluster (K={n_clusters_kmeans})")
                        elif n_clusters_kmeans is None:
                            logger.info("Run script again with 'n_clusters_kmeans' set after viewing Elbow plot.")
                    else:
                        logger.warning("Skipping clustering: Not enough valid features available.")
                else:
                    logger.warning("Skipping clustering: scikit-learn library not installed.")


                # --- Final Data Output ---
                logger.info("\nFirst 5 rows of final processed data:")
                print(prepared_data.head()) # Show head of the data

            else:
                 logger.error("No valid metrics left to analyze after checking columns.")
        # No explicit else needed here, message printed in filter_tracks function

    else:
        logger.error("Exiting: No data was loaded.")

    logger.info("Script finished.")
