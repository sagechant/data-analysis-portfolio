import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import glob
import tkinter as tk
from tkinter import filedialog
import logging
import time
import traceback

# --- Clustering & Scaling Imports ---
SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from scipy.stats import linregress
    from scipy.spatial import distance
    SKLEARN_AVAILABLE = True
except ImportError:
    print("CRITICAL WARNING: scikit-learn, scipy, or pandas not found.")
    print("                  Please install them using: pip install -U scikit-learn scipy pandas")

# --- Configuration ---
assumed_frame_interval = 5.0  # seconds
pixel_size_um = 0.08          # micrometers per pixel
min_track_length_filter = 20  # frames
num_trajectory_samples = 50   # Number of sample trajectories to plot

# --- Advanced Analysis Configuration ---
CLUSTERING_ALGORITHM = 'GMM'
N_CLUSTERS_KMEANS = 3
N_COMPONENTS_GMM = 3

# --- Features for PCA and Clustering ---
# Added 'msd_alpha' and 'diffusion_coeff' to analysis
FEATURES_FOR_ANALYSIS = [
    'area_um2', 'fak_intensity_mean_norm', 'vimentin_intensity_mean_norm',
    'velocity', 'directionality', 'persistence', 'msd_alpha', 'diffusion_coeff'
]
METRICS_TO_ANALYZE_RAW = [
    'vimentin_intensity_mean', 'vimentin_intensity_integrated',
    'fak_intensity_mean', 'fak_intensity_integrated',
    'pcc', 'm1_coeff', 'm2_coeff', 'icq', 'area_pixels'
]

# --- Helper Functions ---
def setup_logging(level=logging.INFO):
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level, stream=sys.stdout
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()

def get_paths_interactive():
    """
    Opens a GUI dialog to select input and output directories.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    logger.info("Please select the directory containing the analysis CSV or XLSX files...")
    csv_dir = filedialog.askdirectory(title="Select Directory Containing Data Files")
    if not csv_dir:
        sys.exit(1)
    logger.info(f"Selected input directory: {csv_dir}")
    logger.info("Please select the directory where output plots and data should be saved...")
    output_plot_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_plot_dir:
        sys.exit(1)
    os.makedirs(output_plot_dir, exist_ok=True)
    root.destroy()
    return csv_dir, output_plot_dir

def find_data_files(data_dir):
    """
    Finds all CSV and XLSX files in the directory matching specific patterns.
    """
    logger.info(f"Searching for data files in: {data_dir}")
    patterns = ["*_D3D.csv", "*_Analysis_tracks_filtered.csv", "*.xlsx"]
    data_files_found = []
    for pattern in patterns:
        data_files_found.extend(glob.glob(os.path.join(data_dir, pattern)))
    data_files_found = [f for f in data_files_found if not os.path.basename(f).startswith('~$')]
    data_files_found = sorted(list(set(data_files_found)))
    if not data_files_found:
        logger.error(f"No CSV or XLSX files matching patterns found in '{data_dir}'. Exiting.")
        sys.exit(1)
    return data_files_found

def load_and_combine_data(data_files):
    """
    Loads multiple files, standardizes column names, and combines them into one DataFrame.
    """
    all_tracks_list = []
    logger.info("Loading and combining data files...")
    
    column_name_map = {
        'particle': ['particle', 'track #', 'track_id', 'track id', 'id'],
        'frame': ['frame', 'slice', 'time'],
        'x': ['x', 'x coordinate', 'x-coordinate', 'position x'],
        'y': ['y', 'y coordinate', 'y-coordinate', 'position y'],
    }

    for i, f_path in enumerate(data_files):
        f_name = os.path.basename(f_path)
        try:
            df_single = None
            if f_path.lower().endswith('.csv'):
                # Heuristic for delimiters
                with open(f_path, 'r') as temp_f:
                    header = temp_f.readline()
                    delimiter = ';' if ';' in header else ','
                decimal_separator = ',' if delimiter == ';' else '.'
                df_single = pd.read_csv(f_path, delimiter=delimiter, decimal=decimal_separator)
            elif f_path.lower().endswith('.xlsx'):
                df_single = pd.read_excel(f_path, sheet_name=0)
            
            if df_single is None: continue

            df_single.columns = [col.strip().lower() for col in df_single.columns]
            
            for standard_name, possible_names in column_name_map.items():
                for possible_name in possible_names:
                    if possible_name in df_single.columns:
                        df_single.rename(columns={possible_name: standard_name}, inplace=True)
                        break 

            if 'particle' not in df_single.columns or 'frame' not in df_single.columns:
                continue

            df_single['source_file'] = f_name
            df_single['particle'] = df_single['particle'].astype(int)
            df_single['global_track_id'] = f"{f_name}_" + df_single['particle'].astype(str)
            all_tracks_list.append(df_single)
        except Exception as e:
            logger.error(f"  Error reading file {f_name}: {e}. Skipping.")
    if not all_tracks_list:
        logger.error("No valid data loaded. Exiting.")
        sys.exit(1)
    combined_df = pd.concat(all_tracks_list, ignore_index=True)
    return combined_df

def interpolate_missing_data(df):
    """
    SCIENTIFIC FIX: Linearly interpolate missing data within tracks 
    instead of using global median imputation. This respects the time-series nature of the data.
    """
    logger.info("Interpolating missing values within tracks (Linear Method)...")
    # Sort to ensure time order
    df = df.sort_values(by=['global_track_id', 'frame'])
    
    # We only interpolate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Apply interpolation per track
    # limit_direction='both' handles NaN at start/end if possible
    df[numeric_cols] = df.groupby('global_track_id')[numeric_cols].transform(
        lambda group: group.interpolate(method='linear', limit_direction='forward', axis=0)
    )
    return df

def filter_tracks_by_length(df, min_len):
    if 'global_track_id' not in df.columns: return pd.DataFrame(), 0
    df['track_length_frames'] = df.groupby('global_track_id')['frame'].transform('count')
    filtered_df = df[df['track_length_frames'] >= min_len].copy()
    return filtered_df, filtered_df['global_track_id'].nunique()

def prepare_data_cols(df, interval, px_size_um_val):
    df['time_sec'] = df['frame'] * interval
    area_col_name_actual = 'area_pixels'
    if px_size_um_val and 'area_pixels' in df.columns:
        df['area_pixels'] = pd.to_numeric(df['area_pixels'], errors='coerce')
        df['area_um2'] = df['area_pixels'] * (px_size_um_val**2)
        area_col_name_actual = 'area_um2'
    return df, area_col_name_actual, f'Area (µm²)' if area_col_name_actual == 'area_um2' else 'Area (pixels)'

def normalize_intensity_robust(df, intensity_col, norm_col_name, n_frames=3):
    """
    SCIENTIFIC FIX: Robust normalization using the mean of the first N frames
    instead of just the single first frame. This avoids 'shot noise' spikes.
    """
    logger.info(f"Normalizing '{intensity_col}' -> '{norm_col_name}' (Baseline: Avg of first {n_frames} frames)")
    if intensity_col not in df.columns: return df
    df[intensity_col] = pd.to_numeric(df[intensity_col], errors='coerce')
    
    # Calculate baseline per track
    def get_baseline(x):
        # Take the first n_frames and calculate mean, ignoring NaNs
        return x.iloc[:n_frames].mean()

    df_sorted = df.sort_values(['global_track_id', 'frame'])
    baselines = df_sorted.groupby('global_track_id')[intensity_col].transform(get_baseline)
    
    # Avoid division by zero
    baselines = baselines.replace(0, np.nan)
    df[norm_col_name] = df[intensity_col] / baselines
    return df

def calculate_msd_alpha(track_df, x_col='x', y_col='y', dt_sec=5.0):
    """
    SCIENTIFIC FIX: Calculates Mean Squared Displacement (MSD) and estimates
    the anomalous diffusion exponent (alpha).
    MSD(tau) = <(r(t+tau) - r(t))^2> ~ 4*D*tau^alpha
    """
    coords = track_df[[x_col, y_col]].values
    n = len(coords)
    # We only look at the first 25% of the track for robust fitting (standard practice)
    max_lag = max(2, int(n * 0.25)) 
    
    lags = range(1, max_lag + 1)
    msds = []
    
    for lag in lags:
        diffs = coords[lag:] - coords[:-lag]
        sq_dists = np.sum(diffs**2, axis=1)
        msds.append(np.mean(sq_dists))
    
    # Convert pixels to um^2 for MSD magnitude (diffusion coeff)
    msds_um = np.array(msds) * (pixel_size_um**2)
    time_lags = np.array(lags) * dt_sec
    
    # Log-Log Fit for Alpha
    # log(MSD) = alpha * log(tau) + log(C)
    if len(msds_um) < 2:
        return np.nan, np.nan
        
    try:
        slope, intercept, _, _, _ = linregress(np.log(time_lags), np.log(msds_um))
        alpha = slope
        # D (Diffusion Coeff) approx from intercept.
        d_coeff = np.exp(intercept) 
        return alpha, d_coeff
    except:
        return np.nan, np.nan

def calculate_dynamic_and_motility_metrics(df, id_col='global_track_id', time_col='time_sec'):
    logger.info("Calculating advanced per-track metrics (MSD, smooth velocity, persistence)...")
    if df.empty: return pd.DataFrame()
    all_track_stats = []
    
    # Sort data
    df_sorted = df.sort_values(by=[id_col, time_col]).copy()
    
    # SCIENTIFIC FIX: Smoothing trajectory to remove jitter before derivative
    # Rolling window of 3, centered.
    df_sorted['x_smooth'] = df_sorted.groupby(id_col)['x'].transform(lambda x: x.rolling(window=3, center=True, min_periods=1).mean())
    df_sorted['y_smooth'] = df_sorted.groupby(id_col)['y'].transform(lambda y: y.rolling(window=3, center=True, min_periods=1).mean())

    # Calculate instantaneous derivatives on SMOOTHED data
    df_sorted['dx'] = df_sorted.groupby(id_col)['x_smooth'].diff()
    df_sorted['dy'] = df_sorted.groupby(id_col)['y_smooth'].diff()
    df_sorted['dt'] = df_sorted.groupby(id_col)[time_col].diff()
    
    # Displacement & Velocity
    df_sorted['displacement'] = np.sqrt(df_sorted['dx']**2 + df_sorted['dy']**2)
    # Use smoothed displacement for velocity
    df_sorted['velocity'] = (df_sorted['displacement'] * pixel_size_um) / df_sorted['dt'] 
    
    # Persistence (Cosine Theta)
    # Note: We calculate this on smoothed vectors to avoid noise flipping the angle
    dot_product = (df_sorted['dx'].shift(1) * df_sorted['dx'] + df_sorted['dy'].shift(1) * df_sorted['dy'])
    magnitude_prod = np.sqrt(df_sorted['dx'].shift(1)**2 + df_sorted['dy'].shift(1)**2) * df_sorted['displacement']
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = dot_product / magnitude_prod
    df_sorted['persistence'] = np.clip(cos_theta, -1.0, 1.0)

    # Per-Track Aggregation
    for track_id, track_df in df_sorted.groupby(id_col):
        if len(track_df) < 5: continue # Need minimal length for MSD
        
        stats = {'global_track_id': track_id}
        
        # 1. Existing aggregations
        for col in ['fak_intensity_mean_norm', 'vimentin_intensity_mean_norm', 'area_um2', 'pcc', 'icq']:
            if col in track_df.columns: stats[f'{col}_std_dev'] = track_df[col].std()
        
        for col in ['area_um2', 'fak_intensity_mean_norm']:
            if col in track_df.columns:
                clean_df = track_df[[time_col, col]].dropna()
                if len(clean_df) > 1:
                    stats[f'{col}_slope'] = linregress(clean_df[time_col], clean_df[col]).slope
        
        # 2. Motility Metrics (Directionality)
        # Use raw x,y for net displacement to avoid boundary smoothing artifacts
        start_pos, end_pos = track_df.iloc[0], track_df.iloc[-1]
        net_displacement = np.sqrt((end_pos['x'] - start_pos['x'])**2 + (end_pos['y'] - start_pos['y'])**2)
        total_distance = track_df['displacement'].sum() # Path length
        stats['directionality'] = net_displacement / total_distance if total_distance > 0 else 0
        stats['persistence_avg'] = track_df['persistence'].mean()
        stats['velocity_avg'] = track_df['velocity'].mean()
        
        # 3. MSD Analysis (New)
        # Using Smoothed coords for MSD is debatable. Usually Raw is better for MSD 
        # to detect sub-pixel diffusion, but if jitter is high, Smooth helps. 
        # Standard physics uses Raw. Let's use Raw 'x' and 'y' for MSD to capture true alpha.
        alpha, d_coeff = calculate_msd_alpha(track_df, 'x', 'y', assumed_frame_interval)
        stats['msd_alpha'] = alpha
        stats['diffusion_coeff'] = d_coeff
        
        all_track_stats.append(stats)
        
    return pd.DataFrame(all_track_stats)

def calculate_per_track_stats(df, metrics, id_col='global_track_id'):
    logger.info("Calculating per-track average statistics...")
    if not metrics or df.empty: return pd.DataFrame()
    numeric_metrics_for_avg = [m for m in metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]
    avg_stats = df.groupby(id_col)[numeric_metrics_for_avg].mean().reset_index()
    avg_stats.columns = [id_col] + [f"{col}_avg" for col in numeric_metrics_for_avg]
    
    # Preserve metadata
    cols_to_carry = ['track_length_frames', 'source_file']
    existing_cols_to_carry = [c for c in cols_to_carry if c in df.columns]
    if existing_cols_to_carry:
        first_occurrence_stats = df.groupby(id_col)[existing_cols_to_carry].first().reset_index()
        avg_stats = pd.merge(avg_stats, first_occurrence_stats, on=id_col, how='left')
    return avg_stats

def save_plotly_figure(fig, html_path, png_path):
    try:
        pio.write_html(fig, html_path, auto_open=False)
    except Exception as e: logger.error(f"Error saving HTML: {e}")
    try:
        # Static export requires kaleido or orca
        pio.write_image(fig, png_path, scale=2)
    except Exception:
        pass # Silently fail on static image if engine missing

def calculate_time_course_stats(df, metrics, group_col=None):
    if not metrics or df.empty: return pd.DataFrame()
    agg_funcs = ['mean', 'sem']
    grouping_cols = ['time_sec']
    if group_col and group_col in df.columns: grouping_cols.append(group_col)
    time_stats = df.groupby(grouping_cols)[[m for m in metrics if m in df.columns]].agg(agg_funcs)
    time_stats.columns = ['_'.join(col).strip() for col in time_stats.columns.values]
    return time_stats.reset_index()

def plot_interactive_time_courses(stats_df, metrics_to_plot, area_col_actual, area_label_str, n_tracks_val, min_len_val, html_path, png_path, group_col=None, title_suffix=""):
    if stats_df.empty or not metrics_to_plot: return
    valid_metrics = [m for m in metrics_to_plot if f"{m}_mean" in stats_df.columns]
    
    fig = make_subplots(rows=len(valid_metrics), cols=1, shared_xaxes=True, 
                        subplot_titles=[m.replace('_',' ').title() for m in valid_metrics])
    
    groups = sorted(stats_df[group_col].unique()) if group_col else [None]
    colors = px.colors.qualitative.Plotly
    
    for i, metric in enumerate(valid_metrics):
        for j, group_val in enumerate(groups):
            group_stats = stats_df[stats_df[group_col] == group_val] if group_val else stats_df
            mean = group_stats[f"{metric}_mean"]
            sem = group_stats[f"{metric}_sem"].fillna(0)
            time_ = group_stats['time_sec']
            color = colors[j % len(colors)]
            name = str(group_val) if group_val else 'All'
            
            # Ribbon (SEM)
            fig.add_trace(go.Scatter(
                x=pd.concat([time_, time_[::-1]]),
                y=pd.concat([mean+sem, (mean-sem)[::-1]]),
                fill='toself', fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}",
                line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'
            ), row=i+1, col=1)
            
            # Line (Mean)
            fig.add_trace(go.Scatter(
                x=time_, y=mean, line=dict(color=color), name=name,
                showlegend=(i==0)
            ), row=i+1, col=1)
            
    fig.update_layout(height=250*len(valid_metrics), title=f"Time Courses {title_suffix}")
    save_plotly_figure(fig, html_path, png_path)

def run_pca_and_visualize(features_df, output_dir, prefix, n_components=3, cluster_labels=None):
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(features_df)
    pca_df = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_components)])
    if cluster_labels is not None: pca_df['cluster'] = cluster_labels
    
    var_exp = pca.explained_variance_ratio_
    
    # 2D Plot
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='cluster' if 'cluster' in pca_df.columns else None,
                     title=f'PCA (PC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%})')
    save_plotly_figure(fig, os.path.join(output_dir, f"{prefix}_PCA_2D.html"), os.path.join(output_dir, f"{prefix}_PCA_2D.png"))
    return pca_df

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Migration & Focal Adhesion Pipeline Started ---")
    csv_directory, output_directory = get_paths_interactive()
    output_prefix = os.path.basename(csv_directory)
    data_files = find_data_files(csv_directory)
    
    # 1. Load Data
    combined_data = load_and_combine_data(data_files)
    
    # 2. Interpolate Missing Data (SCIENTIFIC FIX)
    combined_data = interpolate_missing_data(combined_data)

    # 3. Filter Length
    filtered_data, _ = filter_tracks_by_length(combined_data, min_track_length_filter)
    
    # 4. Prepare Cols & Normalize (SCIENTIFIC FIX: Robust Norm)
    prepared_data, area_col, area_lbl = prepare_data_cols(filtered_data, assumed_frame_interval, pixel_size_um)
    for metric in ['fak_intensity_mean', 'vimentin_intensity_mean']:
        prepared_data = normalize_intensity_robust(prepared_data, metric, f"{metric}_norm")
    
    # 5. Calculate Features (SCIENTIFIC FIX: Smoothing + MSD)
    base_metrics = [m + '_norm' for m in METRICS_TO_ANALYZE_RAW if m in prepared_data.columns]
    if area_col: base_metrics.append(area_col)
    
    avg_stats = calculate_per_track_stats(prepared_data, base_metrics)
    # This function now performs Smoothing internally before derivative calc
    dynamic_stats = calculate_dynamic_and_motility_metrics(prepared_data)
    
    final_features = pd.merge(avg_stats, dynamic_stats, on='global_track_id', how='inner')
    
    # 6. Clustering (PCA / GMM)
    if SKLEARN_AVAILABLE and not final_features.empty:
        # Build feature matrix
        feat_cols = []
        for base in FEATURES_FOR_ANALYSIS:
            # Check for various aggregations in columns
            possible = [f"{base}", f"{base}_avg", f"{base}_slope"]
            # Specifically check for dynamic metrics that don't have _avg
            if base in final_features.columns:
                feat_cols.append(base)
            else:
                for p in possible: 
                    if p in final_features.columns: 
                        feat_cols.append(p)
                        break
        
        # Clean infinite/NaN from MSD calc
        feat_cols = list(set(feat_cols)) # Remove duplicates
        X = final_features[feat_cols].replace([np.inf, -np.inf], np.nan).dropna()
        
        if not X.empty:
            logger.info(f"Clustering on features: {feat_cols}")
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # GMM Clustering
            gmm = GaussianMixture(n_components=N_COMPONENTS_GMM, random_state=42)
            labels = gmm.fit_predict(X_scaled)
            
            # Re-attach labels
            final_features.loc[X.index, 'cluster'] = labels
            
            # Visuals
            run_pca_and_visualize(X_scaled, output_directory, output_prefix, cluster_labels=labels)
            
            # Time Courses by Cluster
            prepared_clustered = pd.merge(prepared_data, final_features[['global_track_id', 'cluster']], on='global_track_id')
            ts_stats = calculate_time_course_stats(prepared_clustered, base_metrics, group_col='cluster')
            plot_interactive_time_courses(ts_stats, base_metrics, area_col, area_lbl, 0, 0, 
                                          os.path.join(output_directory, "Cluster_TimeCourses.html"), 
                                          os.path.join(output_directory, "Cluster_TimeCourses.png"), 
                                          group_col='cluster')

    # Save Results
    final_features.to_csv(os.path.join(output_directory, f"{output_prefix}_Results.csv"), index=False)
    logger.info("Done. Check output directory for results.")