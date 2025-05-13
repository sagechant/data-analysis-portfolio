import os
import sys
import glob
import logging

import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import seaborn as sns
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

# --- Logging Setup ---

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level
    )
    return logging.getLogger(__name__)

# --- Interactive Path Selection ---

def get_paths():
    root = tk.Tk(); root.withdraw()
    print("Select directory containing input files (.csv/.xls/.xlsx)...")
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    if not input_dir:
        print("No input directory selected. Exiting.")
        sys.exit(1)
    print(f"Input directory: {input_dir}")

    print("Select directory to save output plots...")
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected. Exiting.")
        sys.exit(1)
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    return input_dir, output_dir

# --- File Discovery ---

def find_input_files(input_dir):
    patterns = ["*.csv", "*.xls", "*.xlsx"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat)))
    files = [f for f in files if not os.path.basename(f).startswith('~$')]
    if not files:
        print(f"No input files found in {input_dir}. Exiting.")
        sys.exit(1)
    files = sorted(files)
    print(f"Found {len(files)} files to process.")
    return files

# --- Data Loading ---

def load_and_combine(files, logger):
    dfs = []
    for path in files:
        name = os.path.basename(path)
        logger.info(f"Loading {name}")
        try:
            ext = os.path.splitext(name)[1].lower()
            if ext in ['.xls', '.xlsx']:
                df = pd.read_excel(path)
            else:
                sample = open(path, 'r', errors='ignore').read(2048)
                delim = ';' if ';' in sample and sample.count(';')>sample.count(',') else ','
                df = pd.read_csv(path, delimiter=delim)
            if 'particle' not in df.columns or 'frame' not in df.columns:
                logger.warning(f"Skipping {name}: missing 'particle' or 'frame'.")
                continue
            df['source_file'] = name
            df['particle'] = df['particle'].astype(int)
            df['global_track_id'] = name + '_' + df['particle'].astype(str)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {name}: {e}")
    if not dfs:
        logger.error("No valid data loaded. Exiting.")
        sys.exit(1)
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(dfs)} files.")
    return combined

# --- Filtering ---

def filter_tracks(df, min_len, logger):
    counts = df.groupby('global_track_id')['frame'].transform('count')
    filtered = df[counts >= min_len].copy()
    kept = filtered['global_track_id'].nunique()
    total = df['global_track_id'].nunique()
    logger.info(f"Tracks >= {min_len} frames: {kept}/{total}")
    return filtered, kept

# --- Data Preparation ---

def prepare_data(df, frame_interval, pixel_size, logger):
    df['time_sec'] = df['frame'] * frame_interval
    area_col = None
    if pixel_size and 'area_pixels' in df.columns:
        df['area_um2'] = pd.to_numeric(df['area_pixels'], errors='coerce') * pixel_size**2
        area_col = 'area_um2'
        logger.info("Calculated area_um2.")
    elif 'area_pixels' in df.columns:
        area_col = 'area_pixels'
    return df, area_col

# --- Metrics Selection ---

METRICS = [
    'vimentin_intensity_mean','vimentin_intensity_integrated',
    'fak_intensity_mean','fak_intensity_integrated',
    'pcc','m1_coeff','m2_coeff','icq','area_pixels'
]

def select_metrics(df, area_col, logger):
    metrics = [m for m in METRICS if m in df.columns]
    if area_col and area_col in df.columns and area_col not in metrics:
        metrics.append(area_col)
    logger.info(f"Metrics: {metrics}")
    return metrics

# --- Numeric Conversion ---

def convert_numeric(df, cols, logger):
    before = len(df)
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    essential = [c for c in ['x','y'] if c in cols]
    if 'area_um2' in cols:
        essential.append('area_um2')
    elif 'area_pixels' in cols:
        essential.append('area_pixels')
    df.dropna(subset=essential, inplace=True)
    after = len(df)
    logger.info(f"Dropped {before-after} rows with NaNs in {essential}")
    return df

# --- Plot Saving Helper ---

def save_plotly(fig, html_path, png_path, logger):
    try:
        pio.write_html(fig, html_path, auto_open=False)
        logger.info(f"Saved HTML: {html_path}")
    except Exception as e:
        logger.error(f"HTML save error: {e}")
    try:
        pio.write_image(fig, png_path, scale=2)
        logger.info(f"Saved PNG: {png_path}")
    except Exception:
        logger.warning("Install kaleido for PNG export.")

# --- Analysis & Plotting Functions ---

def calculate_time_course(df, metrics):
    stats = df.groupby('time_sec')[metrics].agg(['mean','sem'])
    stats.columns = [f"{m}_{stat}" for m,stat in stats.columns]
    return stats.reset_index()


def plot_time_course(stats_df, metrics, area_col, area_label, n_tracks, min_len,
                     html_out, png_out, save_fn, logger):
    num = len(metrics)
    fig = make_subplots(rows=num, cols=1, shared_xaxes=True,
                        subplot_titles=[m.replace('_',' ').title() for m in metrics])
    for i, m in enumerate(metrics, start=1):
        mean_col, sem_col = f"{m}_mean", f"{m}_sem"
        if mean_col in stats_df and sem_col in stats_df:
            t = stats_df['time_sec']; mu = stats_df[mean_col]; se = stats_df[sem_col].fillna(0)
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([mu+se, (mu-se)[::-1]]),
                fill='toself', showlegend=False, hoverinfo='skip'
            ), row=i, col=1)
            fig.add_trace(go.Scatter(x=t, y=mu, mode='lines', name='Mean'), row=i, col=1)
            ylabel = m.replace('_',' ').title()
            if m==area_col: ylabel=area_label
            fig.update_yaxes(title_text=ylabel, row=i, col=1)
    fig.update_layout(title=f"Time Course ({n_tracks} tracks ≥{min_len} frames)",
                      height=300*num, hovermode='x unified')
    save_fn(fig, html_out, png_out, logger)


def calculate_track_averages(df, metrics):
    return df.groupby('global_track_id')[metrics].mean().reset_index()


def plot_distributions(avg_df, metrics, area_col, area_label,
                       n_tracks, min_len, html_out, png_out, save_fn, logger):
    valid = [m for m in metrics if m in avg_df]
    if not valid:
        logger.warning("No metrics for distribution plots.")
        return
    n_cols=3; n_rows=(len(valid)+n_cols-1)//n_cols
    fig=make_subplots(rows=n_rows, cols=n_cols,
                      subplot_titles=[m.title() for m in valid])
    r=c=1
    for m in valid:
        fig.add_trace(go.Violin(y=avg_df[m], box_visible=True, meanline_visible=True), row=r, col=c)
        if c==n_cols: c=1; r+=1
        else: c+=1
    fig.update_layout(title=f"Distribution ({n_tracks} tracks)", height=350*n_rows)
    save_fn(fig, html_out, png_out, logger)


def plot_correlation(avg_df, metrics, png_out):
    valid=[m for m in metrics if m in avg_df]
    if len(valid)<2: return
    corr=avg_df[valid].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout(); plt.savefig(png_out, dpi=150); plt.close()


def plot_trajectories(df, n_samples, html_out, png_out, save_fn, logger):
    df['length']=df.groupby('global_track_id')['frame'].transform('count')
    ids=df['global_track_id'].unique()
    sampled=np.random.choice(ids, min(n_samples,len(ids)), replace=False)
    fig=go.Figure()
    for gid in sampled:
        sub=df[df['global_track_id']==gid].sort_values('frame')
        fig.add_trace(go.Scatter(x=sub['x'], y=sub['y'], mode='lines', name=str(gid)))
    fig.update_layout(title=f"Trajectories (sample {len(sampled)})", xaxis_title='X', yaxis_title='Y', yaxis_autorange='reversed')
    save_fn(fig, html_out, png_out, logger)

# --- Main ---
if __name__ == "__main__":
    logger=setup_logging()
    in_dir, out_dir = get_paths()
    files = find_input_files(in_dir)
    combined = load_and_combine(files, logger)
    filtered, track_count = filter_tracks(combined, min_len=20, logger=logger)
    prepared, area_col = prepare_data(filtered, frame_interval=5.0, pixel_size=0.08, logger=logger)
    metrics = select_metrics(prepared, area_col, logger)
    cols = metrics + [c for c in ['x','y'] if c in prepared]
    converted = convert_numeric(prepared, cols, logger)

    # Time course
    tc = calculate_time_course(converted, metrics)
    plot_time_course(tc, metrics, area_col, area_col, track_count, 20,
                     os.path.join(out_dir,'time_course.html'), os.path.join(out_dir,'time_course.png'), save_plotly, logger)
    # Distributions
    avg = calculate_track_averages(converted, metrics)
    plot_distributions(avg, metrics, area_col, area_col, track_count, 20,
                       os.path.join(out_dir,'distributions.html'), os.path.join(out_dir,'distributions.png'), save_plotly, logger)
    # Correlation heatmap
    plot_correlation(avg, metrics, os.path.join(out_dir,'correlation.png'))
    # Trajectories
    plot_trajectories(converted, n_samples=50,
                      html_out=os.path.join(out_dir,'trajectories.html'),
                      png_out=os.path.join(out_dir,'trajectories.png'), save_fn=save_plotly, logger=logger)

    logger.info("Processing complete.")
