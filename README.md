Mammalian Cell Migration & Focal Adhesion Dynamics Pipeline

Overview

This repository contains a high-throughput analysis pipeline designed for analyzing time-lapse microscopy data of Focal Adhesions (FA) and Vimentin filaments in migrating mammalian cells.

Unlike generic particle trackers, this pipeline implements specific biophysical metrics relevant to cell migration. It calculates Mean Squared Displacement (MSD), Anomalous Diffusion ($\alpha$), and Persistence, allowing for the mathematical differentiation between random diffusion, confined adhesion, and directed migration.

Key Scientific Features

1. Robust Normalization

Instead of normalizing to the very first frame (which is susceptible to shot noise and detection artifacts), trajectories are normalized to the mean of the first 3 frames ($I / I_{t_{0-2}}$). This ensures stable "Fold Change" metrics even in noisy live-cell data.

2. Trajectory Smoothing

Applies a rolling window average (window=3) to X/Y coordinates before calculating derivatives. This suppresses stage jitter and tracking noise, ensuring that calculated velocities reflect actual cell motility rather than pixel fluctuation.

3. Mean Squared Displacement (MSD) Analysis

Calculates the MSD for every track to classify the mode of motion:


$$MSD(\tau) \propto \tau^\alpha$$

$\alpha \approx 1$: Random Diffusion (Brownian motion).

$\alpha > 1$: Directed / Super-diffusive Motion (Active Migration).

$\alpha < 1$: Confined / Sub-diffusive Motion (Stable or trapped Adhesions).

4. Unsupervised Phenotyping

Uses PCA (Principal Component Analysis) and GMM (Gaussian Mixture Models) to cluster tracks into distinct populations (e.g., Disassembling FAs, Stable FAs, Sliding FAs) based on their multiparametric signature.

Installation

Prerequisites

Python 3.8+

Scientific Python Stack (numpy, pandas, scipy, scikit-learn)

Visualization (plotly, seaborn, matplotlib)

GUI (tkinter - usually included with standard Python installations)

Quick Install

Run the following command to install the necessary dependencies:

pip install pandas numpy plotly seaborn matplotlib scikit-learn scipy


Usage

Prepare Data: Ensure your tracking data is in .csv or .xlsx format. The script auto-detects column names, but standard headers include:

Track ID (or 'particle', 'id')

Frame (or 'time', 'slice')

X, Y

Intensity/Area columns (e.g., Mean Intensity, Area)

Run Script:

python migration_analysis_pipeline.py


Interactive Interface:

A file dialog will open asking for the Input Directory (the folder containing your CSVs).

A second dialog will ask for the Output Directory (where plots and results will be saved).

Output Files & Data Structure

The pipeline generates three types of outputs in the selected output directory.

1. The Results Table (_Results.csv)

This is the master data file. Each row represents a single tracked particle (FA or Cell). Key columns include:

Column Name

Description

global_track_id

Unique identifier (Filename + Particle ID).

msd_alpha

The Anomalous Diffusion Exponent. The primary metric for mode of motion (Directed vs. Random).

diffusion_coeff

The magnitude of diffusion (Speed of spreading).

velocity_avg

Average instantaneous speed ($\mu m / s$).

persistence_avg

Directional stability (-1 to 1). Closer to 1 means straight-line migration.

directionality

Net Displacement / Total Path Length.

cluster

The phenotypic group assigned by the GMM algorithm (0, 1, 2...).

[Metric]_norm_avg

The average intensity normalized to the start of the track.

2. Interactive Visualizations (.html)

These files can be opened in any web browser.

_PCA_2D.html: A scatter plot of the Principal Component Analysis. Hover over points to see which track belongs to which cluster.

Cluster_TimeCourses.html: Ribbon plots showing the behavior of each cluster over time. The solid line is the Mean, and the shaded region is the SEM (Standard Error of the Mean).

3. Static Plots (.png)

High-resolution images suitable for presentations or manuscripts.

PCA Scatter Plots: Visual representation of the feature space.

Time Course Plots: Static versions of the ribbon plots.

Configuration & Parameters

To adapt the script to your specific microscope or experiment, you can modify the Configuration section at the top of the Python script.

Key Variables

# --- Configuration ---

assumed_frame_interval = 5.0  
# The time gap between frames in seconds. 
# CRITICAL: This affects Velocity and Diffusion Coefficient calculations.

pixel_size_um = 0.08          
# The physical size of one pixel in microns.
# CRITICAL: This converts pixels to micrometers for Area and Speed.

min_track_length_filter = 20  
# Minimum number of frames a track must exist to be analyzed.
# Short tracks (<20) often yield unreliable MSD/Alpha calculations.

# --- Analysis Settings ---

CLUSTERING_ALGORITHM = 'GMM' 
# Options: 'GMM' (Gaussian Mixture) or 'KMEANS'.
# GMM is recommended for biological populations as it handles variance better.

N_COMPONENTS_GMM = 3
# The number of clusters (populations) to find. 


Advanced Features List

If you want to add or remove metrics from the Clustering/PCA analysis, modify the FEATURES_FOR_ANALYSIS list:

FEATURES_FOR_ANALYSIS = [
    'area_um2', 
    'fak_intensity_mean_norm', 
    'vimentin_intensity_mean_norm',
    'velocity', 
    'directionality', 
    'persistence', 
    'msd_alpha',       # The MSD exponent
    'diffusion_coeff'  # The MSD magnitude
]
