import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tkinter as tk # Import tkinter for the file dialog
from tkinter import filedialog # Import filedialog module
import json # Import json for saving results as JSON

# --- Configuration ---
# The list of CSV files and the destination folder will be selected interactively

# Define the column names based on the CSV inspection
# Ensure these match the column headers in your CSV files exactly.
frame_col = 'frame' # Column representing the time point or frame number
object_id_col = 'label' # Column identifying individual ROIs/FAs within a file
area_col = 'area_um2' # Column for Area data
fak_intensity_col = 'fak_intensity_mean' # Column for FAK Intensity data
vimentin_intensity_col = 'vimentin_intensity_mean' # Column for Vimentin Intensity data

# Define the time interval between frames in seconds
frame_interval_sec = 5 # Each frame is obtained every 5 seconds

# List of parameters to analyze. Map a descriptive name to the actual column name.
parameters_to_analyze = {
    'Area': area_col,
    'FAK Intensity': fak_intensity_col,
    'Vimentin Intensity': vimentin_intensity_col
}

# --- Calculate Time Units ---
frame_interval_min = frame_interval_sec / 60.0 # Convert frame interval to minutes

# --- Interactive File and Folder Selection ---
# Create a simple tkinter window (it won't be shown)
root = tk.Tk()
root.withdraw() # Hide the main window

print("Opening file selection dialog for input CSVs...")
# Open the file dialog to select multiple CSV files
csv_files = filedialog.askopenfilenames(
    title="Select your CSV tracking files (each file represents a cell)",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
)

# Check if files were selected
if not csv_files:
    print("Input file selection cancelled. Exiting.")
    sys.exit() # Exit the script if no files were selected

print(f"Selected {len(csv_files)} input file(s):")
for f in csv_files:
    print(f"- {f}")

print("\nOpening folder selection dialog for output destination...")
# Open the folder dialog to select the destination directory
destination_folder = filedialog.askdirectory(
    title="Select a destination folder to save results"
)

# Check if a destination folder was selected
if not destination_folder:
    print("Destination folder selection cancelled. Exiting.")
    sys.exit() # Exit the script if no folder was selected

print(f"Selected destination folder: {destination_folder}")

# --- Analysis Loops ---

# Dictionary to store all results for JSON output
all_results = {}

for selected_file in csv_files:
    file_basename = os.path.basename(selected_file)
    cell_name = os.path.splitext(file_basename)[0] # Use filename (without extension) as cell name
    print(f"\n--- Analyzing File (Cell): {file_basename} ---")

    # --- Load CSV Data ---
    try:
        df = pd.read_csv(selected_file)
        print("CSV loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {selected_file}. Skipping.")
        continue # Skip to the next file
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}. Skipping.")
        continue # Skip to the next file

    # --- Identify ROIs/FAs within this Cell's file ---
    if object_id_col not in df.columns:
        print(f"Error: ROI/FA ID column '{object_id_col}' not found in {selected_file}. Skipping analysis for this file.")
        continue

    unique_roi_ids = df[object_id_col].unique()
    print(f"Found {len(unique_roi_ids)} unique ROIs/FAs in this file.")

    if len(unique_roi_ids) == 0:
        print("No ROIs/FAs found in this file. Skipping analysis.")
        continue

    # Add entry for this cell in the results dictionary
    all_results[cell_name] = {}

    # --- Analyze Each ROI/FA within this Cell ---
    for current_roi_id in unique_roi_ids:
        print(f"\n  -- Analyzing ROI/FA ID: {current_roi_id} --")

        # Create a folder for this ROI/FA within the destination folder
        # Use a safe folder name based on the ROI ID
        roi_folder_name = f"ROI_{current_roi_id}"
        roi_output_path = os.path.join(destination_folder, cell_name, roi_folder_name)
        os.makedirs(roi_output_path, exist_ok=True) # Create directory if it doesn't exist

        # Filter data for the current ROI/FA
        df_roi = df[df[object_id_col] == current_roi_id].copy()

        # --- Ensure data is sorted by Frame/Time ---
        if frame_col in df_roi.columns:
            df_roi = df_roi.sort_values(by=frame_col).reset_index(drop=True)
            # Use frame numbers as time points for plotting if frame_rate is None
            # If frame_rate is provided, we'll calculate time based on that
            time_points_for_plot = df_roi[frame_col].values
            print(f"  Data for ROI/FA {current_roi_id} sorted by '{frame_col}'.")
        else:
             print(f"  Warning: Frame/time column '{frame_col}' not found for ROI/FA {current_roi_id}. Assuming data is already ordered by time.")
             time_points_for_plot = np.arange(len(df_roi)) # Use row index as time points

        # Add entry for this ROI in the cell's results dictionary
        all_results[cell_name][str(current_roi_id)] = {} # Use string key for JSON compatibility


        # --- Analyze Each Parameter for the Current ROI/FA ---
        for param_name, col_name in parameters_to_analyze.items():
            print(f"\n    Analyzing periodicity for: {param_name} (column '{col_name}')")

            if col_name not in df_roi.columns:
                print(f"    Error: Parameter column '{col_name}' not found for ROI/FA {current_roi_id}. Skipping.")
                all_results[cell_name][str(current_roi_id)][param_name] = {"status": "Column not found"}
                continue

            # Extract the time series for the current parameter
            time_series = df_roi[col_name].values

            # Check if time series is valid for FFT
            if len(time_series) < 2 or np.all(time_series == time_series[0]):
                print("    Skipping FFT: Time series is too short or constant.")
                all_results[cell_name][str(current_roi_id)][param_name] = {"status": "Time series too short or constant"}
                continue

            # --- Perform FFT Analysis ---
            N = len(time_series) # Number of data points

            # Compute the 1D FFT
            fft_result = np.fft.fft(time_series)

            # Calculate the frequencies in cycles per MINUTE
            frequencies_min = np.fft.fftfreq(N, d=frame_interval_min)

            # Calculate the frequencies in cycles per FRAME (for comparison/completeness)
            frequencies_frame = np.fft.fftfreq(N, d=1.0)


            # We only need the positive frequencies for real input data
            # The spectrum is symmetric, so we take the first half
            positive_frequencies_min = frequencies_min[:N//2]
            positive_frequencies_frame = frequencies_frame[:N//2]
            fft_magnitudes = np.abs(fft_result)[:N//2]

            # --- Find Dominant Frequency and Period ---

            # Ensure there are positive frequencies to analyze (N must be > 1)
            if len(positive_frequencies_min) < 2 or len(fft_magnitudes[1:]) == 0:
                print("    Could not find a dominant peak: Not enough frequency points after excluding DC.")
                dominant_frequency_min = 0
                dominant_frequency_frame = 0
                dominant_magnitude = 0
                dominant_period_min = float('inf')
                dominant_period_frame = float('inf')
                analysis_status = "Could not find dominant peak"
            else:
                # Find the index of the maximum magnitude in the positive frequencies, excluding the DC component (index 0)
                # We start from index 1
                dominant_freq_index = np.argmax(fft_magnitudes[1:]) + 1 # +1 because we excluded index 0

                dominant_frequency_min = positive_frequencies_min[dominant_freq_index]
                dominant_frequency_frame = positive_frequencies_frame[dominant_freq_index]
                dominant_magnitude = fft_magnitudes[dominant_freq_index]

                # Calculate the period
                # Period = 1 / Frequency
                if dominant_frequency_min != 0: # Avoid division by zero
                    dominant_period_min = 1.0 / dominant_frequency_min
                    dominant_period_frame = 1.0 / dominant_frequency_frame # Calculate period in frames/cycle as well
                    analysis_status = "Success"
                else:
                    dominant_period_min = float('inf') # Infinite period for zero frequency
                    dominant_period_frame = float('inf')
                    analysis_status = "Dominant frequency is zero"


            # --- Store Results for Output ---
            result_data = {
                "status": analysis_status,
                "dominant_frequency_cycles_per_minute": dominant_frequency_min,
                "dominant_period_minutes_per_cycle": dominant_period_min,
                "dominant_frequency_cycles_per_frame": dominant_frequency_frame, # Store frame units too
                "dominant_period_frames_per_cycle": dominant_period_frame, # Store frame units too
                "dominant_magnitude": dominant_magnitude,
                "notes": "Analysis of the single most dominant frequency peak excluding DC component."
            }
            all_results[cell_name][str(current_roi_id)][param_name] = result_data

            # --- Save Plot ---
            plt.figure(figsize=(14, 6))
            plt.suptitle(f'Cell: {os.path.basename(selected_file)} - ROI/FA: {current_roi_id} - {param_name}', y=1.02) # Add file/cell and ROI/FA to title

            # Plot the original time series
            plt.subplot(1, 2, 1)
            # Calculate time points in minutes for the plot
            time_points_min = time_points_for_plot * frame_interval_min
            plt.plot(time_points_min, time_series)

            plt.xlabel(f'Time (minutes)')
            plt.ylabel(param_name)
            plt.title(f'{param_name} over Time')
            plt.grid(True)


            # Plot the magnitude spectrum of the FFT (using cycles/minute)
            plt.subplot(1, 2, 2)
            # Plot from the second element (index 1) onwards to exclude the DC component
            # Ensure there are points to plot after excluding index 0
            if len(positive_frequencies_min[1:]) > 0:
                plt.plot(positive_frequencies_min[1:], fft_magnitudes[1:])
                plt.xlabel(f'Frequency (cycles/minute)')
                plt.ylabel('Magnitude')
                plt.title(f'FFT Spectrum ({param_name})')
                plt.grid(True)

                # Highlight the dominant frequency on the plot (optional)
                if dominant_period_min != float('inf') and dominant_frequency_min != 0 and dominant_magnitude > 0:
                     # Avoid plotting text if the dominant magnitude is too small relative to max
                     if len(fft_magnitudes[1:]) > 0 and dominant_magnitude > (np.max(fft_magnitudes[1:]) * 0.1): # Check magnitude against max exclude DC
                         plt.scatter(dominant_frequency_min, dominant_magnitude, color='red', zorder=5)
                         # The previous text annotation for dominant frequency is now replaced by the info box

            # --- Add Text Box with Periodicity Information ---
            if result_data.get('status') == 'Success':
                info_text = (
                    f"Dominant Freq: {dominant_frequency_min:.4f} cycles/min\n"
                    f"Period: {dominant_period_min:.4f} min/cycle\n"
                    f"Dominant Freq: {dominant_frequency_frame:.4f} cycles/frame\n"
                    f"Period: {dominant_period_frame:.4f} frames/cycle"
                )
            else:
                 info_text = f"Analysis Status: {result_data.get('status', 'N/A')}"


            # Add the text box to the top right of the FFT spectrum plot
            plt.text(
                0.95, # x-coordinate (0.95 means 95% of the way across the x-axis from left)
                0.95, # y-coordinate (0.95 means 95% of the way up the y-axis from bottom)
                info_text,
                horizontalalignment='right', # Align the text to the right side of the x-coordinate
                verticalalignment='top',     # Align the text to the top side of the y-coordinate
                transform=plt.gca().transAxes, # Use axes coordinates (0 to 1)
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5) # Add a rounded box with padding and transparency
            )


            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to make space for suptitle

            # Save the plot
            plot_filename = f"{param_name}_FFT_Analysis.png"
            plot_save_path = os.path.join(roi_output_path, plot_filename)
            plt.savefig(plot_save_path)
            plt.close() # Close the plot figure to free up memory

            print(f"    Saved plot to: {plot_save_path}")

        # --- Save Text Summary for the ROI/FA ---
        text_summary_filename = f"ROI_{current_roi_id}_FFT_Summary.txt"
        text_summary_path = os.path.join(roi_output_path, text_summary_filename)
        with open(text_summary_path, 'w') as f:
            f.write(f"--- FFT Analysis Summary for ROI/FA ID: {current_roi_id} ---\n\n")
            for param_name, result_data in all_results[cell_name][str(current_roi_id)].items():
                f.write(f"Parameter: {param_name}\n")
                f.write(f"  Status: {result_data.get('status', 'N/A')}\n")
                if result_data.get('status') == 'Success':
                    f.write(f"  Dominant Frequency (cycles/minute): {result_data['dominant_frequency_cycles_per_minute']:.4f}\n")
                    if result_data['dominant_period_minutes_per_cycle'] != float('inf'):
                         f.write(f"  Corresponding Period (minutes/cycle): {result_data['dominant_period_minutes_per_cycle']:.4f}\n")
                    else:
                         f.write("  Corresponding Period (minutes/cycle): Infinite\n")
                    f.write(f"  Dominant Frequency (cycles/frame): {result_data['dominant_frequency_cycles_per_frame']:.4f}\n") # Add frame units
                    if result_data['dominant_period_frames_per_cycle'] != float('inf'):
                         f.write(f"  Corresponding Period (frames/cycle): {result_data['dominant_period_frames_per_cycle']:.4f}\n") # Add frame units
                    else:
                         f.write("  Corresponding Period (frames/cycle): Infinite\n")
                else:
                    f.write(f"  Notes: {result_data.get('status', 'N/A')}\n")
                f.write("\n")
        print(f"  Saved text summary to: {text_summary_path}")


# --- Save JSON Summary for All Results ---
json_summary_filename = "all_fft_results_summary.json"
json_summary_path = os.path.join(destination_folder, json_summary_filename)

# Handle potential infinity values before saving to JSON
def convert_infinite(obj):
    if isinstance(obj, float) and obj == float('inf'):
        return "Infinity"
    # Handle numpy float types as well
    if isinstance(obj, np.floating) and np.isinf(obj):
        return "Infinity"
    # Handle numpy integer types
    if isinstance(obj, np.integer):
        return int(obj)
    # Handle numpy boolean types
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Handle numpy array types (optional, but good practice if you were saving arrays)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

with open(json_summary_path, 'w') as f:
    # Use json.dump with a custom default function to handle infinity and numpy types
    json.dump(all_results, f, indent=4, default=convert_infinite)

print(f"\nSaved comprehensive JSON summary to: {json_summary_path}")


# --- Analysis Complete ---
print("\n--- Analysis Complete ---")
print("Iterated through each selected file (cell) and each ROI/FA within that file.")
print("For each ROI/FA, FFT was performed on Area, FAK Intensity, and Vimentin Intensity time series.")
print("Dominant frequency and period are calculated and saved in cycles/minute and minutes/cycle, as well as cycles/frame and frames/cycle.")
print("Plots with periodicity information and text summaries for each ROI/FA are saved in subfolders within the destination folder.")
print("A comprehensive JSON summary of all dominant frequencies and periods is saved in the destination folder.")
print("\nTo analyze different sets of files, run the script again and select different files.")
print("To analyze other parameters, modify the 'parameters_to_analyze' dictionary using the correct column names in the script.")
