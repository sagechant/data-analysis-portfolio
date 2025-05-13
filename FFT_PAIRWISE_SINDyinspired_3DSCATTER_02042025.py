import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tkinter as tk # Import tkinter for the file dialog
from tkinter import filedialog # Import filedialog module
import json # Import json for saving results as JSON
import plotly.express as px # Import plotly express for interactive plotting
from scipy.stats import pearsonr # Import pearsonr for correlation calculation

# --- Configuration ---
# The list of files (CSV or XLSX) and the destination folder will be selected interactively

# Define the column names based on the file inspection
# Ensure these match the column headers in your files exactly.
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

print("Opening file selection dialog for input files (CSV or XLSX)...")
# Open the file dialog to select multiple CSV or XLSX files
input_files = filedialog.askopenfilenames(
    title="Select your tracking files (CSV or XLSX, each file or sheet represents a cell)",
    filetypes=(("Tracking Files", "*.csv *.xlsx"), ("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*"))
)

# Check if files were selected
if not input_files:
    print("Input file selection cancelled. Exiting.")
    sys.exit() # Exit the script if no files were selected

print(f"Selected {len(input_files)} input file(s):")
for f in input_files:
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
# List to store data for the interactive summary plots (all parameters)
plot_data_list_all_params = []
# List to store data for paired parameter analysis (Area, FAK, Vimentin)
paired_param_data_list = []


for selected_file in input_files:
    file_basename = os.path.basename(selected_file)
    print(f"\n--- Processing File: {file_basename} ---")

    data_frames = {} # Dictionary to hold dataframes from sheets or the single CSV

    # --- Load Data (Handle CSV and XLSX Sheets) ---
    try:
        if selected_file.lower().endswith('.csv'):
            # Attempt to read the CSV, explicitly setting comma delimiter and skipping bad lines
            df = pd.read_csv(selected_file, sep=',', on_bad_lines='skip')
            if df.empty:
                 print(f"Warning: CSV file {selected_file} is empty or could not be read after skipping bad lines. Skipping analysis for this file.")
                 continue
            # For CSV, treat the whole file as one "cell" and store in data_frames dict
            cell_name_from_file = os.path.splitext(file_basename)[0]
            data_frames[cell_name_from_file] = df
            print("CSV loaded successfully (skipped problematic lines if any).")

        elif selected_file.lower().endswith('.xlsx'):
            # Attempt to read all sheets from the XLSX file
            xls = pd.ExcelFile(selected_file)
            sheet_names = xls.sheet_names
            if not sheet_names:
                 print(f"Warning: XLSX file {selected_file} contains no sheets. Skipping analysis for this file.")
                 continue

            print(f"XLSX file loaded. Found sheets: {sheet_names}")
            # Read each sheet into the data_frames dictionary
            for sheet_name in sheet_names:
                 try:
                     df_sheet = xls.parse(sheet_name)
                     if not df_sheet.empty:
                         data_frames[sheet_name] = df_sheet
                         print(f"  Sheet '{sheet_name}' loaded successfully.")
                     else:
                         print(f"  Warning: Sheet '{sheet_name}' is empty. Skipping.")
                 except Exception as e:
                     print(f"  Error reading sheet '{sheet_name}': {e}. Skipping.")

            if not data_frames:
                 print(f"No data loaded from any sheets in {selected_file}. Skipping analysis for this file.")
                 continue

        else:
            print(f"Error: Unsupported file format for {selected_file}. Skipping.")
            continue # Skip to the next file

    except FileNotFoundError:
        print(f"Error: File not found at {selected_file}. Skipping.")
        continue # Skip to the next file
    except Exception as e:
        print(f"An error occurred while reading the file {selected_file}: {e}. Skipping.")
        continue # Skip to the next file

    # --- Process DataFrames (Sheets/CSV) ---
    # Now iterate through the loaded dataframes (each representing a cell)
    for cell_name, df in data_frames.items():
        print(f"\n--- Analyzing Cell (from '{file_basename}', sheet/name '{cell_name}') ---")

        # --- Identify ROIs/FAs within this Cell's data ---
        if object_id_col not in df.columns:
            print(f"Error: ROI/FA ID column '{object_id_col}' not found in data for cell '{cell_name}'. Skipping analysis for this cell.")
            continue

        unique_roi_ids = df[object_id_col].unique()
        print(f"Found {len(unique_roi_ids)} unique ROIs/FAs in data for cell '{cell_name}'.")

        if len(unique_roi_ids) == 0:
            print("No ROIs/FAs found in data for this cell. Skipping analysis.")
            continue

        # Add entry for this cell in the results dictionary
        all_results[cell_name] = {}

        # --- Analyze Each ROI/FA within this Cell ---
        for current_roi_id in unique_roi_ids:
            print(f"\n  -- Analyzing ROI/FA ID: {current_roi_id} (Cell: {cell_name}) --")

            # Create a folder for this ROI/FA within the destination folder
            # Use a safe folder name based on the ROI ID
            roi_folder_name = f"ROI_{current_roi_id}"
            # Output path structure: destination / CellName / ROI_ID
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

            # Dictionary to temporarily store results for Area, FAK, and Vimentin for this ROI
            roi_param_results = {}

            # --- Analyze Each Parameter for the Current ROI/FA ---
            for param_name, col_name in parameters_to_analyze.items():
                print(f"\n    Analyzing periodicity for: {param_name} (column '{col_name}')")

                if col_name not in df_roi.columns:
                    print(f"    Error: Parameter column '{col_name}' not found for ROI/FA {current_roi_id}. Skipping.")
                    all_results[cell_name][str(current_roi_id)][param_name] = {"status": "Column not found"}
                    # Append data for plot with status
                    plot_data_list_all_params.append({
                        'Cell': cell_name,
                        'ROI_ID': current_roi_id,
                        'Parameter': param_name,
                        'Dominant_Frequency_min': np.nan, # Use NaN for failed analysis
                        'Dominant_Period_min': np.nan,
                        'Dominant_Frequency_frame': np.nan,
                        'Dominant_Period_frame': np.nan,
                        'Dominant_Magnitude': np.nan,
                        'Analysis_Status': "Column not found"
                    })
                    continue

                # Extract the time series for the current parameter
                time_series = df_roi[col_name].values

                # Check if time series is valid for FFT
                if len(time_series) < 2 or np.all(time_series == time_series[0]):
                    print("    Skipping FFT: Time series is too short or constant.")
                    all_results[cell_name][str(current_roi_id)][param_name] = {"status": "Time series too short or constant"}
                     # Append data for plot with status
                    plot_data_list_all_params.append({
                        'Cell': cell_name,
                        'ROI_ID': current_roi_id,
                        'Parameter': param_name,
                        'Dominant_Frequency_min': np.nan, # Use NaN for failed analysis
                        'Dominant_Period_min': np.nan,
                        'Dominant_Frequency_frame': np.nan,
                        'Dominant_Period_frame': np.nan,
                        'Dominant_Magnitude': np.nan,
                        'Analysis_Status': "Time series too short or constant"
                    })
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

                # Store result temporarily for pairing (Area, FAK, Vimentin)
                if analysis_status == 'Success':
                     roi_param_results[param_name] = {
                        'freq_min': dominant_frequency_min,
                        'period_min': dominant_period_min
                    }


                # --- Append data for the interactive summary plots (all parameters) ---
                plot_data_list_all_params.append({
                    'Cell': cell_name,
                    'ROI_ID': current_roi_id,
                    'Parameter': param_name,
                    'Dominant_Frequency_min': dominant_frequency_min if dominant_period_min != float('inf') else np.nan, # Use NaN for infinite period
                    'Dominant_Period_min': dominant_period_min if dominant_period_min != float('inf') else np.nan,
                    'Dominant_Frequency_frame': dominant_frequency_frame if dominant_period_frame != float('inf') else np.nan,
                    'Dominant_Period_frame': dominant_period_frame if dominant_period_frame != float('inf') else np.nan,
                    'Dominant_Magnitude': dominant_magnitude if dominant_magnitude > 0 else np.nan, # Use NaN for zero magnitude
                    'Analysis_Status': analysis_status
                })


                # --- Save Individual Plot ---
                plt.figure(figsize=(14, 6))
                plt.suptitle(f'Cell: {cell_name} - ROI/FA: {current_roi_id} - {param_name}', y=1.02) # Add file/cell and ROI/FA to title

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

            # --- Collect Paired Parameter Data for Relationship Plots ---
            # Check if results for all three parameters are available and successful for this ROI
            if all(param in roi_param_results for param in parameters_to_analyze.keys()):
                paired_param_data_list.append({
                    'Cell': cell_name,
                    'ROI_ID': current_roi_id,
                    'Area_Freq_min': roi_param_results['Area']['freq_min'],
                    'Area_Period_min': roi_param_results['Area']['period_min'],
                    'FAK_Freq_min': roi_param_results['FAK Intensity']['freq_min'],
                    'FAK_Period_min': roi_param_results['FAK Intensity']['period_min'],
                    'Vimentin_Freq_min': roi_param_results['Vimentin Intensity']['freq_min'],
                    'Vimentin_Period_min': roi_param_results['Vimentin Intensity']['period_min']
                })


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

# --- Generate and Save Interactive Summary Plot (Frequency vs Period - All Parameters) ---
print("\nGenerating interactive summary plot (Frequency vs Period - All Parameters)...")

# Convert the list of plot data to a pandas DataFrame
plot_df_all_params = pd.DataFrame(plot_data_list_all_params)

# Drop rows where analysis failed (frequency/period is NaN) for the main scatter plot
plot_df_all_params_success = plot_df_all_params.dropna(subset=['Dominant_Frequency_min', 'Dominant_Period_min'])

if not plot_df_all_params_success.empty:
    # Create the interactive scatter plot using Plotly Express
    # Color by Cell and use different symbols for Parameter
    fig_freq_period_all = px.scatter(
        plot_df_all_params_success,
        x="Dominant_Frequency_min",
        y="Dominant_Period_min",
        color="Cell", # Color points by Cell
        symbol="Parameter", # Use different symbols for Parameter (Area, FAK, Vimentin)
        hover_data=['Cell', 'ROI_ID', 'Parameter', 'Dominant_Frequency_frame', 'Dominant_Period_frame', 'Dominant_Magnitude'], # Show detailed info on hover
        title="Dominant FFT Frequency vs. Period for All ROIs and Parameters (Color by Cell, Symbol by Parameter)",
        labels={
            "Dominant_Frequency_min": "Dominant Frequency (cycles/minute)",
            "Dominant_Period_min": "Dominant Period (minutes/cycle)"
        }
    )

    # Update marker style for transparent fill and outline
    fig_freq_period_all.update_traces(
        marker=dict(
            size=8, # Adjust marker size if needed
            opacity=0.8, # Set transparency
            line=dict(
                width=1, # Set outline width
                color='DarkSlateGrey' # Set outline color
            )
        ),
        selector=dict(mode='markers')
    )

    # Update layout for better readability
    fig_freq_period_all.update_layout(
        xaxis_title="Dominant Frequency (cycles/minute)",
        yaxis_title="Dominant Period (minutes/cycle)",
        hovermode='closest'
    )

    # Save the interactive plot as an HTML file
    interactive_plot_filename_freq_period_all = "interactive_fft_summary_plot_freq_period_all_params.html"
    interactive_plot_path_freq_period_all = os.path.join(destination_folder, interactive_plot_filename_freq_period_all)
    fig_freq_period_all.write_html(interactive_plot_path_freq_period_all)

    print(f"Saved interactive summary plot (Frequency vs Period - All Parameters) to: {interactive_plot_path_freq_period_all}")
    print(f"Open '{interactive_plot_filename_freq_period_all}' in a web browser to view the interactive plot.")

else:
    print("No successful FFT analyses to plot in the interactive summary (Frequency vs Period - All Parameters).")


# --- Generate and Save Interactive Summary Plot (Cells vs Frequency) ---
print("\nGenerating interactive summary plot (Cells vs Frequency)...")

if not plot_df_all_params_success.empty:
    # Create the interactive scatter plot for Cells vs Dominant Frequency
    fig_cell_freq = px.scatter(
        plot_df_all_params_success,
        x="Cell", # Cell on the x-axis
        y="Dominant_Frequency_min", # Dominant Frequency on the y-axis
        color="Parameter", # Color points by Parameter
        symbol="Parameter", # Use different symbols for Parameter
        hover_data=['Cell', 'ROI_ID', 'Parameter', 'Dominant_Frequency_frame', 'Dominant_Period_min', 'Dominant_Period_frame', 'Dominant_Magnitude'], # Show detailed info on hover
        title="Dominant FFT Frequency by Cell (Color and Symbol by Parameter)",
        labels={
            "Cell": "Cell",
            "Dominant_Frequency_min": "Dominant Frequency (cycles/minute)"
        }
    )

    # Update marker style for transparent fill and outline
    fig_cell_freq.update_traces(
        marker=dict(
            size=8, # Adjust marker size if needed
            opacity=0.8, # Set transparency
            line=dict(
                width=1, # Set outline width
                color='DarkSlateGrey' # Set outline color
            )
        ),
        selector=dict(mode='markers')
    )

    # Update layout for better readability
    fig_cell_freq.update_layout(
        xaxis_title="Cell",
        yaxis_title="Dominant Frequency (cycles/minute)",
        hovermode='closest'
    )

    # Save the interactive plot as an HTML file
    interactive_plot_filename_cell_freq = "interactive_fft_summary_plot_cell_frequency.html"
    interactive_plot_path_cell_freq = os.path.join(destination_folder, interactive_plot_filename_cell_freq)
    fig_cell_freq.write_html(interactive_plot_path_cell_freq)

    print(f"Saved interactive summary plot (Cells vs Frequency) to: {interactive_plot_path_cell_freq}")
    print(f"Open '{interactive_plot_filename_cell_freq}' in a web browser to view the interactive plot.")

else:
    print("No successful FFT analyses to plot in the interactive summary (Cells vs Frequency).")


# --- Generate and Save Interactive Summary Plot (Cells vs Period) ---
print("\nGenerating interactive summary plot (Cells vs Period)...")

if not plot_df_all_params_success.empty:
    # Create the interactive scatter plot for Cells vs Dominant Period
    fig_cell_period = px.scatter(
        plot_df_all_params_success,
        x="Cell", # Cell on the x-axis
        y="Dominant_Period_min", # Dominant Period on the y-axis
        color="Parameter", # Color points by Parameter
        symbol="Parameter", # Use different symbols for Parameter
        hover_data=['Cell', 'ROI_ID', 'Parameter', 'Dominant_Frequency_min', 'Dominant_Frequency_frame', 'Dominant_Period_frame', 'Dominant_Magnitude'], # Show detailed info on hover
        title="Dominant FFT Period by Cell (Color and Symbol by Parameter)",
        labels={
            "Cell": "Cell",
            "Dominant_Period_min": "Dominant Period (minutes/cycle)"
        }
    )

    # Update marker style for transparent fill and outline
    fig_cell_period.update_traces(
        marker=dict(
            size=8, # Adjust marker size if needed
            opacity=0.8, # Set transparency
            line=dict(
                width=1, # Set outline width
                color='DarkSlateGrey' # Set outline color
            )
        ),
        selector=dict(mode='markers')
    )

    # Update layout for better readability
    fig_cell_period.update_layout(
        xaxis_title="Cell",
        yaxis_title="Dominant Period (minutes/cycle)",
        hovermode='closest'
    )

    # Save the interactive plot as an HTML file
    interactive_plot_filename_cell_period = "interactive_fft_summary_plot_cell_period.html"
    interactive_plot_path_cell_period = os.path.join(destination_folder, interactive_plot_filename_cell_period)
    fig_cell_period.write_html(interactive_plot_path_cell_period)

    print(f"Saved interactive summary plot (Cells vs Period) to: {interactive_plot_path_cell_period}")
    print(f"Open '{interactive_plot_filename_cell_period}' in a web browser to view the interactive plot.")

else:
    print("No successful FFT analyses to plot in the interactive summary (Cells vs Period).")


# --- Analyze and Report Relationships between Area, FAK, and Vimentin Periodicities ---
print("\n--- Analyzing Relationships between Area, FAK, and Vimentin Periodicities ---")

# Convert the list of paired data to a pandas DataFrame
paired_param_df = pd.DataFrame(paired_param_data_list)

if not paired_param_df.empty:
    print(f"Found {len(paired_param_df)} ROIs with successful FFT analysis for Area, FAK, and Vimentin.")

    # Calculate and print Correlation Matrix for Dominant Frequencies
    correlation_matrix_freq = paired_param_df[['Area_Freq_min', 'FAK_Freq_min', 'Vimentin_Freq_min']].corr(method='pearson')
    print("\nPearson Correlation Matrix (Dominant Frequency, cycles/min):")
    print(correlation_matrix_freq.to_markdown(numalign="left", stralign="left"))

    # Calculate and print Correlation Matrix for Dominant Periods
    correlation_matrix_period = paired_param_df[['Area_Period_min', 'FAK_Period_min', 'Vimentin_Period_min']].corr(method='pearson')
    print("\nPearson Correlation Matrix (Dominant Period, minutes/cycle):")
    print(correlation_matrix_period.to_markdown(numalign="left", stralign="left"))


    # Generate interactive scatter plots for all pairwise relationships
    # FAK vs Vimentin Frequency
    fig_fak_vim_freq = px.scatter(
        paired_param_df,
        x="FAK_Freq_min",
        y="Vimentin_Freq_min",
        color="Cell", # Color by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Freq_min', 'FAK_Freq_min', 'Vimentin_Freq_min'],
        title="Dominant FAK Frequency vs. Dominant Vimentin Frequency (Color by Cell)",
        labels={
            "FAK_Freq_min": "Dominant FAK Frequency (cycles/minute)",
            "Vimentin_Freq_min": "Dominant Vimentin Frequency (cycles/minute)"
        }
    )
    fig_fak_vim_freq.update_layout(hovermode='closest')
    interactive_plot_filename_fak_vim_freq = "interactive_fft_summary_plot_fak_vim_frequency.html"
    interactive_plot_path_fak_vim_freq = os.path.join(destination_folder, interactive_plot_filename_fak_vim_freq)
    fig_fak_vim_freq.write_html(interactive_plot_path_fak_vim_freq)
    print(f"Saved interactive summary plot (FAK vs Vimentin Frequency) to: {interactive_plot_path_fak_vim_freq}")

    # FAK vs Vimentin Period
    fig_fak_vim_period = px.scatter(
        paired_param_df,
        x="FAK_Period_min",
        y="Vimentin_Period_min",
        color="Cell", # Color by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Period_min', 'FAK_Period_min', 'Vimentin_Period_min'],
        title="Dominant FAK Period vs. Dominant Vimentin Period (Color by Cell)",
        labels={
            "FAK_Period_min": "Dominant FAK Period (minutes/cycle)",
            "Vimentin_Period_min": "Dominant Vimentin Period (minutes/cycle)"
        }
    )
    fig_fak_vim_period.update_layout(hovermode='closest')
    interactive_plot_filename_fak_vim_period = "interactive_fft_summary_plot_fak_vim_period.html"
    interactive_plot_path_fak_vim_period = os.path.join(destination_folder, interactive_plot_filename_fak_vim_period)
    fig_fak_vim_period.write_html(interactive_plot_path_fak_vim_period)
    print(f"Saved interactive summary plot (FAK vs Vimentin Period) to: {interactive_plot_path_fak_vim_period}")

    # FAK vs Area Frequency
    fig_fak_area_freq = px.scatter(
        paired_param_df,
        x="FAK_Freq_min",
        y="Area_Freq_min",
        color="Cell", # Color by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Freq_min', 'FAK_Freq_min', 'Vimentin_Freq_min'],
        title="Dominant FAK Frequency vs. Dominant Area Frequency (Color by Cell)",
        labels={
            "FAK_Freq_min": "Dominant FAK Frequency (cycles/minute)",
            "Area_Freq_min": "Dominant Area Frequency (cycles/minute)"
        }
    )
    fig_fak_area_freq.update_layout(hovermode='closest')
    interactive_plot_filename_fak_area_freq = "interactive_fft_summary_plot_fak_area_frequency.html"
    interactive_plot_path_fak_area_freq = os.path.join(destination_folder, interactive_plot_filename_fak_area_freq)
    fig_fak_area_freq.write_html(interactive_plot_path_fak_area_freq)
    print(f"Saved interactive summary plot (FAK vs Area Frequency) to: {interactive_plot_path_fak_area_freq}")

    # FAK vs Area Period
    fig_fak_area_period = px.scatter(
        paired_param_df,
        x="FAK_Period_min",
        y="Area_Period_min",
        color="Cell", # Color by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Period_min', 'FAK_Period_min', 'Vimentin_Period_min'],
        title="Dominant FAK Period vs. Dominant Area Period (Color by Cell)",
        labels={
            "FAK_Period_min": "Dominant FAK Period (minutes/cycle)",
            "Area_Period_min": "Dominant Area Period (minutes/cycle)"
        }
    )
    fig_fak_area_period.update_layout(hovermode='closest')
    interactive_plot_filename_fak_area_period = "interactive_fft_summary_plot_fak_area_period.html"
    interactive_plot_path_fak_area_period = os.path.join(destination_folder, interactive_plot_filename_fak_area_period)
    fig_fak_area_period.write_html(interactive_plot_path_fak_area_period)
    print(f"Saved interactive summary plot (FAK vs Area Period) to: {interactive_plot_path_fak_area_period}")

    # Vimentin vs Area Frequency
    fig_vim_area_freq = px.scatter(
        paired_param_df,
        x="Vimentin_Freq_min",
        y="Area_Freq_min",
        color="Cell", # Color by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Freq_min', 'FAK_Freq_min', 'Vimentin_Freq_min'],
        title="Dominant Vimentin Frequency vs. Dominant Area Frequency (Color by Cell)",
        labels={
            "Vimentin_Freq_min": "Dominant Vimentin Frequency (cycles/minute)",
            "Area_Freq_min": "Dominant Area Frequency (cycles/minute)"
        }
    )
    fig_vim_area_freq.update_layout(hovermode='closest')
    interactive_plot_filename_vim_area_freq = "interactive_fft_summary_plot_vim_area_frequency.html"
    interactive_plot_path_vim_area_freq = os.path.join(destination_folder, interactive_plot_filename_vim_area_freq)
    fig_vim_area_freq.write_html(interactive_plot_path_vim_area_freq)
    print(f"Saved interactive summary plot (Vimentin vs Area Frequency) to: {interactive_plot_path_vim_area_freq}")

    # Vimentin vs Area Period
    fig_vim_area_period = px.scatter(
        paired_param_df,
        x="Vimentin_Period_min",
        y="Area_Period_min",
        color="Cell", # Color by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Period_min', 'FAK_Period_min', 'Vimentin_Period_min'],
        title="Dominant Vimentin Period vs. Dominant Area Period (Color by Cell)",
        labels={
            "Vimentin_Period_min": "Dominant Vimentin Period (minutes/cycle)",
            "Area_Period_min": "Dominant Area Period (minutes/cycle)"
        }
    )
    fig_vim_area_period.update_layout(hovermode='closest')
    interactive_plot_filename_vim_area_period = "interactive_fft_summary_plot_vim_area_period.html"
    interactive_plot_path_vim_area_period = os.path.join(destination_folder, interactive_plot_filename_vim_area_period)
    fig_vim_area_period.write_html(interactive_plot_path_vim_area_period)
    print(f"Saved interactive summary plot (Vimentin vs Area Period) to: {interactive_plot_path_vim_area_period}")


else:
    print("\nNot enough ROIs with successful FFT analysis for all specified parameters to calculate correlation or generate pairwise plots.")


# --- Generate and Save Interactive 3D Scatter Plot (Area, FAK, Vimentin Frequencies) ---
print("\nGenerating interactive 3D scatter plot (Area, FAK, Vimentin Frequencies)...")

if not paired_param_df.empty:
    fig_3d_freq = px.scatter_3d(
        paired_param_df,
        x='Area_Freq_min',
        y='FAK_Freq_min',
        z='Vimentin_Freq_min',
        color='Cell', # Color points by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Freq_min', 'FAK_Freq_min', 'Vimentin_Freq_min', 'Area_Period_min', 'FAK_Period_min', 'Vimentin_Period_min'], # Show detailed info on hover
        title='Dominant Frequency Relationship (Area vs FAK vs Vimentin)',
        labels={
            'Area_Freq_min': 'Dominant Area Frequency (cycles/min)',
            'FAK_Freq_min': 'Dominant FAK Frequency (cycles/min)',
            'Vimentin_Freq_min': 'Dominant Vimentin Frequency (cycles/min)'
        }
    )

    # Update marker style for transparent fill and outline (similar to 2D plots)
    fig_3d_freq.update_traces(
        marker=dict(
            size=5, # Adjust size for 3D
            opacity=0.8,
            line=dict(
                width=1,
                color='DarkSlateGrey'
            )
        )
    )

    # Update layout for readability
    fig_3d_freq.update_layout(
        scene = dict( # Use 'scene' for 3D layout updates
            xaxis_title='Dominant Area Frequency (cycles/min)',
            yaxis_title='Dominant FAK Frequency (cycles/min)',
            zaxis_title='Dominant Vimentin Frequency (cycles/min)'
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Adjust margin
    )

    # Save the 3D interactive plot as an HTML file
    interactive_plot_filename_3d_freq = "interactive_fft_summary_plot_3d_frequency.html"
    interactive_plot_path_3d_freq = os.path.join(destination_folder, interactive_plot_filename_3d_freq)
    fig_3d_freq.write_html(interactive_plot_path_3d_freq)

    print(f"Saved interactive 3D scatter plot (Frequencies) to: {interactive_plot_path_3d_freq}")
    print(f"Open '{interactive_plot_filename_3d_freq}' in a web browser to view the interactive plot.")

else:
    print("\nNot enough ROIs with successful FFT analysis for all specified parameters to generate 3D frequency plot.")


# --- Generate and Save Interactive 3D Scatter Plot (Area, FAK, Vimentin Periods) ---
print("\nGenerating interactive 3D scatter plot (Area, FAK, Vimentin Periods)...")

if not paired_param_df.empty:
    fig_3d_period = px.scatter_3d(
        paired_param_df,
        x='Area_Period_min',
        y='FAK_Period_min',
        z='Vimentin_Period_min',
        color='Cell', # Color points by Cell
        hover_data=['Cell', 'ROI_ID', 'Area_Period_min', 'FAK_Period_min', 'Vimentin_Period_min', 'Area_Freq_min', 'FAK_Freq_min', 'Vimentin_Freq_min'], # Show detailed info on hover
        title='Dominant Period Relationship (Area vs FAK vs Vimentin)',
        labels={
            'Area_Period_min': 'Dominant Area Period (min/cycle)',
            'FAK_Period_min': 'Dominant FAK Period (min/cycle)',
            'Vimentin_Period_min': 'Dominant Vimentin Period (min/cycle)'
        }
    )

    # Update marker style for transparent fill and outline (similar to 2D plots)
    fig_3d_period.update_traces(
        marker=dict(
            size=5, # Adjust size for 3D
            opacity=0.8,
            line=dict(
                width=1,
                color='DarkSlateGrey'
            )
        )
    )

    # Update layout for readability
    fig_3d_period.update_layout(
        scene = dict( # Use 'scene' for 3D layout updates
            xaxis_title='Dominant Area Period (min/cycle)',
            yaxis_title='Dominant FAK Period (min/cycle)',
            zaxis_title='Dominant Vimentin Period (min/cycle)'
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Adjust margin
    )

    # Save the 3D interactive plot as an HTML file
    interactive_plot_filename_3d_period = "interactive_fft_summary_plot_3d_period.html"
    interactive_plot_path_3d_period = os.path.join(destination_folder, interactive_plot_filename_3d_period)
    fig_3d_period.write_html(interactive_plot_path_3d_period)

    print(f"Saved interactive 3D scatter plot (Periods) to: {interactive_plot_path_3d_period}")
    print(f"Open '{interactive_plot_filename_3d_period}' in a web browser to view the interactive plot.")

else:
    print("\nNot enough ROIs with successful FFT analysis for all specified parameters to generate 3D period plot.")


# --- Analysis Complete ---
print("\n--- Analysis Complete ---")
print("Iterated through each selected file (CSV or XLSX).")
print("For XLSX files, iterated through each sheet (cell). For CSV files, treated the file as a single cell.")
print("For each cell/sheet, iterated through each ROI/FA within that data.")
print("For each ROI/FA, FFT was performed on Area, FAK Intensity, and Vimentin Intensity time series.")
print("Dominant frequency and period are calculated and saved in cycles/minute and minutes/cycle, as well as cycles/frame and frames/cycle.")
print("Plots with periodicity information and text summaries for each ROI/FA are saved in subfolders within the destination folder.")
print("A comprehensive JSON summary of all dominant frequencies and periods is saved in the destination folder.")
print("Several interactive scatter plots summarizing dominant periodicities are saved as HTML files:")
print("- Frequency vs Period (All Parameters, Color by Cell, Symbol by Parameter)")
print("- Cells vs Dominant Frequency (All Parameters, Color and Symbol by Parameter)")
print("- Cells vs Dominant Period (All Parameters, Color and Symbol by Parameter)")
print("- Pairwise 2D plots of Dominant Frequency and Period for Area, FAK, and Vimentin (Color by Cell)")
print("- 3D scatter plot of Dominant Frequencies (Area vs FAK vs Vimentin, Color by Cell)")
print("- 3D scatter plot of Dominant Periods (Area vs FAK vs Vimentin, Color by Cell)")

print("\nPearson correlation matrices for dominant frequencies and periods of Area, FAK, and Vimentin are calculated and printed.")
print("\nTo analyze different sets of files, run the script again and select different files.")
print("To analyze other parameters, modify the 'parameters_to_analyze' dictionary using the correct column names in the script.")
