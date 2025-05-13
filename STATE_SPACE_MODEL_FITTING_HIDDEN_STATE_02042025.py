import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tkinter as tk # Import tkinter for the file dialog
from tkinter import filedialog # Import filedialog module
import statsmodels.api as sm # Import statsmodels for SSM

# --- Configuration ---
# The list of files (CSV or XLSX) and the destination folder will be selected interactively

# Define the column names based on the file inspection
# Ensure these match the column headers in your files exactly.
frame_col = 'frame' # Column representing the time point or frame number
object_id_col = 'label' # Column identifying individual ROIs/FAs within a file
area_col = 'area_um2' # Column for Area data
fak_intensity_col = 'fak_intensity_mean' # Column for FAK Intensity data
vimentin_intensity_col = 'vimentin_intensity_mean' # Column for Vimentin Intensity data

# List of parameters to include in the SSM multivariate time series
# Ensure these columns exist in your data and contain numerical values
ssm_parameters = [area_col, fak_intensity_col, vimentin_intensity_col]

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
                time_points = df_roi[frame_col].values # Use frame numbers as time points
                print(f"  Data for ROI/FA {current_roi_id} sorted by '{frame_col}'.")
            else:
                 print(f"  Warning: Frame/time column '{frame_col}' not found for ROI/FA {current_roi_id}. Assuming data is already ordered by time.")
                 time_points = np.arange(len(df_roi)) # Use row index as time points

            # --- Extract Multivariate Time Series for SSM ---
            try:
                # Select the columns for SSM analysis and convert to numpy array
                ssm_data = df_roi[ssm_parameters].values
                print(f"  Extracted SSM data for parameters: {ssm_parameters}")
                print(f"  SSM data shape: {ssm_data.shape}")

                # Check for missing or non-finite values in the data
                if np.isnan(ssm_data).any() or not np.isfinite(ssm_data).all():
                    print("  Warning: SSM data contains missing or non-finite values. Attempting to handle...")
                    # Simple handling: forward fill missing values
                    ssm_data_clean = pd.DataFrame(ssm_data).fillna(method='ffill').values
                    if np.isnan(ssm_data_clean).any() or not np.isfinite(ssm_data_clean).all():
                         print("  Error: Could not handle missing/non-finite values. Skipping SSM analysis for this ROI.")
                         continue
                    else:
                         ssm_data = ssm_data_clean
                         print("  Missing/non-finite values handled with forward fill.")


                if ssm_data.shape[0] < 2 or ssm_data.shape[1] != len(ssm_parameters):
                    print(f"  Error: SSM data shape is invalid ({ssm_data.shape}). Skipping SSM analysis for this ROI.")
                    continue

            except KeyError as e:
                print(f"  Error: One or more specified SSM parameters not found in data for ROI/FA {current_roi_id}: {e}. Skipping SSM analysis.")
                continue
            except Exception as e:
                print(f"  An error occurred while preparing SSM data for ROI/FA {current_roi_id}: {e}. Skipping SSM analysis.")
                continue


            # --- Fit Multivariate Local Level SSM ---
            print("  Fitting Multivariate Local Level SSM...")
            try:
                # Define the multivariate Local Level model
                # k_states = number of state variables (equal to number of observed variables in this model)
                # k_posdef = dimension of the state covariance matrix (equal to k_states in this model)
                model = sm.tsa.statespace.UnobservedComponents(
                    ssm_data,
                    level='local level',
                    k_states=len(ssm_parameters),
                    k_posdef=len(ssm_parameters)
                )

                # Fit the model using Maximum Likelihood Estimation (MLE)
                # disp=False suppresses optimization output
                results = model.fit(disp=False)
                print("  SSM fitting successful.")
                # print(results.summary()) # Uncomment to see the model summary in the console

                # --- Extract Smoothed State Estimates ---
                # The smoothed state provides the best estimate of the hidden state at each time point
                # using all the data (past and future).
                smoothed_states = results.smoothed_state.T # Transpose to have time points as rows

                # --- Save Smoothed State Estimates ---
                state_df = pd.DataFrame(smoothed_states, columns=[f'Smoothed_State_{i+1}' for i in range(smoothed_states.shape[1])])
                state_df.insert(0, frame_col, time_points) # Add frame numbers
                state_save_path = os.path.join(roi_output_path, f"ROI_{current_roi_id}_Smoothed_States.csv")
                state_df.to_csv(state_save_path, index=False)
                print(f"  Saved smoothed state estimates to: {state_save_path}")

                # --- Plot Smoothed State Estimates ---
                plt.figure(figsize=(10, 6))
                plt.suptitle(f'Cell: {cell_name} - ROI/FA: {current_roi_id} - Smoothed State Estimates', y=1.02)

                for i in range(smoothed_states.shape[1]):
                    plt.plot(time_points, smoothed_states[:, i], label=f'State {i+1}') # Plot each state variable

                plt.xlabel('Frame Number')
                plt.ylabel('Smoothed State Value')
                plt.title('Inferred Hidden State Trajectories')
                plt.legend()
                plt.grid(True)

                # Save the state plot
                state_plot_filename = "Smoothed_States_Plot.png"
                state_plot_save_path = os.path.join(roi_output_path, state_plot_filename)
                plt.savefig(state_plot_save_path)
                plt.close() # Close the plot figure

                print(f"  Saved smoothed state plot to: {state_plot_save_path}")


            except Exception as e:
                print(f"  An error occurred during SSM fitting or analysis for ROI/FA {current_roi_id}: {e}. Skipping SSM analysis.")
                continue


# --- Analysis Complete ---
print("\n--- Analysis Complete ---")
print("Iterated through each selected file (CSV or XLSX).")
print("For XLSX files, iterated through each sheet (cell). For CSV files, treated the file as a single cell.")
print("For each cell/sheet, iterated through each ROI/FA within that data.")
print(f"For each ROI/FA, a multivariate Local Level SSM was fitted to the time series of: {ssm_parameters}.")
print("The inferred smoothed hidden state estimates and plots of these states over time are saved in subfolders within the destination folder.")
print("\nNote: The interpretation of these hidden states depends on the chosen SSM structure.")
print("This script uses a simple Local Level model where states represent estimated underlying levels.")
print("More complex biological dynamics may require different SSM structures.")
print("\nTo analyze different sets of files, run the script again and select different files.")
print("To include different parameters in the SSM, modify the 'ssm_parameters' list in the script.")
