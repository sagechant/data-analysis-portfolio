import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap functionality
import seaborn as sns
import os
import sys
import math
import tkinter as tk
from tkinter import filedialog
import logging
import glob # Kept import, may be useful for future batch processing
import time # For unique filenames and handling file save errors

# --- Configuration ---

# >>> Pixel Size (IMPORTANT for Micrometer Conversion) <<<
# SET THIS MANUALLY based on your 10X objective and camera setup.
# Value should be in micrometers per pixel.
cfg_pixel_size_um = 0.879 # <<< EXAMPLE VALUE - ADJUST AS NEEDED >>>

# >>> Frame Interval (IMPORTANT for Velocity Calculation) <<<
# Set this manually based on your imaging setup.
# Value should be in seconds per frame.
cfg_frame_interval_sec = 300.0 # 5 minutes = 300 seconds

# Minimum track length filter (number of time points/frames)
# Set to 2 for basic displacement, 3+ recommended for persistence/meaningful velocity.
min_track_length_filter = 3

# Output file prefix (base name for output files - will be prepended with input filename)
# This is now set dynamically in the main block based on the input file name.

# --- Expected Input Column Names (Case-sensitive) ---
input_col_track_id = 'track n'
input_col_frame = 'slice n'
input_col_x = 'X'
input_col_y = 'Y'

# --- Internal Column Names (Used after loading and renaming) ---
internal_col_track_id = 'particle'
internal_col_frame = 'frame'
internal_col_x = 'x'
internal_col_y = 'y'
internal_col_global_track_id = 'global_track_id'
internal_col_condition = 'condition'

# Columns to calculate and plot statistics for
metrics_to_plot = [
    'mean_velocity_um_min',
    'directionality',
    'persistence',
]

# --- Helper Functions ---

def setup_logging(level=logging.INFO):
    """Sets up basic logging to the console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    # Setting this option globally might affect other parts if pandas behavior changes.
    # Consider setting it locally if issues arise elsewhere.
    try:
        pd.set_option('future.no_silent_downcasting', True)
    except Exception as e:
        logger.warning(f"Could not set pandas option 'future.no_silent_downcasting': {e}")
    return logging.getLogger(__name__)

def get_excel_path():
    """Uses Tkinter dialog to get the input Excel file path."""
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    logger.info("Please select the input Excel file (.xlsx)...")
    excel_path = filedialog.askopenfilename(
        title="Select Input Excel File",
        filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
    )
    root.destroy()
    if not excel_path: logger.error("No input file selected. Exiting."); sys.exit()
    logger.info(f"Selected input file: {excel_path}")
    return excel_path

def get_output_dir():
    """Uses Tkinter dialog to get the output directory path."""
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    logger.info("Please select the directory where output files should be saved...")
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    root.destroy()
    if not output_dir: logger.error("No output directory selected. Exiting."); sys.exit()
    logger.info(f"Selected output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_paths():
    """Combines calls to get excel path and output directory."""
    excel_path = get_excel_path()
    output_dir = get_output_dir()
    return excel_path, output_dir

# <<< REWRITTEN FUNCTION: load_excel_data (Handles Empty Row After Header + Debug) >>>
def load_excel_data(excel_path):
    """
    Loads data from all sheets in an Excel file, detecting arbitrarily placed
    track data blocks based on the header 'track n'. Assumes 4 columns per
    block ('track n', 'slice n', 'X', 'Y') and an empty row below the header.
    Identifies block end by looking for breaks in the 'slice n' sequence.
    Renames columns, handles data types, and assigns conditions based on sheet names.
    """
    logger.info(f"Loading data from all sheets in {os.path.basename(excel_path)}...")
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at '{excel_path}'. Exiting."); sys.exit()
    except Exception as e:
        logger.error(f"Error reading Excel file structure '{excel_path}': {e}"); sys.exit()

    if not sheet_names:
        logger.error(f"No sheets found in Excel file '{excel_path}'. Exiting."); sys.exit()

    all_tracks_list = [] # List to hold DataFrames for each valid track block found

    # Define expected headers for a block and the rename mapping
    expected_headers = [input_col_track_id, input_col_frame, input_col_x, input_col_y]
    num_expected_cols = len(expected_headers)
    rename_dict = {
        input_col_track_id: internal_col_track_id,
        input_col_frame: internal_col_frame,
        input_col_x: internal_col_x,
        input_col_y: internal_col_y,
    }
    # Columns to convert to numeric after renaming
    cols_to_convert_numeric = [internal_col_track_id, internal_col_frame, internal_col_x, internal_col_y]
    # Essential columns that cannot be NaN after conversion
    essential_numeric_cols = [internal_col_track_id, internal_col_frame, internal_col_x, internal_col_y]

    # Process each sheet
    for sheet_name in sheet_names:
        logger.info(f" Processing sheet: '{sheet_name}'")
        try:
            # Read the entire sheet, keep empty strings, read as string initially
            df_sheet_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, keep_default_na=False, dtype=str)
            # Replace empty strings with NaN for easier processing later
            df_sheet_raw = df_sheet_raw.replace('', np.nan).infer_objects(copy=False)
        except Exception as e:
            logger.error(f"  Error reading sheet '{sheet_name}': {e}. Skipping.")
            continue

        if df_sheet_raw.empty:
            logger.warning(f"  Skipping sheet '{sheet_name}': Sheet is empty.")
            continue

        # --- Find all potential block starting points ---
        block_starts = [] # List of (row_index, col_index) tuples
        potential_header_indices = {} # Store potential header row indices to avoid re-checking

        # Iterate through all cells to find potential 'track n' headers
        for r_idx in range(df_sheet_raw.shape[0]):
            for c_idx in range(df_sheet_raw.shape[1]):
                cell_value = df_sheet_raw.iloc[r_idx, c_idx]
                if pd.notna(cell_value) and str(cell_value).strip() == input_col_track_id:
                    # Check if the next columns match the rest of the expected headers
                    if c_idx + num_expected_cols <= df_sheet_raw.shape[1]:
                        potential_headers = df_sheet_raw.iloc[r_idx, c_idx : c_idx + num_expected_cols].astype(str).tolist()
                        potential_headers_stripped = [h.strip() for h in potential_headers]
                        if potential_headers_stripped == expected_headers:
                            block_starts.append((r_idx, c_idx))
                            potential_header_indices[r_idx] = True # Mark this row as containing headers
                            logger.debug(f"    Found potential block start at ({r_idx}, {c_idx})")


        if not block_starts:
             logger.warning(f"  Skipping sheet '{sheet_name}': No valid header blocks found.")
             continue

        logger.info(f"  Found {len(block_starts)} potential track blocks starting at: {block_starts}")

        # --- Extract and process data for each block ---
        processed_blocks_in_sheet = 0
        for header_row_idx, start_col_idx in block_starts:
            logger.debug(f"    Processing block starting at ({header_row_idx}, {start_col_idx})...")
            block_col_indices = list(range(start_col_idx, start_col_idx + num_expected_cols))

            # --- REVISED Block End Detection Logic (v4 - Skip empty row after header) ---
            # Find the end row of the block based on the 'slice n'/frame column
            # Start checking from TWO rows below the header to skip the empty row
            current_row_for_check = header_row_idx + 2
            end_row_idx = current_row_for_check # Initialize end row index assuming at least one data row exists

            # Get the column index for the frame column within the raw data block
            try:
                frame_col_index_in_block = expected_headers.index(input_col_frame)
                frame_col_raw_index = start_col_idx + frame_col_index_in_block
            except ValueError:
                 logger.error(f"    Critical error: Expected header '{input_col_frame}' not found in expected_headers list {expected_headers}. Skipping block at ({header_row_idx}, {start_col_idx}).")
                 continue

            while current_row_for_check < df_sheet_raw.shape[0]:
                # Stop if we hit another row that we identified as a header row
                if current_row_for_check in potential_header_indices:
                    logger.debug(f"      Stopping row search at {current_row_for_check}: Encountered another header row.")
                    break

                # Check the frame column value in the current row
                cell_value_raw = df_sheet_raw.iloc[current_row_for_check, frame_col_raw_index]
                cell_value_str = str(cell_value_raw).strip() # Convert to string and strip

                # Check if the stripped string is empty
                if not cell_value_str:
                    logger.debug(f"      Stopping block search at row {current_row_for_check}: Frame value is empty string.")
                    break

                # Try converting the stripped string to numeric
                numeric_value = pd.to_numeric(cell_value_str, errors='coerce')

                if pd.isna(numeric_value):
                     logger.debug(f"      Stopping block search at row {current_row_for_check}: Frame value '{cell_value_str}' could not be converted to numeric.")
                     break # Stop if frame number is missing or non-numeric

                # If conversion succeeded and is not NaN, this row is part of the block. Update the end_row_idx
                end_row_idx = current_row_for_check + 1 # The block extends up to this row (exclusive)
                current_row_for_check += 1
            # --- END REVISED Block End Detection Logic (v4) ---

            # The block data runs from header_row_idx + 2 up to (but not including) end_row_idx
            first_data_row_idx = header_row_idx + 2
            logger.debug(f"      Block data identified from row {first_data_row_idx} to {end_row_idx - 1} based on frame column.")

            # Select the columns and rows for this block
            if end_row_idx <= first_data_row_idx: # Check if any data rows were found
                logger.warning(f"    Skipping block at ({header_row_idx}, {start_col_idx}): No valid data rows found starting from row {first_data_row_idx} based on frame column check.")
                continue

            # Extract data rows, use original headers for initial column names
            # Slice starts from first_data_row_idx and goes up to end_row_idx (exclusive)
            df_block = df_sheet_raw.iloc[first_data_row_idx : end_row_idx, block_col_indices].copy()
            df_block.columns = expected_headers

            # --- Add Debugging ---
            logger.debug(f"      Extracted df_block (head):\n{df_block.head().to_string()}")
            logger.debug(f"      Extracted df_block (tail):\n{df_block.tail().to_string()}")
            # --- End Debugging ---


            # --- Process the extracted block ---
            # Rename columns to internal names
            df_block = df_block.rename(columns=rename_dict)

            # Convert specified columns to numeric, coercing errors to NaN
            for col in cols_to_convert_numeric:
                if col in df_block.columns:
                    # Replace comma decimal separator if present before converting
                    if df_block[col].dtype == 'object': # Only apply to string-like columns
                        df_block[col] = df_block[col].str.replace(',', '.', regex=False)
                    df_block[col] = pd.to_numeric(df_block[col], errors='coerce')


            # Drop rows where any essential numeric column ended up as NaN AFTER conversion
            initial_rows = len(df_block)
            df_block.dropna(subset=essential_numeric_cols, inplace=True)
            rows_dropped = initial_rows - len(df_block)
            if rows_dropped > 0:
                 logger.debug(f"      Dropped {rows_dropped} internal rows from block at ({header_row_idx}, {start_col_idx}) due to NaNs in essential columns post-conversion.")

            if df_block.empty:
                logger.debug(f"      Skipping block at ({header_row_idx}, {start_col_idx}): No valid data after cleaning.")
                continue

            # --- Check Frame Sequence Contiguity ---
            # Convert frame to int first
            try:
                df_block[internal_col_frame] = df_block[internal_col_frame].astype(int)
            except Exception as e_int:
                 logger.error(f"    Error converting frame column to integer for block at ({header_row_idx}, {start_col_idx}): {e_int}. Skipping block.")
                 continue

            df_block = df_block.sort_values(by=internal_col_frame)
            frame_diff = df_block[internal_col_frame].diff()
            # Find the first index where the difference is not 1 (after the first row, index might not be sequential)
            non_contiguous_indices = frame_diff.index[frame_diff != 1]
            if not non_contiguous_indices.empty:
                # Get the first index location where the break occurs
                first_break_index = non_contiguous_indices[0]
                # Find the integer position of this index in the sorted dataframe
                break_iloc = df_block.index.get_loc(first_break_index)
                if break_iloc > 0: # Ensure the break isn't the very first diff (which is NaN)
                    logger.warning(f"    Block at ({header_row_idx}, {start_col_idx}) has non-contiguous frame sequence starting around frame {df_block.loc[first_break_index, internal_col_frame]}. Truncating block.")
                    # Keep rows from the start up to the break point (exclusive)
                    df_block = df_block.iloc[:break_iloc]

            if df_block.empty:
                 logger.warning(f"    Skipping block at ({header_row_idx}, {start_col_idx}): Block became empty after checking frame contiguity.")
                 continue
            # --- End Frame Sequence Check ---


            # Ensure integer types for particle ID
            try:
                # Get the particle ID (should be constant within a valid block)
                particle_id_val = df_block[internal_col_track_id].unique()
                if len(particle_id_val) > 1:
                    valid_ids = [pid for pid in particle_id_val if pd.notna(pid)]
                    if not valid_ids:
                         logger.error(f"    Block at ({header_row_idx}, {start_col_idx}) in sheet '{sheet_name}' has no valid particle IDs after cleaning. Skipping block.")
                         continue
                    if len(valid_ids) > 1:
                         logger.warning(f"    Block at ({header_row_idx}, {start_col_idx}) in sheet '{sheet_name}' has multiple particle IDs ({valid_ids}). Using the first: {valid_ids[0]}.")
                    particle_id = int(float(valid_ids[0])) # Convert to float first for safety
                elif pd.isna(particle_id_val[0]):
                     logger.error(f"    Block at ({header_row_idx}, {start_col_idx}) in sheet '{sheet_name}' has only NaN particle ID after cleaning. Skipping block.")
                     continue
                else:
                    particle_id = int(float(particle_id_val[0])) # Convert to float first for safety

                df_block[internal_col_track_id] = particle_id # Ensure constant integer ID

            except Exception as e:
                logger.error(f"    Error converting track ID for block at ({header_row_idx}, {start_col_idx}) in sheet '{sheet_name}': {e}. Skipping block.")
                continue

            # Add condition (sheet name) and unique global track ID
            df_block[internal_col_condition] = sheet_name
            # Use row/col in ID to make it unique if particle numbers repeat in a sheet
            df_block[internal_col_global_track_id] = f"{sheet_name}_r{header_row_idx}c{start_col_idx}_{particle_id}"

            # Select only the necessary renamed columns before appending
            final_block_cols = [internal_col_global_track_id, internal_col_condition,
                                internal_col_track_id, internal_col_frame,
                                internal_col_x, internal_col_y]
            all_tracks_list.append(df_block[final_block_cols])
            processed_blocks_in_sheet += 1
            logger.info(f"    Successfully processed track {particle_id} from block at ({header_row_idx}, {start_col_idx}). Final length: {len(df_block)}")

        logger.info(f"  Finished processing sheet '{sheet_name}'. Found {processed_blocks_in_sheet} valid track blocks.")


    if not all_tracks_list:
        logger.error("No valid track data found in any sheet after processing blocks. Exiting.")
        sys.exit()

    # Concatenate all processed track DataFrames
    combined_df = pd.concat(all_tracks_list, ignore_index=True)
    n_unique_tracks = combined_df[internal_col_global_track_id].nunique()
    logger.info(f"Successfully combined data from {len(all_tracks_list)} track blocks across all sheets.")
    logger.info(f"Total unique tracks loaded: {n_unique_tracks}")
    logger.info(f"Total valid data points loaded: {len(combined_df)}")
    return combined_df
# <<< END REWRITTEN FUNCTION >>>


def filter_tracks_by_length(df, min_len):
    """Filters the combined dataframe to keep tracks >= min_len."""
    logger.info(f"Filtering tracks: Keeping tracks with length >= {min_len} frames...")
    id_col = internal_col_global_track_id
    frame_col = internal_col_frame

    if id_col not in df.columns:
        logger.error(f" '{id_col}' column missing. Cannot filter by track length.")
        return pd.DataFrame(), 0 # Return empty DataFrame and 0 tracks

    n_total_tracks = df[id_col].nunique()
    logger.info(f"Total unique tracks found before length filtering: {n_total_tracks}")
    logger.info(f"Total data points before length filtering: {len(df)}")

    # Calculate track length for each track using transform
    track_lengths = df.groupby(id_col)[frame_col].transform('count')
    df['track_length'] = track_lengths # Add length column to the original df

    # Perform the filtering
    filtered_df = df[df['track_length'] >= min_len].copy() # Use .copy() to avoid SettingWithCopyWarning

    n_filtered_tracks = filtered_df[id_col].nunique()
    n_removed_tracks = n_total_tracks - n_filtered_tracks

    if n_filtered_tracks == 0:
         logger.warning(f"No tracks remained after filtering (min length = {min_len}).")
    else:
         logger.info(f"Tracks remaining after filtering: {n_filtered_tracks} (Removed {n_removed_tracks})")
         logger.info(f"Data points remaining: {len(filtered_df)}")

    return filtered_df, n_filtered_tracks

# <<< VERIFIED FUNCTION: calculate_motility_metrics >>>
def calculate_motility_metrics(tracks_df, frame_interval_s, pixel_size_um=None):
    """
    Calculates velocity (um/min), directionality, and persistence for each track
    based on X, Y coordinates.

    Args:
        tracks_df (pd.DataFrame): DataFrame containing filtered track data with columns
                                   like 'global_track_id', 'frame', 'x', 'y'.
        frame_interval_s (float): Time interval between frames in seconds.
        pixel_size_um (float, optional): Pixel size in micrometers. If None, velocity
                                         will be NaN. Defaults to None.

    Returns:
        tuple: (pd.DataFrame containing per-track summary stats, str label for velocity units)
    """
    logger.info("Calculating motility metrics...")
    id_col = internal_col_global_track_id
    frame_col = internal_col_frame
    x_col = internal_col_x
    y_col = internal_col_y
    condition_col = internal_col_condition
    particle_col = internal_col_track_id # Original particle ID

    # Get unique track identifiers, original particle ID, and conditions for merging later
    # Include track_length which was added during filtering
    results_df = tracks_df[[id_col, particle_col, condition_col, 'track_length']].drop_duplicates().copy()
    # Rename 'track_length' to 'track_length_frames' for clarity in the final output
    results_df = results_df.rename(columns={'track_length': 'track_length_frames'})


    # Ensure coordinates are numeric (should be done in load, but robust check)
    tracks_df[x_col] = pd.to_numeric(tracks_df[x_col], errors='coerce')
    tracks_df[y_col] = pd.to_numeric(tracks_df[y_col], errors='coerce')
    tracks_df.dropna(subset=[x_col, y_col], inplace=True)

    # --- Calculate metrics requiring step-by-step data ---
    # Sort by track ID and frame number is crucial for difference calculations
    tracks_df = tracks_df.sort_values(by=[id_col, frame_col])

    # Calculate frame-to-frame differences within each track group
    grouped = tracks_df.groupby(id_col)
    tracks_df['dx'] = grouped[x_col].diff()
    tracks_df['dy'] = grouped[y_col].diff()
    # Calculate distance in pixels for each step
    tracks_df['distance_pix'] = np.sqrt(tracks_df['dx']**2 + tracks_df['dy']**2)

    # Calculate instantaneous velocity from displacement in um/min
    velocity_col_name = 'velocity_um_min' # Define the column name we will use
    velocity_unit_label = "Velocity (Unknown Units)" # Default label
    if pixel_size_um is not None and frame_interval_s is not None and frame_interval_s > 0:
        logger.info(f"  Calculating velocity ({velocity_col_name}) using pixel size {pixel_size_um} um/pix and interval {frame_interval_s} s/frame.")
        frame_interval_min = frame_interval_s / 60.0
        if frame_interval_min > 1e-9:
            tracks_df[velocity_col_name] = (tracks_df['distance_pix'] * pixel_size_um) / frame_interval_min
            velocity_unit_label = "Velocity (µm/min)"
        else:
            logger.warning(" Frame interval is zero or effectively zero in minutes. Cannot calculate um/min velocity.")
            tracks_df[velocity_col_name] = np.nan
    else:
        logger.warning(f"  Cannot calculate velocity in um/min (pixel size = {pixel_size_um}, frame interval = {frame_interval_s}). Mean velocity will be NaN.")
        tracks_df[velocity_col_name] = np.nan

    # --- Calculate per-track summaries ---
    per_track_data = []
    for track_id, track_df in tracks_df.groupby(id_col):
        if len(track_df) < 2:
            logger.debug(f"Skipping track {track_id}: less than 2 points ({len(track_df)}).")
            continue

        # Drop rows with NaN displacement/velocity for calculations within this track
        cols_to_check_na = ['dx', 'dy', 'distance_pix', velocity_col_name]
        cols_to_check_na = [c for c in cols_to_check_na if c in track_df.columns]
        track_df_valid_steps = track_df.dropna(subset=cols_to_check_na)

        if len(track_df_valid_steps) < 1:
             logger.debug(f"Skipping track {track_id}: less than 1 valid step after NaN drop ({len(track_df_valid_steps)}).")
             continue

        # --- Mean Velocity ---
        mean_velocity = track_df_valid_steps[velocity_col_name].mean(skipna=True) if velocity_col_name in track_df_valid_steps else np.nan

        # --- Directionality ---
        # Total path length (sum of step distances in pixels)
        total_path_pix = track_df_valid_steps['distance_pix'].sum(skipna=True)

        # Net Displacement (distance between first and last point of the *original* track_df)
        # Find first and last non-NaN index for x coordinate in the original track data
        first_valid_idx = track_df[x_col].first_valid_index()
        last_valid_idx = track_df[x_col].last_valid_index()

        net_disp_pix = np.nan
        directionality = np.nan # Default to NaN if calculation fails

        # Check if valid start and end indices were found and are different
        if first_valid_idx is not None and last_valid_idx is not None and first_valid_idx != last_valid_idx:
            start_pos = track_df.loc[first_valid_idx, [x_col, y_col]]
            end_pos = track_df.loc[last_valid_idx, [x_col, y_col]]

            # Ensure start/end pos coordinates are valid numbers
            if pd.notna(start_pos[x_col]) and pd.notna(start_pos[y_col]) and pd.notna(end_pos[x_col]) and pd.notna(end_pos[y_col]):
                # Calculate Euclidean distance
                net_disp_pix = math.sqrt((end_pos[x_col] - start_pos[x_col])**2 + (end_pos[y_col] - start_pos[y_col])**2)
                # Calculate Directionality = Net Displacement / Total Path Length
                # Avoid division by zero if total path is zero (stationary track)
                if total_path_pix > 1e-9:
                    directionality = net_disp_pix / total_path_pix
                else:
                    # If total path is zero, net displacement must also be zero (or close due to float precision)
                    # Set directionality to 0 for stationary tracks.
                    directionality = 0.0
                    if net_disp_pix > 1e-9 : # Should not happen if total_path_pix is 0
                         logger.warning(f"Track {track_id}: Net displacement > 0 but total path = 0. Check data.")

            else:
                 logger.debug(f"Track {track_id}: Invalid start/end coordinates ({start_pos}, {end_pos}) for net displacement.")
                 # Set to 0 if coordinates were invalid
                 net_disp_pix = 0.0
                 directionality = 0.0
        else:
            # Handle tracks with only one valid point or where start/end index are the same
             logger.debug(f"Track {track_id}: Cannot calculate net displacement (start_idx={first_valid_idx}, end_idx={last_valid_idx}). Setting to 0.")
             net_disp_pix = 0.0
             directionality = 0.0 # If no net movement detected or calculable

        # --- Persistence (Mean Cosine of Angles between Consecutive Steps) ---
        persistence = np.nan
        if len(track_df_valid_steps) >= 2: # Need at least 3 points (2 valid steps)
            dx = track_df_valid_steps['dx'].values; dy = track_df_valid_steps['dy'].values
            dx_t = dx[1:]; dy_t = dy[1:]
            dx_tm1 = dx[:-1]; dy_tm1 = dy[:-1]

            dot_prods = dx_tm1 * dx_t + dy_tm1 * dy_t
            mag_t = np.sqrt(dx_t**2 + dy_t**2)
            mag_tm1 = np.sqrt(dx_tm1**2 + dy_tm1**2)

            valid_indices = (mag_t > 1e-9) & (mag_tm1 > 1e-9)

            if np.any(valid_indices):
                cos_angles_valid = dot_prods[valid_indices] / (mag_t[valid_indices] * mag_tm1[valid_indices])
                cos_angles_valid = np.clip(cos_angles_valid, -1.0, 1.0)
                persistence = np.nanmean(cos_angles_valid)
            else:
                 logger.debug(f"Track {track_id}: No valid consecutive steps with non-zero magnitude found for persistence calculation.")
                 # Keep persistence as NaN if uncalculable
        else:
             logger.debug(f"Track {track_id}: Not enough valid steps ({len(track_df_valid_steps)}) to calculate persistence.")


        # Store results for this track
        per_track_data.append({
            id_col: track_id,
            'mean_velocity_um_min': mean_velocity,
            'directionality': directionality,
            'persistence': persistence,
            'total_path_length_pix': total_path_pix,
            'net_displacement_pix': net_disp_pix,
        })

    # Create DataFrame from the collected track summaries
    motility_summary_df = pd.DataFrame(per_track_data)
    logger.info(f"Calculated motility metrics for {len(motility_summary_df)} tracks.")

    # Merge summary stats back with the unique track/condition/length info DataFrame
    results_df = pd.merge(results_df, motility_summary_df, on=id_col, how='left')

    # Add physical unit conversions if possible
    if pixel_size_um is not None:
        results_df['total_path_length_um'] = results_df['total_path_length_pix'] * pixel_size_um
        results_df['net_displacement_um'] = results_df['net_displacement_pix'] * pixel_size_um

    return results_df, velocity_unit_label
# <<< END VERIFIED FUNCTION >>>


def plot_distributions(stats_df, metrics, unit_labels, output_dir, prefix, plot_type='violin'):
    """
    Plots distributions of specified metrics grouped by 'condition'
    using either violin or box plots.
    """
    plot_type_name = plot_type.capitalize()
    metrics_to_actually_plot = [m for m in metrics if m != 'mean_pixel_value'] # Ensure pixel value is excluded

    logger.info(f"Generating {plot_type_name} distribution plots for metrics: {', '.join(metrics_to_actually_plot)}...")
    if stats_df.empty or not metrics_to_actually_plot:
        logger.warning(f"Skipping {plot_type_name} plots: No data or no valid metrics specified.")
        return

    condition_col = internal_col_condition

    if condition_col not in stats_df.columns:
        logger.error(f"Skipping distribution plots: '{condition_col}' column not found in summary data.")
        return

    valid_metrics = [m for m in metrics_to_actually_plot if m in stats_df.columns and stats_df[m].notna().any()]
    if not valid_metrics:
        logger.warning(f"Skipping {plot_type_name} plots: None of the specified metrics ({metrics_to_actually_plot}) have valid data.")
        return

    num_metrics = len(valid_metrics)
    n_cols = min(3, num_metrics)
    n_rows = (num_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    plot_idx = 0
    for metric in valid_metrics:
        ax = axes[plot_idx]
        try:
            if plot_type == 'violin':
                sns.violinplot(data=stats_df, x=condition_col, y=metric, ax=ax, palette='viridis',
                               hue=condition_col, legend=False,
                               inner='quartile', cut=0, density_norm='width')
            elif plot_type == 'box':
                sns.boxplot(data=stats_df, x=condition_col, y=metric, ax=ax, palette='viridis',
                            hue=condition_col, legend=False,
                            showfliers=False)
            else:
                logger.warning(f"Unknown plot_type '{plot_type}'. Skipping plot for {metric}.")
                continue

            title = metric.replace('_', ' ').title()
            if metric == 'mean_velocity_um_min':
                title = f"Mean {unit_labels.get(metric, 'Velocity (Units Unknown)')}"

            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Condition", fontsize=9)
            ax.set_ylabel("Value", fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)

            if metric == 'directionality': ax.set_ylim(-0.05, 1.05)
            if metric == 'persistence': ax.set_ylim(-1.05, 1.05)

            plot_idx += 1

        except Exception as plot_e:
            logger.error(f"Error plotting {plot_type} for {metric}: {plot_e}")


    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    if plot_idx > 0:
        plt.suptitle(f"Distribution of Motility Metrics by Condition ({plot_type_name} Plots)", y=1.02, fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])

        plot_filename = os.path.join(output_dir, f"{prefix}_Motility_Distributions_{plot_type_name}.png")
        try:
            fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
            logger.info(f"✅ {plot_type_name} distribution plot saved to {os.path.basename(plot_filename)}")
        except Exception as e:
            logger.error(f"Error saving {plot_type_name} distribution plot '{os.path.basename(plot_filename)}': {e}")
    else:
        logger.warning(f"No {plot_type_name} plots were successfully generated.")

    plt.close(fig)

# <<< MODIFIED FUNCTION: plot_trajectories_by_condition >>>
def plot_trajectories_by_condition(track_data_df, output_dir, prefix, colormap_name='viridis'):
    """
    Plots all cell trajectories (X vs Y) with each condition in a separate subplot.
    Uses a colormap to indicate time along the track.

    Args:
        track_data_df (pd.DataFrame): DataFrame containing filtered track data, must include
                                      'global_track_id', 'condition', 'x', 'y', and 'frame' columns.
        output_dir (str): Path to the directory to save the plot.
        prefix (str): Prefix for the output plot filename.
        colormap_name (str): Name of the matplotlib colormap to use (e.g., 'viridis', 'plasma', 'magma').
    """
    logger.info(f"Generating time-colored trajectory plot by condition using '{colormap_name}' colormap...")
    id_col = internal_col_global_track_id
    condition_col = internal_col_condition
    x_col = internal_col_x
    y_col = internal_col_y
    frame_col = internal_col_frame

    required_cols = [id_col, condition_col, x_col, y_col, frame_col]
    if not all(col in track_data_df.columns for col in required_cols):
        logger.error(f"Skipping trajectory plot: Missing one or more required columns: {required_cols}")
        return
    if track_data_df.empty:
        logger.warning("Skipping trajectory plot: Input DataFrame is empty.")
        return

    conditions = sorted(track_data_df[condition_col].unique())
    n_conditions = len(conditions)
    if n_conditions == 0:
        logger.warning("Skipping trajectory plot: No conditions found in data.")
        return

    n_cols = int(np.ceil(np.sqrt(n_conditions)))
    n_rows = int(np.ceil(n_conditions / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False, sharex=False, sharey=False)
    axes = axes.flatten()

    # Get the specified colormap
    try:
        cmap = plt.get_cmap(colormap_name)
    except ValueError:
        logger.warning(f"Colormap '{colormap_name}' not found. Defaulting to 'viridis'.")
        cmap = plt.get_cmap('viridis')

    # Find global min/max frame across all data for consistent color scaling
    global_min_frame = track_data_df[frame_col].min()
    global_max_frame = track_data_df[frame_col].max()
    # Handle case where all tracks might have the same frame number (unlikely but possible)
    if global_max_frame == global_min_frame:
        frame_norm = plt.Normalize(vmin=global_min_frame - 1, vmax=global_max_frame + 1) # Avoid zero range
    else:
        frame_norm = plt.Normalize(vmin=global_min_frame, vmax=global_max_frame)


    # Plot trajectories for each condition
    for i, condition in enumerate(conditions):
        ax = axes[i]
        condition_data = track_data_df[track_data_df[condition_col] == condition]
        n_tracks_in_condition = condition_data[id_col].nunique()

        if condition_data.empty:
            logger.warning(f"  No data found for condition '{condition}'. Skipping subplot.")
            ax.set_title(f"{condition} (No Tracks)", fontsize=10)
            ax.text(0.5, 0.5, 'No Tracks', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        logger.info(f"  Plotting {n_tracks_in_condition} trajectories for condition: {condition}")

        # --- Plotting logic changed to use scatter with colormap ---
        for track_id, track_df in condition_data.groupby(id_col):
            track_df_sorted = track_df.sort_values(by=frame_col)
            x_coords = track_df_sorted[x_col].values
            y_coords = track_df_sorted[y_col].values
            frames = track_df_sorted[frame_col].values

            # Get colors from colormap based on normalized frames using global min/max
            point_colors = cmap(frame_norm(frames))

            # Plot using scatter with time-based coloring
            # Adjust marker size (s) and alpha as needed
            scatter_plot = ax.scatter(x_coords, y_coords, c=point_colors, s=5, alpha=0.7, edgecolors='none', label=f"Track {track_id}")

            # Optionally: Add faint lines connecting points (can make plot busy)
            # ax.plot(x_coords, y_coords, linestyle='-', linewidth=0.3, color='gray', alpha=0.4)

        # --- End of changed plotting logic ---

        ax.set_title(f"Condition: {condition} ({n_tracks_in_condition} tracks)", fontsize=10)
        ax.set_xlabel("X Position (pixels)", fontsize=9)
        ax.set_ylabel("Y Position (pixels)", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add a colorbar to *each* subplot for clarity
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=frame_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Frame Number', size=8)
        cbar.ax.tick_params(labelsize=7)


    # Hide any unused subplots
    for j in range(n_conditions, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Cell Trajectories by Condition (Color = Time)", fontsize=16, y=1.0) # Adjust y if needed
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout

    # Save the figure
    plot_filename = os.path.join(output_dir, f"{prefix}_Trajectories_TimeColored.png") # Changed filename
    try:
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Time-colored trajectory plot saved to {os.path.basename(plot_filename)}")
    except Exception as e:
        logger.error(f"Error saving time-colored trajectory plot '{os.path.basename(plot_filename)}': {e}")
    plt.close(fig) # Close the figure
# <<< END MODIFIED FUNCTION >>>


# --- Main Execution Block ---
if __name__ == "__main__":
    logger = setup_logging(level=logging.DEBUG) # <<< Set logging to DEBUG for more info >>>
    logger.info("--- Starting Cell Motility Analysis Script ---")

    # --- Configuration Check ---
    pixel_size_provided = True
    if cfg_pixel_size_um is None or cfg_pixel_size_um <= 0:
        logger.warning("cfg_pixel_size_um is not positive or not set. Velocity/distance in 'um' will not be calculated.")
        cfg_pixel_size_um = None # Ensure it's None if invalid
        pixel_size_provided = False
    if cfg_frame_interval_sec <= 0:
        logger.error("cfg_frame_interval_sec must be positive. Cannot calculate velocity. Exiting.")
        sys.exit()
    if min_track_length_filter < 2:
        logger.warning(f"min_track_length_filter is {min_track_length_filter}. Setting to 2 for basic displacement calculations.")
        min_track_length_filter = 2
    if min_track_length_filter < 3:
         logger.warning(f"min_track_length_filter is {min_track_length_filter}. Persistence calculation requires >= 3 points (2 steps) and may result in NaNs.")


    # 1. Get Input File and Output Directory Paths
    try:
        excel_file_path, output_directory = get_paths()
    except Exception as e:
        logger.error(f"Error getting file/directory paths: {e}. Ensure GUI environment is available or modify script for command-line args. Exiting.")
        sys.exit()

    # Define output prefix based on input filename (remove extension)
    base_filename = os.path.basename(excel_file_path)
    output_prefix = os.path.splitext(base_filename)[0] + "_Analysis" # Append suffix
    logger.info(f"Using output file prefix: {output_prefix}")

    # 2. Load and Combine Data from Excel Sheets (using the REWRITTEN function)
    combined_data = load_excel_data(excel_file_path)

    if not combined_data.empty:
        # 3. Filter Tracks by Minimum Length
        filtered_data, num_filt_tracks = filter_tracks_by_length(combined_data, min_track_length_filter)

        if not filtered_data.empty and num_filt_tracks > 0:
            # 4. Calculate Motility Metrics
            motility_results_df, velocity_units_label = calculate_motility_metrics(
                filtered_data.copy(), cfg_frame_interval_sec, cfg_pixel_size_um
            )

            # 5. Save Motility Results to CSV
            output_csv_path = os.path.join(output_directory, f"{output_prefix}_PerTrack_Stats.csv")
            logger.info(f"Attempting to save per-track motility stats to: {output_csv_path}")
            try:
                # *** UPDATED cols_to_save (no pixel value) ***
                cols_to_save = [
                    internal_col_condition, internal_col_track_id, internal_col_global_track_id,
                    'mean_velocity_um_min', 'directionality', 'persistence',
                    'total_path_length_pix', 'net_displacement_pix',
                    'total_path_length_um', 'net_displacement_um', # Add um values if calculated
                    'track_length_frames'
                 ]
                cols_exist = [c for c in cols_to_save if c in motility_results_df.columns]

                motility_results_df[cols_exist].to_csv(output_csv_path, index=False, float_format='%.4f')
                logger.info(f"✅ Per-track stats saved successfully to {os.path.basename(output_csv_path)}.")

            except PermissionError:
                 timestamp = time.strftime("%Y%m%d%H%M%S")
                 alt_file = os.path.join(output_directory, f"{output_prefix}_PerTrack_Stats_{timestamp}.csv")
                 logger.warning(f"Permission denied writing to {os.path.basename(output_csv_path)}. It might be open. Trying to save to: {os.path.basename(alt_file)}")
                 try:
                     motility_results_df[cols_exist].to_csv(alt_file, index=False, float_format='%.4f')
                     logger.info(f"✅ Per-track stats saved successfully to alternate file: {os.path.basename(alt_file)}")
                 except Exception as e_alt:
                     logger.error(f"Error saving per-track stats to alternate file {os.path.basename(alt_file)}: {e_alt}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while saving per-track stats: {e}")

            # --- Plotting Section ---
            plot_output_dir = os.path.join(output_directory, "Plots")
            os.makedirs(plot_output_dir, exist_ok=True)

            # 6a. Plot Trajectories by Condition (using filtered_data and new style)
            plot_trajectories_by_condition(filtered_data, plot_output_dir, output_prefix, colormap_name='viridis') # Specify colormap

            # 6b. Plot Motility Distributions (using motility_results_df)
            # *** metrics_to_plot already updated ***
            metrics_for_dist_plot = [m for m in metrics_to_plot if m in motility_results_df.columns]

            unit_labels_for_plot = {'mean_velocity_um_min': velocity_units_label}

            if metrics_for_dist_plot:
                plot_distributions(motility_results_df, metrics_for_dist_plot, unit_labels_for_plot, plot_output_dir, output_prefix, plot_type='violin')
                plot_distributions(motility_results_df, metrics_for_dist_plot, unit_labels_for_plot, plot_output_dir, output_prefix, plot_type='box')
            else:
                logger.warning("No valid metrics found in the results to plot distributions.")

        else:
             logger.error("No tracks remained after length filtering. Cannot calculate metrics or plot.")
    else:
        logger.error("Exiting: No data was loaded successfully from the Excel file.")

    logger.info("--- Script finished ---")
