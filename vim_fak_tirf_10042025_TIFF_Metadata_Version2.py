# import cv2 # Still needed for ROI drawing, thresholding
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys # Import sys for exit
import tifffile # <<< Added for TIFF reading
from tifffile import TiffFile # <<< Import TiffFile for metadata access
import re # <<< Import regular expressions for parsing description tag

# --- CONFIGURATION ---
# >>> IMPORTANT: Update this path to your multi-channel TIFF file <<<
# Use raw string (r'...') or forward slashes ('/') for Windows paths
tiff_path = r'C:\Users\Arun\Desktop\TIRF MCHERRYVIM_GFPFAK_TIFF\MEF KO GFP FAK MCHERRY VIM G4.tiff'
# Example: tiff_path = 'C:/Users/Arun/Desktop/TIRF MCHERRYVIM_GFPFAK_TIFF/MEF KO GFP FAK MCHERRY VIM G4.tiff'


# >>> Channel Indices (Based on C=1 in TCYX order) <<<
vimentin_channel_index = 0 # Assuming Vimentin is the first channel (index 0 of axis 1)
fak_channel_index = 1      # Assuming FAK is the second channel (index 1 of axis 1)

use_interactive_roi = True # Set to True to use interactive, False for predefined below
# Predefined ROIs (only used if use_interactive_roi = False) - Ensure these are valid for your video dimensions
predefined_rois = [(290, 300, 40, 30), (350, 310, 40, 30)] # Example, adjust if needed

# Processing parameters
adaptive_threshold_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
adaptive_threshold_block_size = 11  # Must be odd
adaptive_threshold_constant = 2
smooth_window = 5

# --- Globals for Mouse Callback (if using interactive mode) ---
rois_being_drawn = []
current_roi_start = None
is_drawing = False
frame_display_base = None
frame_copy_dynamic = None
window_name_roi = "Define ROIs (Drag Mouse; ENTER/s=Save, ESC/q=Cancel)" # Define window name here
# --- End Globals ---

# --- INITIALIZATION ---
print(f"Checking TIFF file: {tiff_path}")
if not os.path.exists(tiff_path):
    print(f"Error: TIFF file not found at '{tiff_path}'")
    sys.exit()

print("Loading TIFF stack and attempting to read metadata...")
frame_interval_sec = None # Initialize frame interval
ome_meta = None
ij_meta = None
description = None
image_stack = None

try:
    with tifffile.TiffFile(tiff_path) as tif:
        # Load the image data first
        image_stack = tif.asarray()
        print(f"Successfully loaded image stack.")

        # --- Attempt to get metadata for time interval ---
        try:
            ome_meta = tif.ome_metadata
            if ome_meta:
                 print("Found OME metadata.")
                 # Basic check if it's parsed XML dict, sometimes it's raw XML string
                 if isinstance(ome_meta, str):
                      print("Attempting to parse OME XML string...")
                      try:
                           import xml.etree.ElementTree as ET
                           root = ET.fromstring(ome_meta)
                           # Example: Find DeltaT - requires knowing OME schema structure
                           # This is a simplified example, real parsing might be complex
                           pixels_node = root.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')
                           if pixels_node is not None:
                                plane_node = pixels_node.find('.//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Plane')
                                if plane_node is not None and 'DeltaT' in plane_node.attrib:
                                     delta_t_str = plane_node.attrib['DeltaT']
                                     delta_t_unit = plane_node.attrib.get('DeltaTUnit', 's') # Default to seconds
                                     delta_t = float(delta_t_str)
                                     # Convert to seconds if necessary (e.g., from 'ms')
                                     if delta_t_unit.lower() == 'ms':
                                          frame_interval_sec = delta_t / 1000.0
                                     elif delta_t_unit.lower() == 's':
                                           frame_interval_sec = delta_t
                                     else:
                                          print(f"Warning: Unknown OME DeltaTUnit '{delta_t_unit}', assuming seconds.")
                                          frame_interval_sec = delta_t

                                     if frame_interval_sec is not None:
                                          print(f"Found frame interval in OME metadata (DeltaT): {frame_interval_sec:.4f} sec")

                      except Exception as xml_e:
                           print(f"Could not parse OME XML string: {xml_e}")

        except Exception as e:
            print(f"Could not access OME metadata: {e}") # Non-critical

        try:
            ij_meta = tif.imagej_metadata
            if ij_meta:
                 print("Found ImageJ metadata.")
                 if frame_interval_sec is None and isinstance(ij_meta, dict) and 'finterval' in ij_meta:
                     try:
                         frame_interval_sec = float(ij_meta['finterval'])
                         print(f"Found frame interval in ImageJ metadata (finterval): {frame_interval_sec:.4f} sec")
                     except (ValueError, TypeError):
                         print(f"Could not parse ImageJ 'finterval': {ij_meta['finterval']}")
        except Exception as e:
            print(f"Could not access ImageJ metadata: {e}") # Non-critical

        # Get description tag from first page for fallback parsing
        if frame_interval_sec is None:
             try:
                 # Check if pages exist
                 if tif.pages and len(tif.pages) > 0:
                      description_tag = tif.pages[0].tags.get(270) # Tag ID 270 is ImageDescription
                      description = description_tag.value if description_tag else None
                      if description:
                           print("Found ImageDescription tag. Attempting to parse...")
                           # Example parsing - highly dependent on how software writes it
                           desc_str = str(description) # Ensure it's a string
                           # Look for patterns like "interval=3.0 s" or "frame interval = 3000 ms" etc.
                           # Regex to find numbers possibly followed by 's' or 'ms' after 'interval' or 'deltaT'
                           match = re.search(r"(?:interval|deltaT)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(ms|s)?", desc_str, re.IGNORECASE)
                           if match:
                                value = float(match.group(1))
                                unit = match.group(2)
                                if unit and unit.lower() == 'ms':
                                     frame_interval_sec = value / 1000.0
                                else: # Assume seconds if unit is 's' or missing
                                     frame_interval_sec = value
                                print(f"Attempted parsing ImageDescription, found potential interval: {frame_interval_sec:.4f} sec")
                           else:
                                print("Could not find a recognizable interval pattern in ImageDescription.")
                 else:
                      print("TIFF file has no pages to read ImageDescription from.")
             except Exception as e:
                 print(f"Could not access or parse ImageDescription tag: {e}") # Non-critical

except Exception as e:
    print(f"Error reading TIFF file or metadata: {e}")
    if image_stack is None: # If image loading failed, exit
         sys.exit()
    # If only metadata failed, proceed with image stack if loaded

# Check if image stack was loaded successfully
if image_stack is None:
     print("Error: Image stack could not be loaded. Exiting.")
     sys.exit()


# >>> CRITICAL: Check the actual shape of your loaded TIFF stack <<<
print(f"Loaded TIFF stack with shape: {image_stack.shape}")
print("--> IMPORTANT: Verify the dimension order!")
print(f"--> Script NOW ASSUMES order: (Time, Channel, Height, Width) -> (T, C, Y, X)")
print(f"--> Based on actual shape: {image_stack.shape}")
print(f"--> If incorrect, ADJUST INDEXING in sections marked '### SHAPE-DEPENDENT INDEXING ###'")

# --- Determine dimensions based on ASSUMED shape (T, C, Y, X) ---
### SHAPE-DEPENDENT INDEXING ### (Adjust if shape is different!)
try:
    if image_stack.ndim == 4: # Assuming T, C, Y, X
        num_frames = image_stack.shape[0]       # Time = Axis 0
        num_channels = image_stack.shape[1]     # Channel = Axis 1
        frame_height = image_stack.shape[2]     # Y = Axis 2
        frame_width = image_stack.shape[3]      # X = Axis 3
        print(f"Interpreted as: T={num_frames}, C={num_channels}, H={frame_height}, W={frame_width}")

        # Check if C=2 as expected
        if num_channels != 2:
             print(f"Warning: Expected 2 channels at axis 1, but found {num_channels}. Check channel indices.")
             if max(vimentin_channel_index, fak_channel_index) >= num_channels:
                  raise IndexError(f"Channel index out of bounds. Max requested index: {max(vimentin_channel_index, fak_channel_index)}, Channels found: {num_channels}")
    # Add handling for other dimensions if necessary, e.g., 3D (T, Y, X)
    elif image_stack.ndim == 3: # Example: T, Y, X (single channel)
         num_frames = image_stack.shape[0]
         num_channels = 1
         frame_height = image_stack.shape[1]
         frame_width = image_stack.shape[2]
         print(f"Interpreted as single channel: T={num_frames}, H={frame_height}, W={frame_width}")
         # Force channel indices if only one channel
         vimentin_channel_index = 0
         fak_channel_index = 0 # Or handle error if FAK analysis is expected
    else:
        raise ValueError(f"Unexpected number of dimensions: {image_stack.ndim}. Expected 4D (T, C, Y, X) or 3D (T, Y, X).")

    # Check if assumed channel indices are valid for the channel dimension (axis 1)
    if not (0 <= vimentin_channel_index < num_channels and 0 <= fak_channel_index < num_channels):
         raise IndexError(f"Channel index out of bounds for axis 1. Vimentin: {vimentin_channel_index}, FAK: {fak_channel_index}, Num Channels: {num_channels}")

except IndexError as e:
    print(f"Error interpreting TIFF dimensions based on assumed order: {e}")
    print("Check the printed shape above and adjust the '### SHAPE-DEPENDENT INDEXING ###' logic.")
    sys.exit()
except ValueError as e:
     print(f"Error: {e}")
     sys.exit()


# --- Determine Frame Rate ---
# Use automatically detected interval if found, otherwise fallback
if frame_interval_sec is not None and frame_interval_sec > 0:
     frame_rate = 1.0 / frame_interval_sec
     print(f"Using automatically detected frame interval: {frame_interval_sec:.4f} sec (Rate: {frame_rate:.3f} FPS)")
else:
     print("Warning: Could not automatically detect frame interval from metadata.")
     # Fallback to the manually known interval (3.0 seconds)
     manual_interval = 3.0
     frame_interval_sec = manual_interval # Store for potential future use/reference
     frame_rate = 1.0 / frame_interval_sec
     print(f"Using manually set interval of {manual_interval:.1f} sec (Rate: {frame_rate:.3f} FPS)")
     # --- Alternative: Use placeholder if interval is unknown ---
     # frame_rate = 1.0 # Placeholder
     # print(f"Using placeholder frame rate: {frame_rate:.3f} FPS. Update manually if known.")


print(f"TIFF Info: Frames(T): {num_frames}, Channels: {num_channels}, Resolution: {frame_width}x{frame_height}")

# Define rois variable here; it will be overwritten by interactive selection if enabled
rois = predefined_rois

# --- Define Mouse Callback Function (no changes needed here) ---
def mouse_callback_roi_user(event, x, y, flags, param):
    global is_drawing, current_roi_start, rois_being_drawn, frame_display_base, frame_copy_dynamic

    if event == cv2.EVENT_LBUTTONDOWN:
        if frame_display_base is not None: # Ensure base frame exists
            frame_copy_dynamic = frame_display_base.copy()
            is_drawing = True
            current_roi_start = (x, y)
        else:
            print("Error: Base frame not initialized for drawing.")

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing and frame_display_base is not None:
            frame_copy_dynamic = frame_display_base.copy() # Re-copy base to clear old green box
            cv2.rectangle(frame_copy_dynamic, current_roi_start, (x, y), (0, 255, 0), 1) # Green dynamic box
            cv2.imshow(window_name_roi, frame_copy_dynamic) # window_name_roi needs to be accessible or passed

    elif event == cv2.EVENT_LBUTTONUP:
        if is_drawing and frame_display_base is not None:
            is_drawing = False
            x1, y1 = current_roi_start
            x2, y2 = x, y
            roi_x, roi_y = min(x1, x2), min(y1, y2)
            roi_w, roi_h = abs(x1 - x2), abs(y1 - y2)

            if roi_w > 0 and roi_h > 0:
                new_roi = (roi_x, roi_y, roi_w, roi_h)
                rois_being_drawn.append(new_roi)
                print(f"ROI defined: {new_roi}")
                # Draw confirmed ROI onto the base display frame
                cv2.rectangle(frame_display_base, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 0, 255), 2) # Red box
                cv2.imshow(window_name_roi, frame_display_base)
            else:
                print("Warning: ROI ignored (zero width/height).")
                cv2.imshow(window_name_roi, frame_display_base) # Show base frame without the invalid box
            current_roi_start = None # Reset start point
# --- End Mouse Callback Definition ---

# --- ROI SELECTION ---
if use_interactive_roi:
    print("Attempting interactive ROI selection using the first time point...")

    try:
        # --- Extract the first time frame (t=0) from the stack ---
        ### SHAPE-DEPENDENT INDEXING ### (Adjust if shape is different!)
        # Assuming shape (T, C, Y, X)
        # Select time t=0 (axis 0)
        first_time_frame_multi_channel = image_stack[0, :, :, :] # Shape: (C, Y, X)

        # --- Select Vimentin channel (axis 0 of the slice) for display ---
        display_channel = first_time_frame_multi_channel[vimentin_channel_index, :, :] # Shape: (Y, X)
        print(f"Using frame for ROI selection with shape: {display_channel.shape}")

        # Normalize and convert to uint8 for display with OpenCV
        if display_channel.size > 0:
            # Check data type before normalization
            # print(f"ROI display frame dtype: {display_channel.dtype}, min: {np.min(display_channel)}, max: {np.max(display_channel)}")
            frame_norm = cv2.normalize(display_channel, None, 0, 255, cv2.NORM_MINMAX)
            frame_uint8 = frame_norm.astype(np.uint8)
            # Convert grayscale uint8 to BGR for color drawing (boxes)
            frame_for_roi = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
        else:
            print("Error: Extracted frame for ROI selection is empty.")
            sys.exit()

    except IndexError as e:
        print(f"Error extracting first frame for ROI selection: {e}")
        print("Check TIFF shape, channel indices. Ensure TIFF is not empty.")
        sys.exit()
    except ValueError as e:
        print(f"Error during ROI frame preparation: {e}")
        sys.exit()
    except Exception as e: # Catch other potential errors during ROI prep
        print(f"An unexpected error occurred during ROI frame preparation: {e}")
        sys.exit()


    # Initialize global variables needed for drawing
    rois_being_drawn = [] # Reset list for this selection session
    frame_display_base = frame_for_roi.copy()
    frame_copy_dynamic = frame_for_roi.copy()

    cv2.namedWindow(window_name_roi)
    cv2.setMouseCallback(window_name_roi, mouse_callback_roi_user) # Attach callback

    print("\n--- Interactive ROI Selection ---")
    print("INSTRUCTIONS:")
    print(" - Using first time frame (Vimentin channel) for selection.")
    print(" - Click and DRAG mouse on the window to draw an ROI.")
    print(" - RELEASE mouse button to finalize the current ROI (red box).")
    print(" - Draw multiple ROIs if needed.")
    print(" - Press ENTER or 's' to SAVE all drawn ROIs and continue.")
    print(" - Press ESC or 'q' to CANCEL selection (discards all ROIs).")

    # Main loop for interactive ROI selection window
    while True:
        # Decide which frame to show (dynamic if drawing, base otherwise)
        display_frame = frame_copy_dynamic if is_drawing else frame_display_base
        cv2.imshow(window_name_roi, display_frame)
        key = cv2.waitKey(20) & 0xFF

        # Check for SAVE keys (Enter=13)
        if key == 13 or key == ord('s'):
            if not rois_being_drawn:
                print("Warning: No ROIs were selected. Exiting.")
                cv2.destroyAllWindows()
                sys.exit()
            print(f"Saving {len(rois_being_drawn)} ROIs: {rois_being_drawn}")
            rois = rois_being_drawn # Overwrite the main 'rois' variable
            break

        # Check for CANCEL keys (ESC=27)
        elif key == 27 or key == ord('q'):
            print("ROI selection cancelled. Exiting.")
            cv2.destroyAllWindows()
            sys.exit()

    # Clean up ROI selection window
    cv2.destroyAllWindows()
    for _ in range(5): cv2.waitKey(1)

    print("ROI selection complete. Proceeding with processing...")

# --- End of 'if use_interactive_roi' block ---

# Check if ROIs are valid before proceeding
elif not rois: # Handle case where predefined list might be empty and interactive was False
    print("Error: No ROIs defined (predefined list is empty and interactive mode was off).")
    sys.exit()
else: # Use predefined ROIs (if use_interactive_roi was False)
    print(f"Using {len(rois)} predefined ROIs: {rois}")


# --- Initialize Data Storage ---
num_rois = len(rois)
if num_rois == 0:
    print("Error: ROI list is empty after selection/definition step.")
    sys.exit()

vimentin_intensity = [[] for _ in range(num_rois)]
fak_intensity = [[] for _ in range(num_rois)]
fak_area = [[] for _ in range(num_rois)]
print(f"Initialized data storage for {num_rois} ROIs.")

# --- PROCESSING TIFF FRAMES ---
print("Starting frame processing loop...")
frame_counter = 0
# Loop through time dimension (axis 0)
for t in range(num_frames):

    ### SHAPE-DEPENDENT INDEXING ### (Adjust if shape is different!)
    try:
        # Get the 2D multi-channel frame for the current time point 't'
        # Assuming shape (T, C, Y, X)
        # Select time t (axis 0)
        current_frame_multi_channel = image_stack[t, :, :, :] # Shape: (C, Y, X)

    except IndexError:
        print(f"Error accessing frame {t}. Check TIFF stack integrity or loop bounds.")
        break # Stop processing if a frame can't be accessed
    except ValueError as e:
         print(f"Error during frame extraction: {e}")
         break

    frame_counter += 1

    for i, roi_coords in enumerate(rois):
        x, y, w, h = roi_coords # Unpack ROI coordinates
        # Validate ROI bounds against frame dimensions (Y=height, X=width)
        x, y = max(0, x), max(0, y)
        w = min(w, frame_width - x) if x < frame_width else 0
        h = min(h, frame_height - y) if y < frame_height else 0

        if w <= 0 or h <= 0:
            # Append NaN/0 if ROI is invalid or outside frame
            vimentin_intensity[i].append(np.nan)
            fak_intensity[i].append(np.nan)
            fak_area[i].append(0)
            if frame_counter == 1: # Print warning only once per invalid ROI
                 print(f"Warning: ROI {i+1} ({roi_coords}) resulted in zero width/height for frame dimensions {frame_width}x{frame_height}. Appending NaN/0.")
            continue

        try:
            # --- Extract ROI from the 2D multi-channel frame ---
            # current_frame_multi_channel has shape (C, Y, X)
            roi_data = current_frame_multi_channel[:, y:y+h, x:x+w] # Shape: (C, roi_h, roi_w)

            # --- Extract specific channels from the ROI (axis 0 of the slice) ---
            vimentin_frame = roi_data[vimentin_channel_index, :, :] # Shape: (roi_h, roi_w)
            fak_frame = roi_data[fak_channel_index, :, :]         # Shape: (roi_h, roi_w)

            # Check if ROI extraction was successful
            if vimentin_frame.size == 0 or fak_frame.size == 0:
                 vimentin_intensity[i].append(np.nan)
                 fak_intensity[i].append(np.nan)
                 fak_area[i].append(0)
                 continue

            # --- Calculate Mean Intensity ---
            vimentin_intensity[i].append(np.mean(vimentin_frame))
            fak_intensity[i].append(np.mean(fak_frame))

            # --- FAK Area Calculation ---
            # Use the 2D 'fak_frame'. Normalize to uint8 for adaptiveThreshold.
            if fak_frame.size > 0:
                try:
                    # Normalize fak_frame (potentially uint16 or float) to 0-255 range
                    fak_norm = cv2.normalize(fak_frame, None, 0, 255, cv2.NORM_MINMAX)
                    # Convert to uint8
                    fak_uint8 = fak_norm.astype(np.uint8)

                    thresh = cv2.adaptiveThreshold(fak_uint8, 255, adaptive_threshold_method,
                                                   cv2.THRESH_BINARY_INV, adaptive_threshold_block_size,
                                                   adaptive_threshold_constant)
                    fak_area[i].append(np.sum(thresh == 255)) # Count white pixels (inverted threshold)
                except (cv2.error, ValueError) as e:
                    # print(f"Warning: Error during adaptiveThreshold on frame {t}, ROI {i+1}: {e}") # Verbose
                    fak_area[i].append(0) # Append 0 if thresholding fails
            else:
                fak_area[i].append(0) # Append 0 if fak_frame was empty

        except IndexError as e:
             print(f"Error processing ROI {i+1} on frame {t}: Indexing error {e}.")
             print(f"ROI coords: {roi_coords}, Frame shape used: {current_frame_multi_channel.shape}")
             print("Check ROI coordinates and SHAPE-DEPENDENT INDEXING logic.")
             # Append NaN/0 to keep lists aligned
             vimentin_intensity[i].append(np.nan)
             fak_intensity[i].append(np.nan)
             fak_area[i].append(0)
             continue # Move to next ROI
        except Exception as e: # Catch any other unexpected error during ROI processing
             print(f"Unexpected error processing ROI {i+1} on frame {t}: {e}")
             vimentin_intensity[i].append(np.nan)
             fak_intensity[i].append(np.nan)
             fak_area[i].append(0)
             continue

    if frame_counter % 50 == 0: # Print slightly more often
        print(f"Processed frame {frame_counter}/{num_frames}...")


print(f"Finished processing {frame_counter} frames.")

# --- Data Integrity Check ---
# (Keep this section as is, it checks the collected lists)
if not any(v for roi_list in vimentin_intensity for v in roi_list if v is not None and not np.isnan(v)):
    print("Error: No valid Vimentin intensity data was collected. Check ROI validity, channel index, and processing loop.")
    # sys.exit() # Maybe allow continuing if only one channel failed
if not any(f for roi_list in fak_intensity for f in roi_list if f is not None and not np.isnan(f)):
     print("Error: No valid FAK intensity data was collected. Check ROI validity, channel index, and processing loop.")
     # sys.exit()

min_len = 0
try:
    all_lists = [lst for lst in vimentin_intensity + fak_intensity + fak_area if lst is not None]
    if not all_lists or all(not lst for lst in all_lists): # Check if all lists were None or empty
        raise ValueError("No valid data lists found after processing.")
    # Calculate min length only from non-empty lists to avoid errors
    min_len = min(len(lst) for lst in all_lists if lst) # Get min length of non-empty lists
except ValueError as e:
    print(f"Error checking data integrity: {e}")
    sys.exit()

if min_len == 0:
    print("Error: Minimum data list length is zero after processing (all lists might be empty).")
    sys.exit()
print(f"Collected data points per ROI (minimum across ROIs/metrics): {min_len}")


# --- NORMALIZATION ---
# (Keep this section as is, it works on the collected intensity lists)
print("Normalizing data...")
normalized_vimentin_intensity = []
normalized_fak_intensity = []
for i in range(num_rois):
    vim_data_roi = vimentin_intensity[i][:min_len] if i < len(vimentin_intensity) and vimentin_intensity[i] is not None else [np.nan] * min_len
    fak_data_roi = fak_intensity[i][:min_len] if i < len(fak_intensity) and fak_intensity[i] is not None else [np.nan] * min_len

    vim_valid_data = [v for v in vim_data_roi if v is not None and not np.isnan(v)]
    fak_valid_data = [f for f in fak_data_roi if f is not None and not np.isnan(f)]

    # Handle cases where a channel might have had all NaN values
    max_vim = max(vim_valid_data) if vim_valid_data else 1.0
    max_fak = max(fak_valid_data) if fak_valid_data else 1.0
    max_vim = max(max_vim, 1.0) # Ensure max is at least 1.0 to avoid division by zero or tiny numbers
    max_fak = max(max_fak, 1.0)

    normalized_vimentin_intensity.append([(val / max_vim) if val is not None and not np.isnan(val) else np.nan for val in vim_data_roi])
    normalized_fak_intensity.append([(val / max_fak) if val is not None and not np.isnan(val) else np.nan for val in fak_data_roi])


# --- SMOOTHING ---
# (Keep the smooth function as is, but ensure input lists are handled if empty/NaN)
print("Smoothing data...")
def smooth(data, window):
    """Applies convolution smoothing, handling NaNs by interpolation."""
    if not data: # Handle empty list case
        return []
    # Ensure window size is valid for data length
    if len(data) < window:
         # print(f"Warning: Data length ({len(data)}) is less than smooth window ({window}). Returning original data.")
         return data # Or return list of NaNs, or handle as appropriate

    data_arr = np.array(data, dtype=float) # Ensure float type
    nan_mask = np.isnan(data_arr)

    try:
        if np.any(nan_mask) and not np.all(nan_mask):
            indices = np.arange(len(data_arr))
            valid_indices = indices[~nan_mask]
            valid_data = data_arr[~nan_mask]
            if len(valid_indices) > 1: # Need at least two valid points for interpolation
                data_arr[nan_mask] = np.interp(indices[nan_mask], valid_indices, valid_data)
            elif len(valid_indices) == 1: # If only one valid point, fill NaNs with it
                 data_arr[nan_mask] = valid_data[0]
            else: # All values were NaN initially or became NaN
                 data_arr.fill(np.nan) # Keep as NaN if no valid points
        elif np.all(nan_mask):
             pass # Keep all NaNs

    except Exception as e:
        # print(f"Warning: Interpolation failed during smoothing: {e}. Keeping NaNs.")
        pass # Keep original NaNs if interpolation fails

    # Apply convolution only where data is not NaN after potential interpolation
    smoothed_data = np.full_like(data_arr, np.nan) # Start with NaNs
    valid_for_conv = ~np.isnan(data_arr)
    if np.any(valid_for_conv):
         # Pad edges to handle convolution mode 'same' correctly, especially with NaNs nearby
         temp_data = data_arr.copy()
         # Simple edge padding: replicate first/last valid values
         try: # Add try-except for cases where valid_for_conv might be empty despite earlier check
             valid_indices_conv = np.where(valid_for_conv)[0]
             if len(valid_indices_conv) > 0: # Check if there are any valid points
                 first_valid_idx = valid_indices_conv[0]
                 last_valid_idx = valid_indices_conv[-1]
                 temp_data[:first_valid_idx] = data_arr[first_valid_idx]
                 temp_data[last_valid_idx+1:] = data_arr[last_valid_idx]
                 temp_data[np.isnan(temp_data)] = 0 # Temporarily replace remaining NaNs with 0 for convolution

                 convolved = np.convolve(temp_data, np.ones(window)/window, mode='same')
                 smoothed_data[valid_for_conv] = convolved[valid_for_conv] # Only put back results where original data was valid
             # else: leave smoothed_data as NaNs if no valid points at all
         except IndexError: # Handle case where valid_for_conv becomes empty unexpectedly
              pass # Leave smoothed_data as NaNs

    return smoothed_data.tolist()


# --- PLOTTING ---
# (Keep this section as is, it uses the processed lists)
print("Generating plots...")
for i in range(num_rois):
    # Check if data lists exist and have enough length
    if i < len(vimentin_intensity) and i < len(fak_intensity) and \
       i < len(normalized_vimentin_intensity) and i < len(normalized_fak_intensity) and \
       i < len(fak_area):

        # Slice all lists to the consistent minimum length 'min_len'
        vim_int_plot = vimentin_intensity[i][:min_len]
        fak_int_plot = fak_intensity[i][:min_len]
        norm_vim_plot = normalized_vimentin_intensity[i][:min_len]
        norm_fak_plot = normalized_fak_intensity[i][:min_len]
        fak_area_plot = fak_area[i][:min_len]

        # Check if lists (after slicing) are empty before smoothing/plotting
        if not any(v is not None and not np.isnan(v) for v in vim_int_plot) and \
           not any(f is not None and not np.isnan(f) for f in fak_int_plot) and \
           not any(fa is not None and not np.isnan(fa) for fa in fak_area_plot):
            print(f"Warning: Skipping plot for ROI {i+1} due to empty or all-NaN data lists after slicing.")
            continue

        plt.figure(figsize=(12, 8))

        # --- Plot 1: Smoothed Raw Intensity ---
        plt.subplot(3, 1, 1)
        smoothed_vim = smooth(vim_int_plot, smooth_window)
        smoothed_fak = smooth(fak_int_plot, smooth_window)
        if any(v is not None and not np.isnan(v) for v in smoothed_vim): # Only plot if smoothing produced valid data
             plt.plot(smoothed_vim, label='Vimentin', color='red')
        if any(f is not None and not np.isnan(f) for f in smoothed_fak):
             plt.plot(smoothed_fak, label='FAK', color='green')
        plt.title(f'ROI {i+1}: Intensity & Area Over Time (Smoothed, Window={smooth_window})')
        plt.ylabel('Smoothed Intensity')
        if any(v is not None and not np.isnan(v) for v in smoothed_vim) or \
           any(f is not None and not np.isnan(f) for f in smoothed_fak):
             plt.legend() # Only show legend if something was plotted
        plt.grid(True)

        # --- Plot 2: Smoothed Normalized Intensity ---
        plt.subplot(3, 1, 2)
        smoothed_norm_vim = smooth(norm_vim_plot, smooth_window)
        smoothed_norm_fak = smooth(norm_fak_plot, smooth_window)
        if any(v is not None and not np.isnan(v) for v in smoothed_norm_vim):
             plt.plot(smoothed_norm_vim, label='Norm. Vimentin', color='red')
        if any(f is not None and not np.isnan(f) for f in smoothed_norm_fak):
            plt.plot(smoothed_norm_fak, label='Norm. FAK', color='green')
        plt.ylabel('Normalized Intensity (0-1)')
        if any(v is not None and not np.isnan(v) for v in smoothed_norm_vim) or \
           any(f is not None and not np.isnan(f) for f in smoothed_norm_fak):
             plt.legend()
        plt.grid(True)

        # --- Plot 3: Smoothed FAK Area ---
        plt.subplot(3, 1, 3)
        smoothed_fak_area = smooth(fak_area_plot, smooth_window)
        if any(fa is not None and not np.isnan(fa) for fa in smoothed_fak_area):
             plt.plot(smoothed_fak_area, label='FAK Area', color='blue')
             plt.legend()
        # Use the determined frame_rate for the label
        plt.xlabel(f'Frame (Frame Rate approx. {frame_rate:.3f} FPS)') # Use calculated/fallback frame_rate
        plt.ylabel('Smoothed Area (Pixels)')
        plt.grid(True)

        plt.tight_layout()
        plt.show() # Display the plot
    else:
        print(f"Warning: Skipping plot for ROI {i+1} due to index out of range (data lists might be inconsistent before slicing).")


# --- EXPORT TO CSV ---
# (Keep this section as is, it uses the processed lists)
print("Exporting data to CSV...")
frames_to_export = min_len # Use the minimum consistent length calculated earlier
time_points_sec = [f * frame_interval_sec for f in range(frames_to_export)] if frame_interval_sec is not None else list(range(1, frames_to_export + 1))
data = {
     'Frame': list(range(1, frames_to_export + 1)), # Frame numbers correspond to Time axis (axis 0)
     'Time_sec': time_points_sec # Add time in seconds if interval was found/set
     }


export_successful = True
try:
    for i in range(num_rois):
        # Ensure lists exist and slice to the minimum consistent length 'frames_to_export'
        data[f'ROI{i+1}_Vimentin_Intensity'] = vimentin_intensity[i][:frames_to_export] if i < len(vimentin_intensity) and vimentin_intensity[i] is not None else [np.nan] * frames_to_export
        data[f'ROI{i+1}_FAK_Intensity'] = fak_intensity[i][:frames_to_export] if i < len(fak_intensity) and fak_intensity[i] is not None else [np.nan] * frames_to_export
        data[f'ROI{i+1}_FAK_Area_Pixels'] = fak_area[i][:frames_to_export] if i < len(fak_area) and fak_area[i] is not None else [0] * frames_to_export # Use 0 for area if missing
        data[f'ROI{i+1}_Norm_Vimentin'] = normalized_vimentin_intensity[i][:frames_to_export] if i < len(normalized_vimentin_intensity) and normalized_vimentin_intensity[i] is not None else [np.nan] * frames_to_export
        data[f'ROI{i+1}_Norm_FAK'] = normalized_fak_intensity[i][:frames_to_export] if i < len(normalized_fak_intensity) and normalized_fak_intensity[i] is not None else [np.nan] * frames_to_export

    df = pd.DataFrame(data)
    # Reorder columns to put Time_sec after Frame
    cols = df.columns.tolist()
    if 'Time_sec' in cols:
         cols.insert(1, cols.pop(cols.index('Time_sec')))
         df = df[cols]

    # >>> Ensure this output path is correct <<<
    # Use raw string or forward slashes for path
    csv_output_dir = r'C:/Users/Arun/Downloads/PhD/Manuscript/TIRF LIVE IMAGING/Python Analysis Results/'
    csv_output_filename = 'FAK_VIM_ROI_Tracking_FROM_TIFF_TCYX_autoFPS.csv' # Changed suffix
    csv_output_path = os.path.join(csv_output_dir, csv_output_filename)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False, float_format='%.4f') # Format float precision
    print(f"✅ Data successfully exported to {csv_output_path}")

except Exception as e:
    print(f"\nError constructing DataFrame or exporting data to CSV: {e}")
    print("Please check data consistency and file path permissions.")
    export_successful = False

# Fallback save (keep as is, maybe update filename)
if not export_successful:
    try:
        raw_data_dict = {
            'vimentin_intensity': vimentin_intensity,
            'fak_intensity': fak_intensity,
            'fak_area': fak_area,
            'normalized_vimentin_intensity': normalized_vimentin_intensity,
            'normalized_fak_intensity': normalized_fak_intensity
        }
        fallback_dir = r'C:/Users/Arun/Downloads/PhD/Manuscript/TIRF LIVE IMAGING/Python Analysis Results/'
        os.makedirs(fallback_dir, exist_ok=True) # Ensure directory exists
        fallback_path = os.path.join(fallback_dir, 'FALLBACK_RAW_DATA_FROM_TIFF_TCYX_autoFPS.json') # Changed suffix
        import json

        # Custom encoder to handle numpy types and NaN for JSON
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    # Handle NaN specifically for JSON compatibility
                    return None if np.isnan(obj) else float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist() # Convert arrays to lists
                return super(NumpyEncoder, self).default(obj)

        with open(fallback_path, 'w') as f:
            json.dump(raw_data_dict, f, cls=NumpyEncoder, indent=4)
        print(f"⚠️ DataFrame export failed. Saved raw data lists to {fallback_path}")
    except Exception as json_e:
        print(f"Failed to save fallback raw data: {json_e}")


print("\nScript finished.")
