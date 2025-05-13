import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import tifffile
from tifffile import TiffFile
import re
import functools # For trackbar callbacks
import json      # For saving parameters
import xml.etree.ElementTree as ET # For parsing OME-XML string

# --- >>> ADDED: For File Dialogs <<< ---
import tkinter as tk
from tkinter import filedialog
# --- >>> END ADDED <<< ---

# --- Image Processing & Tracking Libraries ---
from skimage import morphology
from skimage import filters
from skimage import measure
from skimage import segmentation
from skimage.morphology import disk
import trackpy as tp # For tracking
from adjustText import adjust_text

# --- CONFIGURATION ---
# >>> File Paths (These will be requested at runtime via dialog) <<<
# output_prefix defines the base name for output files
output_prefix = "FA_Track_Analysis" # <<< Prefix for output files.

# >>> Pixel Size (IMPORTANT for Micrometer Conversion) <<<
# Script will attempt to read from metadata (ImageJ, OME, Standard Tags).
# Set this manually ONLY if automatic detection fails or to override metadata.
# Value should be in micrometers per pixel (e.g., 0.108).
# If set to None AND not found in metadata, units will remain in pixels.
cfg_pixel_size_um = 0.08 # <<< SET MANUALLY (Example value) >>>

# >>> Channel Indices (Based on T, C, Y, X order) <<<
# Ensure these match your data based on the printed shape later
# <<< MODIFIED based on user metadata >>>
vimentin_channel_index = 1 # Vimentin is W1 (index 1)
fak_channel_index = 0      # FAK is W0 (index 0)
# <<< END MODIFICATION >>>

# >>> Segmentation Parameters (Defaults - MAY BE OVERRIDDEN BY INTERACTIVE TUNING) <<<
# Set segmentation method to 'local' to use scikit-image local adaptive thresholding
cfg_segmentation_method = 'local' # Options: 'local', 'otsu', 'manual' (Use cfg_ prefix for clarity)

# -- Parameters for 'local' method (skimage.filters.threshold_local) --
cfg_local_block_size = 35    # Must be odd. Size of pixel neighborhood. Needs tuning! Should be larger than features.
cfg_local_offset = 0         # Constant subtracted from local threshold. Tune this value.
cfg_local_method = 'gaussian'# Method to calculate local threshold ('gaussian', 'mean', 'median')

# -- Parameters for 'otsu' or 'manual' methods (if selected above) --
cfg_manual_threshold_value = 50 # Set manually if using 'manual'

# -- Morphological Cleaning --
cfg_min_fa_area = 10         # Minimum pixel area. Initial value if tuning.
cfg_max_fa_area = 1500       # Max area filter
cfg_remove_border_objects = True # Remove objects touching the image border?
cfg_add_opening = True       # Flag for opening step
cfg_opening_disk_radius = 1  # Radius for opening disk
cfg_add_closing = True       # Flag for closing step
cfg_closing_disk_radius = 2  # Radius for closing disk
# -- Optional Preprocessing --
cfg_use_gaussian_blur = True
cfg_gaussian_sigma = 1.0     # Initial value if tuning.

# >>> Tracking Parameters (CRITICAL - REQUIRES TUNING) <<<
# trackpy linking parameters
cfg_search_range = 10        # Max distance (pixels) an FA can move between frames
cfg_memory = 6               # Allows bridging up to 5 missed frames

# >>> Track Filtering Parameters <<<
cfg_min_track_length = 5     # Minimum number of frames a track must exist for to be kept

# --- END CONFIGURATION ---


# --- Helper Functions ---
def preprocess_image(img, use_gaussian=True, sigma=1.0):
    """Optional preprocessing: Gaussian blur."""
    if use_gaussian and sigma > 0:
        return filters.gaussian(img, sigma=sigma, preserve_range=True)
    return img

def segment_image(img, method='local', block_size=35, offset=0, local_method='gaussian', manual_thresh=50):
    """Segment image using specified method."""
    if method == 'local':
        block_size = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        try:
            local_thresh = filters.threshold_local(img, block_size, method=local_method, offset=offset)
            binary = img > local_thresh
        except ValueError as e:
            print(f"\nError in threshold_local (block_size={block_size}): {e}. Falling back.")
            try: thresh = filters.threshold_otsu(img); binary = img > thresh
            except ValueError: binary = img > np.median(img)
    elif method == 'otsu':
        try: thresh = filters.threshold_otsu(img); binary = img > thresh
        except ValueError: binary = img > np.median(img)
    elif method == 'manual':
        binary = img > manual_thresh
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    return binary.astype(bool)

def clean_mask(mask, min_area=10, max_area=None, remove_border=True,
               add_opening=True, opening_radius=1, add_closing=True, closing_radius=2):
    """Clean binary mask using morphological operations."""
    mask = mask.astype(bool)
    cleaned_mask = mask
    if min_area > 0:
        if np.any(cleaned_mask):
            cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=min_area)
        else: return cleaned_mask # Return empty mask if already empty
    if add_opening and opening_radius > 0:
        if np.any(cleaned_mask):
            cleaned_mask = morphology.binary_opening(cleaned_mask, footprint=disk(opening_radius))
        else: return cleaned_mask
    if add_closing and closing_radius > 0:
        if np.any(cleaned_mask):
            cleaned_mask = morphology.binary_closing(cleaned_mask, footprint=disk(closing_radius))
        else: return cleaned_mask
    if max_area is not None and max_area > 0:
         labels, num_labels = measure.label(cleaned_mask, return_num=True, connectivity=2)
         if num_labels > 0:
             props = measure.regionprops(labels)
             large_labels = {prop.label for prop in props if prop.area > max_area}
             if large_labels:
                  large_mask = np.isin(labels, list(large_labels))
                  cleaned_mask[large_mask] = False
    if remove_border:
        if np.any(cleaned_mask):
            cleaned_mask = segmentation.clear_border(cleaned_mask)
    return cleaned_mask

# --- Interactive Tuning Function ---
# [interactive_parameter_tuning function remains the same - omitted for brevity]
def interactive_parameter_tuning(full_image_stack, fak_ch_index, num_total_frames, initial_params):
    """
    Displays an interactive window to tune segmentation parameters
    on user-selected key frames (Start, Mid, End). Focused on local thresholding.
    """
    print("\n--- Interactive Segmentation Tuning ('local' method) ---")
    print("Use 'Key Frame' slider to switch preview (0=Start, 1=Mid, 2=Break, 3=End).")
    print("Adjust other sliders to optimize segmentation across key frames.")
    print("Segmentation boundaries will be shown in GREEN.")
    print("Press 'ENTER' to accept current parameters and continue.")
    print("Press 'ESC' to exit script.")

    window_name = "Segmentation Tuning (Key Frames | ENTER=Accept, ESC=Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600) # Adjust initial size if needed

    # Define key frames
    mid_frame_idx = num_total_frames // 2
    last_frame_idx = num_total_frames - 1
    break_frame_idx = min(last_frame_idx, 217) # Frame near previous break point
    key_frame_indices = [0, mid_frame_idx, break_frame_idx, last_frame_idx]
    key_frame_labels = ["Start (0)", f"Mid ({mid_frame_idx})", f"Break ({break_frame_idx})", f"End ({last_frame_idx})"]
    num_key_frames = len(key_frame_indices)

    # --- Trackbar Setup ---
    def on_trackbar(val): pass # Dummy callback

    current_params = initial_params.copy()
    current_params['segmentation_method'] = 'local' # Force local for this tuner

    # Key Frame selection trackbar (0 to num_key_frames - 1)
    cv2.createTrackbar('Key Frame', window_name, 0, num_key_frames - 1, on_trackbar)
    # Gaussian Sigma (Scale 0-50 maps to 0.0-5.0)
    cv2.createTrackbar('Sigma*10', window_name, int(current_params['gaussian_sigma'] * 10), 50, on_trackbar)
    # Min Area (0-500 pixels, adjust range if needed)
    cv2.createTrackbar('Min Area', window_name, current_params['min_fa_area'], 500, on_trackbar)
    # Local Block Size (Trackbar 1-100 maps to odd sizes 3-201) - Adjust max range if needed
    cv2.createTrackbar('Local Block/2', window_name, (current_params['local_block_size'] - 1) // 2, 100, on_trackbar)
    # Local Offset (Trackbar 0-50 maps to -25 to +25) - Adjust range/center if needed
    offset_center = 25
    cv2.createTrackbar('Local Offset+'+str(offset_center), window_name, current_params['local_offset'] + offset_center, 50, on_trackbar)

    last_key_frame_slider_pos = -1 # To track when frame selection changes
    preview_img = None # Initialize preview image

    while True:
        # --- Get current trackbar values ---
        key_frame_slider_pos = cv2.getTrackbarPos('Key Frame', window_name)
        sigma_val = cv2.getTrackbarPos('Sigma*10', window_name) / 10.0
        min_area_val = cv2.getTrackbarPos('Min Area', window_name)
        block_size_val = max(3, 2 * cv2.getTrackbarPos('Local Block/2', window_name) + 1) # Ensure odd >= 3
        offset_val = cv2.getTrackbarPos('Local Offset+'+str(offset_center), window_name) - offset_center

        # Update parameters based on trackbar positions
        params_changed = (sigma_val != current_params['gaussian_sigma'] or
                          min_area_val != current_params['min_fa_area'] or
                          block_size_val != current_params['local_block_size'] or
                          offset_val != current_params['local_offset'])

        current_params['gaussian_sigma'] = sigma_val
        current_params['min_fa_area'] = min_area_val
        current_params['local_block_size'] = block_size_val
        current_params['local_offset'] = offset_val

        # --- Get the selected KEY frame if it changed or parameters changed ---
        if key_frame_slider_pos != last_key_frame_slider_pos or params_changed:
            # Get the actual frame index from the slider position
            frame_idx = key_frame_indices[key_frame_slider_pos]
            frame_label_str = key_frame_labels[key_frame_slider_pos]
            print(f" Previewing Frame: {frame_idx} ({frame_label_str.split(' ')[0]})", end='\r') # Show which frame is previewed

            try:
                img_to_tune = full_image_stack[frame_idx, fak_ch_index, :, :]

                # Apply processing with current parameters
                preprocessed = preprocess_image(img_to_tune,
                                                use_gaussian=current_params['use_gaussian_blur'],
                                                sigma=current_params['gaussian_sigma'])
                binary_mask = segment_image(preprocessed, method='local',
                                            block_size=current_params['local_block_size'],
                                            offset=current_params['local_offset'],
                                            local_method=current_params['local_method'])
                # Use configured opening/closing radii here for preview consistency
                cleaned_mask = clean_mask(binary_mask,
                                          min_area=current_params['min_fa_area'],
                                          max_area=current_params.get('max_fa_area'),
                                          remove_border=current_params['remove_border_objects'],
                                          add_opening=current_params['add_opening'],
                                          opening_radius=current_params['opening_disk_radius'],
                                          add_closing=current_params['add_closing'],
                                          closing_radius=current_params['closing_disk_radius'])


                # --- Create preview image ---
                display_img_norm = cv2.normalize(img_to_tune, None, 0, 255, cv2.NORM_MINMAX)
                display_img_8u = display_img_norm.astype(np.uint8)
                preview_img = cv2.cvtColor(display_img_8u, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(cleaned_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 1)

                # Add text showing current parameter values and frame number/label
                font = cv2.FONT_HERSHEY_SIMPLEX; text_y = 20
                cv2.putText(preview_img, f"Preview: {frame_label_str}", (10, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA); text_y += 25
                cv2.putText(preview_img, f"Sigma: {current_params['gaussian_sigma']:.1f}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA); text_y += 20
                cv2.putText(preview_img, f"Min Area (pixels): {current_params['min_fa_area']}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA); text_y += 20
                cv2.putText(preview_img, f"Local Block: {current_params['local_block_size']}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA); text_y += 20
                cv2.putText(preview_img, f"Local Offset: {current_params['local_offset']}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                last_key_frame_slider_pos = key_frame_slider_pos

            except Exception as e:
                print(f"\nError during preview processing frame {frame_idx}: {e}")
                img_shape = full_image_stack.shape
                err_img = np.zeros((img_shape[2], img_shape[3], 3), dtype=np.uint8)
                cv2.putText(err_img, f"Error processing frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(err_img, f"Check parameters/console", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                preview_img = err_img

        if preview_img is not None: cv2.imshow(window_name, preview_img)
        else:
             img_shape = full_image_stack.shape
             blank_img = np.zeros((img_shape[2], img_shape[3], 3), dtype=np.uint8)
             cv2.putText(blank_img, "Error on first frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
             cv2.imshow(window_name, blank_img)

        key = cv2.waitKey(50) & 0xFF
        if key == 27: print("\nParameter tuning cancelled by user. Exiting."); cv2.destroyAllWindows(); sys.exit()
        elif key == 13: print("\nParameters accepted by user."); break

    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)
    final_params = initial_params.copy()
    final_params.update(current_params)
    return final_params

# --- Segmentation Preview Function ---
# [preview_segmentation function remains the same - omitted for brevity]
def preview_segmentation(full_image_stack, fak_ch_index, num_total_frames, approved_params):
    """
    Displays segmentation results on all frames using fixed parameters
    and allows user to approve or cancel before full analysis.
    """
    print("\n--- Segmentation Preview Across All Frames ---")
    print("Use the 'Frame' slider to scroll through the time series.")
    print("Segmentation (using approved parameters) is shown in GREEN.")
    print("Press 'ENTER' to approve and continue to feature extraction/tracking.")
    print("Press 'ESC' to abort the script.")

    window_name = "Segmentation Preview (ENTER=Continue, ESC=Abort)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    def on_trackbar(val): pass
    cv2.createTrackbar('Frame', window_name, 0, num_total_frames - 1, on_trackbar)
    last_frame_idx = -1
    preview_img = None

    while True:
        frame_idx = cv2.getTrackbarPos('Frame', window_name)
        if frame_idx != last_frame_idx:
            print(f" Previewing Frame: {frame_idx} ", end='\r')
            try:
                img_to_preview = full_image_stack[frame_idx, fak_ch_index, :, :]
                preprocessed = preprocess_image(img_to_preview,
                                                use_gaussian=approved_params['use_gaussian_blur'],
                                                sigma=approved_params['gaussian_sigma'])
                binary_mask = segment_image(preprocessed,
                                            method=approved_params['segmentation_method'],
                                            block_size=approved_params['local_block_size'],
                                            offset=approved_params['local_offset'],
                                            local_method=approved_params['local_method'],
                                            manual_thresh=approved_params['manual_threshold_value'])
                cleaned_mask = clean_mask(binary_mask,
                                          min_area=approved_params['min_fa_area'],
                                          max_area=approved_params.get('max_fa_area'),
                                          remove_border=approved_params['remove_border_objects'],
                                          add_opening=approved_params['add_opening'],
                                          opening_radius=approved_params['opening_disk_radius'],
                                          add_closing=approved_params['add_closing'],
                                          closing_radius=approved_params['closing_disk_radius'])

                display_img_norm = cv2.normalize(img_to_preview, None, 0, 255, cv2.NORM_MINMAX)
                display_img_8u = display_img_norm.astype(np.uint8)
                preview_img = cv2.cvtColor(display_img_8u, cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(cleaned_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(preview_img, f"Frame: {frame_idx}", (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                last_frame_idx = frame_idx
            except Exception as e:
                print(f"\nError during preview processing frame {frame_idx}: {e}")
                img_shape = full_image_stack.shape
                err_img = np.zeros((img_shape[2], img_shape[3], 3), dtype=np.uint8)
                cv2.putText(err_img, f"Error processing frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                preview_img = err_img

        if preview_img is not None: cv2.imshow(window_name, preview_img)
        else:
             img_shape = full_image_stack.shape
             blank_img = np.zeros((img_shape[2], img_shape[3], 3), dtype=np.uint8)
             cv2.putText(blank_img, "Error loading preview", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
             cv2.imshow(window_name, blank_img)

        key = cv2.waitKey(50) & 0xFF
        if key == 27: print("\nSegmentation preview aborted by user. Exiting."); cv2.destroyAllWindows(); return False
        elif key == 13: print("\nSegmentation preview approved by user. Continuing analysis."); break

    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)
    return True

# --- MAIN SCRIPT ---
def main():
    # <<< Use Tkinter for file/dir selection >>>
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window

    print("--- File Path Setup ---")
    print("Opening file dialog to select input TIFF file...")
    current_tiff_path = filedialog.askopenfilename(
        title="Select Input TIFF File",
        filetypes=[("TIFF Files", "*.tif *.tiff"), ("All Files", "*.*")]
    )
    if not current_tiff_path: # Handle cancellation
        print("No input file selected. Exiting.")
        sys.exit()
    print(f"Input file selected: {current_tiff_path}")

    print("Opening directory dialog to select output folder...")
    current_output_dir = filedialog.askdirectory(
        title="Select Output Directory"
    )
    if not current_output_dir: # Handle cancellation
        print("No output directory selected. Exiting.")
        sys.exit()
    print(f"Output directory selected: {current_output_dir}")
    print("-----------------------")
    # <<< END File/Dir Selection >>>

    # <<< Define output_prefix here, possibly based on input name >>>
    base_filename = os.path.basename(current_tiff_path)
    # Use the global output_prefix defined in config
    current_output_prefix = output_prefix # Use the one from config for consistency
    print(f"Using output file prefix: {current_output_prefix}")
    # <<< END Define output_prefix >>>


    print(f"Starting FA analysis for: {current_tiff_path}")
    print(f"Output directory: {current_output_dir}")

    # --- Load Data ---
    print("Loading TIFF stack...")
    pixel_size_um = cfg_pixel_size_um # Start with config value
    frame_interval_sec = None
    try:
        with tifffile.TiffFile(current_tiff_path) as tif: # Use selected path
            image_stack = tif.asarray()
            # --- Attempt to read metadata ---
            # [Metadata reading code remains the same - omitted for brevity]
            # ImageJ Metadata
            try:
                ij_meta = tif.imagej_metadata
                if ij_meta:
                    if 'unit' in ij_meta and ij_meta['unit'] in ('micron', 'um'):
                        if 'pixel_width' in ij_meta and 'pixel_height' in ij_meta:
                             if ij_meta['pixel_width'] == ij_meta['pixel_height']:
                                 if pixel_size_um is None: pixel_size_um = float(ij_meta['pixel_width']); print(f"  Read pixel size from ImageJ metadata: {pixel_size_um:.4f} um/pixel")
                    if 'finterval' in ij_meta: frame_interval_sec = float(ij_meta['finterval']); print(f"  Found frame interval in ImageJ metadata (finterval): {frame_interval_sec:.4f} sec")
            except Exception as e: print(f"Could not process ImageJ metadata: {e}")
            # OME Metadata
            try:
                ome_xml = tif.ome_metadata
                if ome_xml and isinstance(ome_xml, str):
                    root = ET.fromstring(ome_xml); ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}; pixels_node = root.find('.//ome:Pixels', ns)
                    if pixels_node is not None:
                        phys_x = pixels_node.get('PhysicalSizeX'); phys_y = pixels_node.get('PhysicalSizeY'); unit_x = pixels_node.get('PhysicalSizeXUnit')
                        if phys_x and phys_y and phys_x == phys_y and unit_x == 'µm':
                             if pixel_size_um is None: pixel_size_um = float(phys_x); print(f"  Read pixel size from OME metadata: {pixel_size_um:.4f} um/pixel")
                        if frame_interval_sec is None:
                            plane_node = pixels_node.find('.//ome:Plane', ns)
                            if plane_node is not None and 'DeltaT' in plane_node.attrib:
                                delta_t = float(plane_node.attrib['DeltaT']); delta_t_unit = plane_node.attrib.get('DeltaTUnit', 's')
                                if delta_t_unit == 'ms': frame_interval_sec = delta_t / 1000.0
                                elif delta_t_unit == 's': frame_interval_sec = delta_t
                                else: frame_interval_sec = delta_t # Assume seconds
                                if frame_interval_sec is not None: print(f"  Found frame interval in OME metadata (DeltaT): {frame_interval_sec:.4f} sec")
            except Exception as e: print(f"Could not process OME metadata: {e}")
            # Standard TIFF resolution tags
            if pixel_size_um is None:
                 try:
                     page = tif.pages[0]; x_res_tag = page.tags.get(282); y_res_tag = page.tags.get(283); unit_tag = page.tags.get(296)
                     if x_res_tag and y_res_tag and unit_tag:
                         x_res_num, x_res_den = x_res_tag.value; y_res_num, y_res_den = y_res_tag.value; unit = unit_tag.value
                         if x_res_den != 0 and y_res_den != 0 and (x_res_num / x_res_den) == (y_res_num / y_res_den):
                             pixels_per_unit = x_res_num / x_res_den
                             if pixels_per_unit > 0:
                                 res_unit_um = None; unit_name = "Unknown"
                                 if unit == 2: res_unit_um = 25400.0; unit_name = "Inch"
                                 elif unit == 3: res_unit_um = 10000.0; unit_name = "Centimeter"
                                 if res_unit_um is not None: pixel_size_um = res_unit_um / pixels_per_unit; print(f"  Read pixel size from standard TIFF tags (Unit={unit_name}): {pixel_size_um:.4f} um/pixel")
                 except Exception as e: print(f"Could not process standard TIFF resolution tags: {e}")
            # Fallback: ImageDescription tag for frame interval
            if frame_interval_sec is None:
                 try:
                     if tif.pages and len(tif.pages) > 0:
                         desc_tag = tif.pages[0].tags.get(270); description = desc_tag.value if desc_tag else None
                         if description: match = re.search(r"(?:interval|deltaT)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(ms|s)?", str(description), re.IGNORECASE);
                         if match: value = float(match.group(1)); unit = match.group(2); frame_interval_sec = value / 1000.0 if unit and unit.lower() == 'ms' else value; print(f"  Found potential interval in ImageDescription: {frame_interval_sec:.4f} sec")
                 except Exception: pass

    except Exception as e: print(f"Error reading TIFF file: {e}"); sys.exit()

    # --- Final check on pixel size ---
    if pixel_size_um is None:
        print("\nWarning: Pixel size (um/pixel) could not be read from metadata or config.")
        print("         Calculations and plots will use PIXEL units for area.")
        area_per_pixel_um2 = None; area_units_label = "pixels"; area_col_name_csv = "area_pixels"
    else:
        print(f"Using pixel size: {pixel_size_um:.4f} um/pixel")
        area_per_pixel_um2 = pixel_size_um * pixel_size_um
        area_units_label = "um^2"; area_col_name_csv = "area_um2"


    # --- Verify Dimensions (Assuming T, C, Y, X) ---
    print(f"Loaded stack shape: {image_stack.shape}")
    if image_stack.ndim != 4: print(f"Error: Expected 4D stack (T, C, Y, X), but got {image_stack.ndim}D."); sys.exit()
    num_frames = image_stack.shape[0]; num_channels = image_stack.shape[1]; frame_height = image_stack.shape[2]; frame_width = image_stack.shape[3]
    print(f"Interpreted as: T={num_frames}, C={num_channels}, H={frame_height}, W={frame_width}")
    if not (0 <= vimentin_channel_index < num_channels and 0 <= fak_channel_index < num_channels): print(f"Error: Channel index out of bounds"); sys.exit()

    # --- Set Frame Rate ---
    if frame_interval_sec is not None and frame_interval_sec > 0:
        frame_rate = 1.0 / frame_interval_sec; print(f"Using frame interval: {frame_interval_sec:.4f} sec (Rate: {frame_rate:.3f} FPS)")
    else:
        print("Warning: Could not automatically detect frame interval. Using fallback 1.0 sec.")
        frame_interval_sec = 1.0; frame_rate = 1.0 / frame_interval_sec; print(f"Using default interval of {frame_interval_sec:.1f} sec (Rate: {frame_rate:.3f} FPS)")


    # --- >>> Step 1: Interactive Parameter Tuning <<< ---
    param_segmentation_method = cfg_segmentation_method; param_local_block_size = cfg_local_block_size; param_local_offset = cfg_local_offset; param_local_method = cfg_local_method; param_manual_threshold_value = cfg_manual_threshold_value; param_min_fa_area = cfg_min_fa_area; param_max_fa_area = cfg_max_fa_area; param_remove_border_objects = cfg_remove_border_objects; param_use_gaussian_blur = cfg_use_gaussian_blur; param_gaussian_sigma = cfg_gaussian_sigma; param_add_opening = cfg_add_opening; param_opening_radius = cfg_opening_disk_radius; param_add_closing = cfg_add_closing; param_closing_radius = cfg_closing_disk_radius
    initial_params = {'segmentation_method': param_segmentation_method, 'local_block_size': param_local_block_size, 'local_offset': param_local_offset, 'local_method': param_local_method, 'manual_threshold_value': param_manual_threshold_value, 'min_fa_area': param_min_fa_area, 'max_fa_area': param_max_fa_area, 'remove_border_objects': param_remove_border_objects, 'use_gaussian_blur': param_use_gaussian_blur, 'gaussian_sigma': param_gaussian_sigma, 'add_opening': param_add_opening, 'opening_disk_radius': param_opening_radius, 'add_closing': param_add_closing, 'closing_disk_radius': param_closing_radius}
    approved_params = interactive_parameter_tuning(image_stack, fak_channel_index, num_frames, initial_params)
    param_segmentation_method = approved_params['segmentation_method']; param_gaussian_sigma = approved_params['gaussian_sigma']; param_min_fa_area = approved_params['min_fa_area']; param_local_block_size = approved_params['local_block_size']; param_local_offset = approved_params['local_offset']; param_local_method = approved_params['local_method']; param_max_fa_area = approved_params['max_fa_area']; param_remove_border_objects = approved_params['remove_border_objects']; param_use_gaussian_blur = approved_params['use_gaussian_blur']; param_manual_threshold_value = approved_params['manual_threshold_value']; param_add_opening = approved_params['add_opening']; param_opening_radius = approved_params['opening_disk_radius']; param_add_closing = approved_params['add_closing']; param_closing_radius = approved_params['closing_disk_radius']

    print("\n--- Parameters after tuning: ---")
    # [Parameter printing remains the same - omitted for brevity]
    print(f" Segmentation Method: {param_segmentation_method} ({param_local_method if param_segmentation_method == 'local' else 'N/A'})"); print(f" Gaussian Sigma: {param_gaussian_sigma:.1f}"); print(f" Min FA Area: {param_min_fa_area}"); print(f" Max FA Area: {param_max_fa_area if param_max_fa_area is not None else 'None'}");
    if param_segmentation_method == 'local': print(f" Local Block Size: {param_local_block_size}"); print(f" Local Offset: {param_local_offset}")
    elif param_segmentation_method == 'manual': print(f" Manual Threshold: {param_manual_threshold_value}")
    print(f" Use Opening: {param_add_opening} (Radius: {param_opening_radius})"); print(f" Use Closing: {param_add_closing} (Radius: {param_closing_radius})"); print("--------------------------------")

    # --- >>> Step 2: Preview Segmentation Across All Frames <<< ---
    preview_ok = preview_segmentation(image_stack, fak_channel_index, num_frames, approved_params)
    if not preview_ok: print("Analysis aborted after segmentation preview."); sys.exit()

    # --- Step 3: Process Frames: Segmentation & Feature Extraction ---
    all_features = []
    print("\nProcessing frames for FA segmentation and feature extraction...")
    for t in range(num_frames):
        print(f" Processing frame {t+1}/{num_frames}...", end='\r')
        fak_frame = image_stack[t, fak_channel_index, :, :]
        vimentin_frame = image_stack[t, vimentin_channel_index, :, :]

        fak_processed = preprocess_image(fak_frame, use_gaussian=param_use_gaussian_blur, sigma=param_gaussian_sigma)
        fak_binary_mask = segment_image(fak_processed, method=param_segmentation_method,
                                        block_size=param_local_block_size, offset=param_local_offset,
                                        local_method=param_local_method, manual_thresh=param_manual_threshold_value)
        fak_cleaned_mask = clean_mask(fak_binary_mask, min_area=param_min_fa_area,
                                      max_area=param_max_fa_area, remove_border=param_remove_border_objects,
                                      add_opening=param_add_opening, opening_radius=param_opening_radius,
                                      add_closing=param_add_closing, closing_radius=param_closing_radius)

        label_image, num_labels = measure.label(fak_cleaned_mask, background=0, connectivity=2, return_num=True)

        if num_labels > 0:
            props_fak = measure.regionprops(label_image, intensity_image=fak_frame)
            props_vim = measure.regionprops(label_image, intensity_image=vimentin_frame)
            frame_features_list = []
            for i, prop_f in enumerate(props_fak):
                prop_v = props_vim[i]
                label = prop_f.label; centroid = prop_f.centroid; area_pix = prop_f.area; bbox = prop_f.bbox
                fak_mean = prop_f.mean_intensity; fak_max = prop_f.max_intensity
                vim_mean = prop_v.mean_intensity
                fak_integ = fak_mean * area_pix
                vim_integ = vim_mean * area_pix if pd.notna(vim_mean) else np.nan
                pcc, m1_coeff, m2_coeff, icq = np.nan, np.nan, np.nan, np.nan
                try: # Combined try block for colocalization
                    coords = prop_f.coords
                    if coords.shape[0] > 1:
                        fak_pixels = fak_frame[coords[:, 0], coords[:, 1]]; vim_pixels = vimentin_frame[coords[:, 0], coords[:, 1]]
                        std_fak = np.std(fak_pixels); std_vim = np.std(vim_pixels)
                        if std_fak > 1e-6 and std_vim > 1e-6: pcc = np.corrcoef(fak_pixels, vim_pixels)[0, 1]
                        vim_thresh, fak_thresh = np.nan, np.nan
                        try:
                            if len(np.unique(vim_pixels)) > 1: vim_thresh = filters.threshold_otsu(vim_pixels)
                        except ValueError: pass
                        try:
                            if len(np.unique(fak_pixels)) > 1: fak_thresh = filters.threshold_otsu(fak_pixels)
                        except ValueError: pass
                        if pd.notna(vim_thresh) and pd.notna(fak_thresh):
                            vim_signal_mask = vim_pixels > vim_thresh; fak_signal_mask = fak_pixels > fak_thresh
                            sum_vim_total = np.sum(vim_pixels); sum_fak_total = np.sum(fak_pixels)
                            sum_vim_coloc = np.sum(vim_pixels[fak_signal_mask]); sum_fak_coloc = np.sum(fak_pixels[vim_signal_mask])
                            m1_coeff = sum_vim_coloc / sum_vim_total if sum_vim_total > 0 else 0.0
                            m2_coeff = sum_fak_coloc / sum_fak_total if sum_fak_total > 0 else 0.0
                        mean_fak = np.mean(fak_pixels); mean_vim = np.mean(vim_pixels)
                        products = (fak_pixels - mean_fak) * (vim_pixels - mean_vim)
                        num_positive = np.sum(products > 0); total_pixels = len(products)
                        if total_pixels > 0: icq = (num_positive / total_pixels) - 0.5
                except Exception as coloc_e: print(f"\nWarning: Coloc calculation failed label {label} frame {t}: {coloc_e}")

                features_dict = {'label': label, 'centroid-0': centroid[0], 'centroid-1': centroid[1], 'area_pixels': area_pix, 'bbox-0': bbox[0], 'bbox-1': bbox[1], 'bbox-2': bbox[2], 'bbox-3': bbox[3], 'fak_intensity_mean': fak_mean, 'fak_intensity_max': fak_max, 'fak_intensity_integrated': fak_integ, 'vimentin_intensity_mean': vim_mean, 'vimentin_intensity_integrated': vim_integ, 'pcc': pcc, 'm1_coeff': m1_coeff, 'm2_coeff': m2_coeff, 'icq': icq, 'frame': t, 'y': centroid[0], 'x': centroid[1]}
                if area_per_pixel_um2 is not None: features_dict['area_um2'] = area_pix * area_per_pixel_um2
                else: features_dict['area_um2'] = np.nan
                frame_features_list.append(features_dict)
            if frame_features_list: features_df = pd.DataFrame(frame_features_list); all_features.append(features_df)

    print()

    # --- Step 4: Combine features ---
    if not all_features: print("Error: No features extracted."); sys.exit()
    features_all_frames = pd.concat(all_features, ignore_index=True)
    print(f"\nExtracted features for {len(features_all_frames)} objects across all frames.")

    # --- Step 5: Tracking ---
    print(f"Linking objects into tracks (search_range={cfg_search_range}, memory={cfg_memory})...")
    tp.quiet()
    try:
        tracks = tp.link(features_all_frames, search_range=cfg_search_range, memory=cfg_memory, pos_columns=['y', 'x'], t_column='frame')
        num_raw_tracks = tracks['particle'].nunique(); print(f"Found {num_raw_tracks} raw tracks.")
    except Exception as e: print(f"Error during trackpy linking: {e}"); sys.exit()

    # --- Step 6: Filtering ---
    print(f"Filtering tracks shorter than {cfg_min_track_length} frames...")
    tracks_filtered = tp.filter_stubs(tracks, threshold=cfg_min_track_length)
    num_filtered_tracks = tracks_filtered['particle'].nunique(); print(f"Kept {num_filtered_tracks} tracks after filtering.")
    tracks_filtered = tracks_filtered.reset_index(drop=True)

    if tracks_filtered.empty:
        print("Warning: No tracks remained after filtering.")
        # ... (save unfiltered omitted) ...
    else:
        # --- Step 7: Save Parameters and Results ---
        param_log_path = os.path.join(current_output_dir, f"{output_prefix}_parameters.json"); print(f"Saving analysis parameters to {param_log_path}...") # Use current_output_dir
        analysis_parameters = {'Input File': current_tiff_path, 'Output Directory': current_output_dir, 'Output Prefix': output_prefix, 'Vimentin Channel Index': vimentin_channel_index, 'FAK Channel Index': fak_channel_index, 'Pixel Size (um/pixel)': pixel_size_um if pixel_size_um is not None else "Not Found/Set", 'Frame Interval (sec)': frame_interval_sec, 'Frame Rate (FPS)': frame_rate, '--- Segmentation ---': '--- TUNED PARAMETERS ---', 'Segmentation Method': param_segmentation_method, 'Use Gaussian Blur': param_use_gaussian_blur, 'Gaussian Sigma': param_gaussian_sigma, 'Local Block Size': param_local_block_size if param_segmentation_method == 'local' else 'N/A', 'Local Offset': param_local_offset if param_segmentation_method == 'local' else 'N/A', 'Local Method': param_local_method if param_segmentation_method == 'local' else 'N/A', 'Manual Threshold': param_manual_threshold_value if param_segmentation_method == 'manual' else 'N/A', '--- Cleaning ---': '--- TUNED PARAMETERS ---', 'Min FA Area (pixels)': param_min_fa_area, 'Max FA Area (pixels)': param_max_fa_area, 'Remove Border Objects': param_remove_border_objects, 'Use Opening': param_add_opening, 'Opening Radius': param_opening_radius, 'Use Closing': param_add_closing, 'Closing Radius': param_closing_radius, '--- Tracking ---': '--- FIXED PARAMETERS ---', 'Search Range (pixels)': cfg_search_range, 'Memory (frames)': cfg_memory, '--- Filtering ---': '--- FIXED PARAMETERS ---', 'Min Track Length (frames)': cfg_min_track_length, '--- Measurements ---': '--- CALCULATED ---', 'Colocalization Metrics': 'Pearson Correlation (PCC), Manders Overlap (M1, M2), ICQ'}
        try:
            with open(param_log_path, 'w') as f: json.dump(analysis_parameters, f, indent=4); print(f"✅ Parameters successfully saved.")
        except Exception as e: print(f"Error saving parameters: {e}")

        # Save filtered tracks CSV
        output_csv_path = os.path.join(current_output_dir, f"{output_prefix}_tracks_filtered.csv") # Use current_output_dir
        try:
            output_columns = ['frame', 'particle', 'label', 'y', 'x', 'area_pixels', 'area_um2', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'fak_intensity_mean', 'fak_intensity_max', 'fak_intensity_integrated', 'vimentin_intensity_mean', 'vimentin_intensity_integrated', 'pcc', 'm1_coeff', 'm2_coeff', 'icq']
            final_columns = [col for col in output_columns if col in tracks_filtered.columns]
            tracks_filtered[final_columns].to_csv(output_csv_path, index=False, float_format='%.4f'); print(f"✅ Filtered track data successfully exported to {output_csv_path}")
        except Exception as e: print(f"Error exporting filtered track data: {e}")

        # --- Step 8: Plotting Examples ---
        print("Generating example plots for a few tracks...")
        # Trajectory Plot
        fig_traj, ax_traj = plt.subplots(figsize=(10, 6)); texts = []
        try:
            unique_particle_ids = tracks_filtered['particle'].unique();
            for particle_id in unique_particle_ids:
                particle_data = tracks_filtered[tracks_filtered['particle'] == particle_id].sort_values('frame')
                ax_traj.plot(particle_data['x'], particle_data['y'], marker='o', linestyle='-', markersize=2, linewidth=0.8, alpha=0.8)
                if not particle_data.empty: start_point = particle_data.iloc[0]; texts.append(ax_traj.text(start_point['x'], start_point['y'], str(particle_id), fontsize=6, alpha=0.9))
            ax_traj.set_title(f"FA Trajectories ({num_filtered_tracks} Filtered Tracks)"); ax_traj.set_xlim(0, frame_width); ax_traj.set_ylim(frame_height, 0); ax_traj.set_xlabel("X coordinate (pixels)"); ax_traj.set_ylabel("Y coordinate (pixels)"); ax_traj.set_aspect('equal', adjustable='box'); ax_traj.grid(True, alpha=0.3)
            print("Adjusting text labels to reduce overlap..."); adjust_text(texts, ax=ax_traj, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            traj_plot_path = os.path.join(current_output_dir, f"{output_prefix}_trajectories.png"); fig_traj.savefig(traj_plot_path, dpi=150); print(f"Trajectory plot saved to {traj_plot_path}") # Use current_output_dir
        except Exception as e: print(f"Error during manual trajectory plotting or saving: {e}")
        finally: plt.close(fig_traj)

        # Individual Track Plots
        num_plots_to_show = min(10, num_filtered_tracks)
        if num_plots_to_show > 0:
             unique_tracks = tracks_filtered['particle'].unique()
             print(f"Generating individual plots for first {num_plots_to_show} tracks...")
             for i in range(num_plots_to_show):
                  track_id = unique_tracks[i]
                  track_data = tracks_filtered[tracks_filtered['particle'] == track_id].sort_values('frame')
                  fig_ind, axes = plt.subplots(9, 1, figsize=(10, 20), sharex=True) # 9 subplots now
                  fig_ind.suptitle(f"Track ID: {track_id} (Length: {len(track_data)})")
                  time_sec = track_data['frame'] * frame_interval_sec if frame_interval_sec else track_data['frame']
                  x_label = f'Time (seconds)' if frame_interval_sec else f'Frame'
                  # Plot Vimentin Mean
                  axes[0].plot(time_sec, track_data['vimentin_intensity_mean'], 'r.-', label='Vimentin Mean'); axes[0].set_ylabel('Vim Intensity (Mean)'); axes[0].legend(loc='upper left'); axes[0].grid(True)
                  # Plot Vimentin Integrated
                  axes[1].plot(time_sec, track_data['vimentin_intensity_integrated'], 'm.-', label='Vimentin Integrated'); axes[1].set_ylabel('Vim Intensity (Integ.)'); axes[1].legend(loc='upper left'); axes[1].grid(True)
                  # Plot FAK Mean
                  axes[2].plot(time_sec, track_data['fak_intensity_mean'], 'g.-', label='FAK Mean'); axes[2].set_ylabel('FAK Intensity (Mean)'); axes[2].legend(loc='upper left'); axes[2].grid(True)
                  # Plot FAK Integrated
                  axes[3].plot(time_sec, track_data['fak_intensity_integrated'], 'c.-', label='FAK Integrated'); axes[3].set_ylabel('FAK Intensity (Integ.)'); axes[3].legend(loc='upper left'); axes[3].grid(True)
                  # Plot PCC
                  axes[4].plot(time_sec, track_data['pcc'], 'k.-', label='PCC (Vim/FAK)'); axes[4].set_ylabel('Pearson Corr Coeff'); axes[4].legend(loc='upper left'); axes[4].grid(True); axes[4].set_ylim(-1.1, 1.1)
                  # Plot M1
                  axes[5].plot(time_sec, track_data['m1_coeff'], 'y.-', label='M1 (Vim Overlap FAK)'); axes[5].set_ylabel('Manders M1'); axes[5].legend(loc='upper left'); axes[5].grid(True); axes[5].set_ylim(-0.1, 1.1)
                  # Plot M2
                  axes[6].plot(time_sec, track_data['m2_coeff'], 'b.-', label='M2 (FAK Overlap Vim)'); axes[6].set_ylabel('Manders M2'); axes[6].legend(loc='upper left'); axes[6].grid(True); axes[6].set_ylim(-0.1, 1.1)
                  # Plot ICQ
                  axes[7].plot(time_sec, track_data['icq'], color='orange', marker='.', linestyle='-', label='ICQ (Vim/FAK)'); axes[7].set_ylabel('ICQ'); axes[7].legend(loc='upper left'); axes[7].grid(True); axes[7].set_ylim(-0.6, 0.6)
                  # Plot Area
                  area_col = area_col_name_csv # Use determined column name
                  axes[8].plot(time_sec, track_data[area_col], color='indigo', marker='.', linestyle='-', label='Area'); axes[8].set_ylabel(f'Area ({area_units_label})'); axes[8].legend(loc='upper left'); axes[8].grid(True)
                  axes[8].set_xlabel(x_label + f' (Frame Rate approx. {frame_rate:.3f} FPS)')

                  plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout
                  track_plot_path = os.path.join(current_output_dir, f"{output_prefix}_track_{track_id}_plots.png") # Use current_output_dir
                  try: fig_ind.savefig(track_plot_path, dpi=150)
                  except Exception as e: print(f"Error saving track plot {track_id}: {e}")
                  plt.close(fig_ind)

    print("\nScript finished.")


# --- Run the main function ---
if __name__ == "__main__":
    main()
