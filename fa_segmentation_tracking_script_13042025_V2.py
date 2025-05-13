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

# --- Image Processing & Tracking Libraries ---
from skimage import morphology
from skimage import filters
from skimage import measure
from skimage import segmentation
import trackpy as tp # For tracking
from adjustText import adjust_text

# --- CONFIGURATION ---
# >>> File Paths <<<
# Use raw string (r'...') or forward slashes ('/') for Windows paths
tiff_path = r'C:\Users\Arun\Desktop\TIRF MCHERRYVIM_GFPFAK_TIFF\MEF KO GFP FAK MCHERRY VIM G4.tiff'
# Example: tiff_path = 'C:/Users/Arun/Desktop/TIRF MCHERRYVIM_GFPFAK_TIFF/MEF KO GFP FAK MCHERRY VIM G4.tiff'

# Use raw string or forward slashes
output_dir = r'C:/Users/Arun/Downloads/PhD/Manuscript/TIRF LIVE IMAGING/Python Analysis Results/VIM_FAK_TIRF_seg_track_ANALYSIS_11042025'
output_prefix = "FA_Track_Analysis" # Prefix for output files

# >>> Channel Indices (Based on T, C, Y, X order) <<<
# Ensure these match your data based on the printed shape later
vimentin_channel_index = 0 # Vimentin channel index within axis 1
fak_channel_index = 1      # FAK channel index within axis 1

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
cfg_max_fa_area = 1000       # Maximum pixel area (optional, helps remove huge artifacts)
cfg_remove_border_objects = True # Remove objects touching the image border?
# -- Optional Preprocessing --
cfg_use_gaussian_blur = True
cfg_gaussian_sigma = 1.0     # Initial value if tuning.

# >>> Tracking Parameters (CRITICAL - REQUIRES TUNING) <<<
# trackpy linking parameters
cfg_search_range = 5         # Max distance (pixels) an FA can move between frames
cfg_memory = 1               # How many frames an FA can disappear and still be linked

# >>> Track Filtering Parameters <<<
cfg_min_track_length = 5     # Minimum number of frames a track must exist for to be kept

# --- END CONFIGURATION ---


# --- Helper Functions ---
# [Helper functions preprocess_image, segment_image, clean_mask remain the same]
def preprocess_image(img, use_gaussian=True, sigma=1.0):
    """Optional preprocessing: Gaussian blur."""
    if use_gaussian and sigma > 0: # Only blur if sigma > 0
        return filters.gaussian(img, sigma=sigma, preserve_range=True)
    return img

def segment_image(img, method='local', block_size=35, offset=0, local_method='gaussian', manual_thresh=50):
    """
    Segment image using specified method.
    Now defaults to and primarily supports 'local' thresholding.
    """
    if method == 'local':
        # Ensure block size is odd and >= 3
        block_size = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        try:
            # Calculate local thresholds
            local_thresh = filters.threshold_local(img, block_size,
                                                  method=local_method, offset=offset)
            # Apply threshold
            binary = img > local_thresh
        except ValueError as e:
            print(f"\nError in threshold_local (block_size={block_size}): {e}. Check block_size vs image size.")
            # Fallback: try global Otsu if local fails
            print("Falling back to Otsu threshold.")
            try:
                thresh = filters.threshold_otsu(img)
                binary = img > thresh
            except ValueError:
                print("Otsu fallback failed. Using median threshold.")
                binary = img > np.median(img)

    elif method == 'otsu':
        try:
            thresh = filters.threshold_otsu(img)
            binary = img > thresh
        except ValueError:
             print("\nWarning: Otsu's method failed (likely due to near-uniform image). Trying median threshold.")
             binary = img > np.median(img) # Fallback threshold
    elif method == 'manual':
        binary = img > manual_thresh
    else:
        raise ValueError("Unknown segmentation method: {}".format(method))
    return binary.astype(bool) # Ensure boolean output

def clean_mask(mask, min_area=10, max_area=None, remove_border=True):
    """Clean binary mask using morphological operations."""
    # Ensure mask is boolean
    mask = mask.astype(bool)
    # Remove small objects
    if min_area > 0:
        cleaned_mask = morphology.remove_small_objects(mask, min_size=min_area)
    else:
        cleaned_mask = mask.copy()

    # Optional: Remove large objects
    if max_area is not None and max_area > 0:
         labels = measure.label(cleaned_mask)
         props = measure.regionprops(labels)
         large_labels = {prop.label for prop in props if prop.area > max_area}
         if large_labels:
              large_mask = np.isin(labels, list(large_labels))
              cleaned_mask[large_mask] = False

    # Optional: Remove objects touching the border
    if remove_border:
        cleaned_mask = segmentation.clear_border(cleaned_mask)

    return cleaned_mask

# --- Interactive Tuning Function ---
def interactive_parameter_tuning(img_to_tune, initial_params):
    """
    Displays an interactive window to tune segmentation parameters,
    focused on local thresholding.
    """
    print("\n--- Interactive Segmentation Tuning ('local' method) ---")
    print("Adjust sliders to optimize segmentation on the preview frame.")
    print("Segmentation boundaries will be shown in GREEN.")
    print("Press 'ENTER' to accept current parameters and continue.")
    print("Press 'ESC' to exit script.")

    window_name = "Local Threshold Tuning (ENTER=Accept, ESC=Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    # Normalize image for display (0-255)
    display_img_norm = cv2.normalize(img_to_tune, None, 0, 255, cv2.NORM_MINMAX)
    display_img_8u = display_img_norm.astype(np.uint8)
    display_img_bgr = cv2.cvtColor(display_img_8u, cv2.COLOR_GRAY2BGR)

    # --- Trackbar Setup ---
    def on_trackbar(val): pass # Dummy callback

    current_params = initial_params.copy()
    # Ensure method is 'local' for this tuner
    current_params['segmentation_method'] = 'local'

    # Gaussian Sigma (Scale 0-50 maps to 0.0-5.0)
    cv2.createTrackbar('Sigma*10', window_name, int(current_params['gaussian_sigma'] * 10), 50, on_trackbar)
    # Min Area (0-500 pixels, adjust range if needed)
    cv2.createTrackbar('Min Area', window_name, current_params['min_fa_area'], 500, on_trackbar)
    # Local Block Size (Trackbar 1-100 maps to odd sizes 3-201) - Adjust max range if needed
    cv2.createTrackbar('Local Block/2', window_name, (current_params['local_block_size'] - 1) // 2, 100, on_trackbar)
    # Local Offset (Trackbar 0-50 maps to -25 to +25) - Adjust range/center if needed
    offset_center = 25
    cv2.createTrackbar('Local Offset+'+str(offset_center), window_name, current_params['local_offset'] + offset_center, 50, on_trackbar)

    while True:
        # --- Get current trackbar values ---
        sigma_val = cv2.getTrackbarPos('Sigma*10', window_name) / 10.0
        min_area_val = cv2.getTrackbarPos('Min Area', window_name)
        block_size_val = max(3, 2 * cv2.getTrackbarPos('Local Block/2', window_name) + 1) # Ensure odd >= 3
        offset_val = cv2.getTrackbarPos('Local Offset+'+str(offset_center), window_name) - offset_center

        # Update parameters based on trackbar positions
        current_params['gaussian_sigma'] = sigma_val
        current_params['min_fa_area'] = min_area_val
        current_params['local_block_size'] = block_size_val
        current_params['local_offset'] = offset_val

        # --- Apply processing with current parameters ---
        try:
            preprocessed = preprocess_image(img_to_tune,
                                            use_gaussian=current_params['use_gaussian_blur'],
                                            sigma=current_params['gaussian_sigma'])

            # Use 'local' method here
            binary_mask = segment_image(preprocessed,
                                        method='local',
                                        block_size=current_params['local_block_size'],
                                        offset=current_params['local_offset'],
                                        local_method=current_params['local_method']) # Use configured method ('gaussian')

            cleaned_mask = clean_mask(binary_mask,
                                      min_area=current_params['min_fa_area'],
                                      max_area=current_params.get('max_fa_area'),
                                      remove_border=current_params['remove_border_objects'])
        except Exception as e:
            print(f"\nError during preview processing: {e}")
            cv2.imshow(window_name, display_img_bgr) # Show original on error
            key = cv2.waitKey(50) & 0xFF
            if key == 27: sys.exit()
            elif key == 13: break
            continue

        # --- Create preview image ---
        preview_img = display_img_bgr.copy()
        contours, _ = cv2.findContours(cleaned_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 1)

        # Add text showing current parameter values
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_y = 20
        cv2.putText(preview_img, f"Sigma: {current_params['gaussian_sigma']:.1f}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA); text_y += 20
        cv2.putText(preview_img, f"Min Area: {current_params['min_fa_area']}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA); text_y += 20
        cv2.putText(preview_img, f"Local Block: {current_params['local_block_size']}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA); text_y += 20
        cv2.putText(preview_img, f"Local Offset: {current_params['local_offset']}", (10, text_y), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # --- Display preview ---
        cv2.imshow(window_name, preview_img)

        # --- Handle key press ---
        key = cv2.waitKey(50) & 0xFF # Wait 50ms for key press
        if key == 27: # Esc key
            print("Parameter tuning cancelled by user. Exiting.")
            cv2.destroyAllWindows(); sys.exit()
        elif key == 13: # Enter key
            print("Parameters accepted by user.")
            break # Exit loop

    cv2.destroyAllWindows()
    for i in range(5): cv2.waitKey(1)
    return current_params


# --- MAIN SCRIPT ---

def main():
    print(f"Starting FA analysis for: {tiff_path}")
    print(f"Output directory: {output_dir}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        sys.exit()

    # --- Load Data ---
    print("Loading TIFF stack...")
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            image_stack = tif.asarray()
            # Attempt to get frame interval from metadata
            frame_interval_sec = None
            try:
                ij_meta = tif.imagej_metadata
                if ij_meta and isinstance(ij_meta, dict) and 'finterval' in ij_meta:
                    frame_interval_sec = float(ij_meta['finterval'])
                    print(f"Found frame interval in ImageJ metadata (finterval): {frame_interval_sec:.4f} sec")
            except Exception: pass
            if frame_interval_sec is None:
                 try:
                     if tif.pages and len(tif.pages) > 0:
                         desc_tag = tif.pages[0].tags.get(270)
                         description = desc_tag.value if desc_tag else None
                         if description:
                              match = re.search(r"(?:interval|deltaT)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(ms|s)?", str(description), re.IGNORECASE)
                              if match:
                                   value = float(match.group(1))
                                   unit = match.group(2)
                                   frame_interval_sec = value / 1000.0 if unit and unit.lower() == 'ms' else value
                                   print(f"Found potential interval in ImageDescription: {frame_interval_sec:.4f} sec")
                 except Exception: pass

    except Exception as e:
        print(f"Error reading TIFF file: {e}")
        sys.exit()

    # --- Verify Dimensions (Assuming T, C, Y, X) ---
    print(f"Loaded stack shape: {image_stack.shape}")
    if image_stack.ndim != 4:
        print(f"Error: Expected 4D stack (T, C, Y, X), but got {image_stack.ndim}D.")
        sys.exit()

    num_frames = image_stack.shape[0]
    num_channels = image_stack.shape[1]
    frame_height = image_stack.shape[2]
    frame_width = image_stack.shape[3]
    print(f"Interpreted as: T={num_frames}, C={num_channels}, H={frame_height}, W={frame_width}")

    if not (0 <= vimentin_channel_index < num_channels and 0 <= fak_channel_index < num_channels):
         print(f"Error: Channel index out of bounds (Vim: {vimentin_channel_index}, FAK: {fak_channel_index}, Total: {num_channels})")
         sys.exit()

    # --- Set Frame Rate ---
    if frame_interval_sec is not None and frame_interval_sec > 0:
        frame_rate = 1.0 / frame_interval_sec
        print(f"Using automatically detected frame interval: {frame_interval_sec:.4f} sec (Rate: {frame_rate:.3f} FPS)")
    else:
        print("Warning: Could not automatically detect frame interval. Set manually if known, otherwise using 1.0 sec.")
        frame_interval_sec = 1.0 # Default fallback interval
        frame_rate = 1.0 / frame_interval_sec
        print(f"Using default interval of {frame_interval_sec:.1f} sec (Rate: {frame_rate:.3f} FPS)")


    # --- >>> Interactive Parameter Tuning <<< ---
    # <<< FIX: Read config values into local variables FIRST >>>
    param_segmentation_method = cfg_segmentation_method
    param_local_block_size = cfg_local_block_size
    param_local_offset = cfg_local_offset
    param_local_method = cfg_local_method
    param_manual_threshold_value = cfg_manual_threshold_value
    param_min_fa_area = cfg_min_fa_area
    param_max_fa_area = cfg_max_fa_area
    param_remove_border_objects = cfg_remove_border_objects
    param_use_gaussian_blur = cfg_use_gaussian_blur
    param_gaussian_sigma = cfg_gaussian_sigma

    # Create initial params dict from these local variables
    initial_params = {
        'segmentation_method': param_segmentation_method,
        'local_block_size': param_local_block_size,
        'local_offset': param_local_offset,
        'local_method': param_local_method,
        'manual_threshold_value': param_manual_threshold_value, # Include even if method isn't manual initially
        'min_fa_area': param_min_fa_area,
        'max_fa_area': param_max_fa_area,
        'remove_border_objects': param_remove_border_objects,
        'use_gaussian_blur': param_use_gaussian_blur,
        'gaussian_sigma': param_gaussian_sigma
    }
    # <<< END FIX >>>

    # Use the first FAK frame for tuning
    first_fak_frame = image_stack[0, fak_channel_index, :, :]

    # Call the interactive tuning function
    approved_params = interactive_parameter_tuning(first_fak_frame, initial_params)

    # <<< FIX: Update the SAME local variables with approved parameters >>>
    param_segmentation_method = approved_params['segmentation_method'] # Stays 'local'
    param_gaussian_sigma = approved_params['gaussian_sigma']
    param_min_fa_area = approved_params['min_fa_area']
    param_local_block_size = approved_params['local_block_size']
    param_local_offset = approved_params['local_offset']
    # Keep other params from config unless they were also made tunable
    param_local_method = approved_params['local_method']
    param_max_fa_area = approved_params['max_fa_area']
    param_remove_border_objects = approved_params['remove_border_objects']
    param_use_gaussian_blur = approved_params['use_gaussian_blur']
    # Include manual threshold in case method was changed (though tuner focuses on local)
    param_manual_threshold_value = approved_params['manual_threshold_value']
    # <<< END FIX >>>


    print("\n--- Continuing analysis with approved parameters: ---")
    print(f" Segmentation Method: {param_segmentation_method} ({param_local_method if param_segmentation_method == 'local' else 'N/A'})") # Adjusted print
    print(f" Gaussian Sigma: {param_gaussian_sigma:.1f}")
    print(f" Min FA Area: {param_min_fa_area}")
    if param_segmentation_method == 'local':
        print(f" Local Block Size: {param_local_block_size}")
        print(f" Local Offset: {param_local_offset}")
    elif param_segmentation_method == 'manual':
         print(f" Manual Threshold: {param_manual_threshold_value}")
    print("--------------------------------------------------")

    # --- Process Frames: Segmentation & Feature Extraction ---
    all_features = [] # List to store features from all frames

    print("\nProcessing frames for FA segmentation and feature extraction...")
    for t in range(num_frames):
        # Use print with carriage return to update line (optional)
        print(f" Processing frame {t+1}/{num_frames}...", end='\r')

        # Extract channels for current frame
        fak_frame = image_stack[t, fak_channel_index, :, :]
        vimentin_frame = image_stack[t, vimentin_channel_index, :, :]

        # <<< FIX: Use updated local parameter variables >>>
        # 1. Preprocess FAK image (using approved sigma)
        fak_processed = preprocess_image(fak_frame, use_gaussian=param_use_gaussian_blur, sigma=param_gaussian_sigma)

        # 2. Segment FAK image (using approved method and params)
        fak_binary_mask = segment_image(fak_processed, method=param_segmentation_method,
                                        block_size=param_local_block_size, offset=param_local_offset,
                                        local_method=param_local_method, manual_thresh=param_manual_threshold_value)

        # 3. Clean binary mask (using approved min_area)
        fak_cleaned_mask = clean_mask(fak_binary_mask, min_area=param_min_fa_area,
                                      max_area=param_max_fa_area, remove_border=param_remove_border_objects)
        # <<< END FIX >>>

        # 4. Label objects
        label_image, num_labels = measure.label(fak_cleaned_mask, background=0, connectivity=2, return_num=True)

        # 5. Extract features for labeled objects if any were found
        if num_labels > 0:
            props_to_calculate = ('label', 'centroid', 'area', 'bbox',
                                  'mean_intensity', # This will be FAK intensity
                                  'max_intensity')

            features_fak = measure.regionprops_table(label_image, intensity_image=fak_frame,
                                                     properties=props_to_calculate, cache=True)
            features_df = pd.DataFrame(features_fak)
            features_df.rename(columns={'mean_intensity': 'fak_intensity_mean',
                                        'max_intensity': 'fak_intensity_max'}, inplace=True)

            try:
                 vimentin_props_table = measure.regionprops_table(label_image, intensity_image=vimentin_frame,
                                                                  properties=('label', 'mean_intensity'), cache=True)
                 vimentin_df = pd.DataFrame(vimentin_props_table)
                 vimentin_df.rename(columns={'mean_intensity': 'vimentin_intensity_mean'}, inplace=True)
                 features_df = pd.merge(features_df, vimentin_df, on='label', how='left')
            except Exception as e:
                 print(f"\n  Error calculating Vimentin intensity for frame {t}: {e}. Setting to NaN.")
                 features_df['vimentin_intensity_mean'] = np.nan

            features_df['frame'] = t
            features_df['y'] = features_df['centroid-0']
            features_df['x'] = features_df['centroid-1']
            all_features.append(features_df)

    print() # Newline after frame processing loop

    # --- Combine features from all frames ---
    if not all_features:
        print("Error: No features were extracted from any frame. Check segmentation parameters.")
        sys.exit()

    features_all_frames = pd.concat(all_features, ignore_index=True)
    print(f"\nExtracted features for {len(features_all_frames)} objects across all frames.")

    # --- Perform Tracking ---
    # <<< FIX: Use updated local tracking parameter variables >>>
    print(f"Linking objects into tracks using trackpy (search_range={cfg_search_range}, memory={cfg_memory})...")
    tp.quiet()
    try:
        tracks = tp.link(features_all_frames, search_range=cfg_search_range, memory=cfg_memory,
                         pos_columns=['y', 'x'], t_column='frame')
        num_raw_tracks = tracks['particle'].nunique()
        print(f"Found {num_raw_tracks} raw tracks.")
    except Exception as e:
         print(f"Error during trackpy linking: {e}")
         print("Saving features before tracking for debugging...")
         features_all_frames.to_csv(os.path.join(output_dir, f"{output_prefix}_DEBUG_features_before_tracking.csv"), index=False)
         sys.exit()
    # <<< END FIX >>>

    # --- Filter Tracks ---
    # <<< FIX: Use updated local filtering parameter variable >>>
    print(f"Filtering tracks shorter than {cfg_min_track_length} frames...")
    tracks_filtered = tp.filter_stubs(tracks, threshold=cfg_min_track_length)
    num_filtered_tracks = tracks_filtered['particle'].nunique()
    print(f"Kept {num_filtered_tracks} tracks after filtering.")
    tracks_filtered = tracks_filtered.reset_index(drop=True)
    # <<< END FIX >>>


    if tracks_filtered.empty:
        print("Warning: No tracks remained after filtering.")
        tracks.to_csv(os.path.join(output_dir, f"{output_prefix}_tracks_unfiltered.csv"), index=False, float_format='%.4f')
    else:
        # --- Save Results ---
        output_csv_path = os.path.join(output_dir, f"{output_prefix}_tracks_filtered.csv")
        try:
            output_columns = [
                'frame', 'particle', 'label', 'y', 'x', 'area', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
                'fak_intensity_mean', 'fak_intensity_max', 'vimentin_intensity_mean'
            ]
            final_columns = [col for col in output_columns if col in tracks_filtered.columns]
            tracks_filtered[final_columns].to_csv(output_csv_path, index=False, float_format='%.4f')
            print(f"✅ Filtered track data successfully exported to {output_csv_path}")
        except Exception as e:
            print(f"Error exporting filtered track data to CSV: {e}")
            # ... (fallback save omitted for brevity) ...

        # --- Plotting Examples ---
        print("Generating example plots for a few tracks...")
        # Trajectory Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        texts = []
        try:
            unique_particle_ids = tracks_filtered['particle'].unique()
            for particle_id in unique_particle_ids:
                particle_data = tracks_filtered[tracks_filtered['particle'] == particle_id].sort_values('frame')
                ax.plot(particle_data['x'], particle_data['y'],
                        marker='o', linestyle='-', markersize=2, linewidth=0.8, alpha=0.8)
                if not particle_data.empty:
                    start_point = particle_data.iloc[0]
                    texts.append(ax.text(start_point['x'], start_point['y'], str(particle_id),
                                         fontsize=6, alpha=0.9))
            ax.set_title(f"FA Trajectories ({num_filtered_tracks} Filtered Tracks)")
            ax.set_xlim(0, frame_width); ax.set_ylim(frame_height, 0)
            ax.set_xlabel("X coordinate (pixels)"); ax.set_ylabel("Y coordinate (pixels)")
            ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.3)
            print("Adjusting text labels to reduce overlap...")
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            traj_plot_path = os.path.join(output_dir, f"{output_prefix}_trajectories.png")
            plt.savefig(traj_plot_path, dpi=150)
            print(f"Trajectory plot saved to {traj_plot_path}")
        except Exception as e:
            print(f"Error during manual trajectory plotting or saving: {e}")
        finally:
            plt.close(fig)

        # Individual Track Plots
        num_plots_to_show = min(10, num_filtered_tracks)
        if num_plots_to_show > 0:
             unique_tracks = tracks_filtered['particle'].unique()
             print(f"Generating individual plots for first {num_plots_to_show} tracks...")
             for i in range(num_plots_to_show):
                  track_id = unique_tracks[i]
                  track_data = tracks_filtered[tracks_filtered['particle'] == track_id].sort_values('frame')
                  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                  fig.suptitle(f"Track ID: {track_id} (Length: {len(track_data)})")
                  time_sec = track_data['frame'] * frame_interval_sec if frame_interval_sec else track_data['frame']
                  x_label = f'Time (seconds)' if frame_interval_sec else f'Frame'
                  # Plot Vimentin
                  axes[0].plot(time_sec, track_data['vimentin_intensity_mean'], 'r.-', label='Vimentin Mean')
                  axes[0].set_ylabel('Vimentin Intensity'); axes[0].legend(loc='upper left'); axes[0].grid(True)
                  # Plot FAK
                  axes[1].plot(time_sec, track_data['fak_intensity_mean'], 'g.-', label='FAK Mean')
                  axes[1].set_ylabel('FAK Intensity'); axes[1].legend(loc='upper left'); axes[1].grid(True)
                  # Plot Area
                  axes[2].plot(time_sec, track_data['area'], 'b.-', label='Area')
                  axes[2].set_ylabel('Area (pixels)'); axes[2].legend(loc='upper left'); axes[2].grid(True)
                  axes[2].set_xlabel(x_label + f' (Frame Rate approx. {frame_rate:.3f} FPS)')

                  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                  track_plot_path = os.path.join(output_dir, f"{output_prefix}_track_{track_id}_plots.png")
                  try: plt.savefig(track_plot_path, dpi=150)
                  except Exception as e: print(f"Error saving track plot {track_id}: {e}")
                  plt.close(fig)

    print("\nScript finished.")


# --- Run the main function ---
if __name__ == "__main__":
    main()
