import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import tifffile
from tifffile import TiffFile
import re

# --- Image Processing & Tracking Libraries ---
from skimage import morphology
from skimage import filters
from skimage import measure
from skimage import segmentation
import trackpy as tp # For tracking
from adjustText import adjust_text # <<< Import adjustText

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

# >>> Segmentation Parameters (CRITICAL - REQUIRES TUNING) <<<
# Options: 'otsu', 'adaptive', 'manual'
segmentation_method = 'otsu'
# -- Parameters for 'adaptive' method --
adaptive_block_size = 31 # Must be odd, larger for uneven background
adaptive_offset = 2      # Constant subtracted from weighted mean
# -- Parameters for 'manual' method --
manual_threshold_value = 50 # Set manually if using 'manual'
# -- Morphological Cleaning --
min_fa_area = 10         # Minimum pixel area to keep an object after segmentation
max_fa_area = 1000       # Maximum pixel area (optional, helps remove huge artifacts)
remove_border_objects = True # Remove objects touching the image border?
# -- Optional Preprocessing --
use_gaussian_blur = True
gaussian_sigma = 1.0

# >>> Tracking Parameters (CRITICAL - REQUIRES TUNING) <<<
# trackpy linking parameters
search_range = 5         # Max distance (pixels) an FA can move between frames
memory = 1               # How many frames an FA can disappear and still be linked

# >>> Track Filtering Parameters <<<
min_track_length = 5     # Minimum number of frames a track must exist for to be kept

# --- END CONFIGURATION ---


# --- Helper Functions ---
# [Helper functions preprocess_image, segment_image, clean_mask remain the same - omitted for brevity]
def preprocess_image(img, use_gaussian=True, sigma=1.0):
    """Optional preprocessing: Gaussian blur."""
    if use_gaussian:
        return filters.gaussian(img, sigma=sigma, preserve_range=True)
    return img

def segment_image(img, method='otsu', block_size=31, offset=2, manual_thresh=50):
    """Segment image using specified method."""
    if method == 'otsu':
        thresh = filters.threshold_otsu(img)
        binary = img > thresh
    elif method == 'adaptive':
        # Adaptive threshold needs uint8 or uint16 input
        if img.dtype == np.float32 or img.dtype == np.float64:
             # Scale to uint16 range if float
             img_scaled = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX)
             img_int = img_scaled.astype(np.uint16)
        elif img.dtype == np.uint8 or img.dtype == np.uint16:
             img_int = img
        else:
             print(f"Warning: Unsupported dtype {img.dtype} for adaptive threshold. Trying Otsu.")
             thresh = filters.threshold_otsu(img)
             binary = img > thresh
             return binary

        # Ensure block size is odd and > 1
        block_size = max(3, block_size if block_size % 2 == 1 else block_size + 1)
        binary_inv = cv2.adaptiveThreshold(img_int, 1, # Output 0 or 1
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV,
                                           block_size, offset)
        binary = binary_inv == 0 # Invert back so objects are True/1
    elif method == 'manual':
        binary = img > manual_thresh
    else:
        raise ValueError("Unknown segmentation method: {}".format(method))
    return binary

def clean_mask(mask, min_area=10, max_area=None, remove_border=True):
    """Clean binary mask using morphological operations."""
    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(mask, min_size=min_area)

    # Optional: Remove large objects
    if max_area is not None:
         # This is a simple way, better might be labeling then filtering by area
         cleaned_mask = morphology.remove_small_objects(cleaned_mask, min_size=1) # Keep all > 0 initially
         # A better approach for max_area requires labeling first:
         # labels = measure.label(cleaned_mask)
         # props = measure.regionprops(labels)
         # for prop in props:
         #     if prop.area > max_area:
         #         cleaned_mask[labels == prop.label] = 0 # Set large objects to background


    # Optional: Remove objects touching the border
    if remove_border:
        # <<< Use segmentation.clear_border >>>
        cleaned_mask = segmentation.clear_border(cleaned_mask)

    return cleaned_mask

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
            # Attempt to get frame interval from metadata (copied from previous script)
            frame_interval_sec = None
            try:
                ij_meta = tif.imagej_metadata
                if ij_meta and isinstance(ij_meta, dict) and 'finterval' in ij_meta:
                    frame_interval_sec = float(ij_meta['finterval'])
                    print(f"Found frame interval in ImageJ metadata (finterval): {frame_interval_sec:.4f} sec")
            except Exception: pass # Ignore errors here
            if frame_interval_sec is None:
                 try:
                     # Check if pages exist before accessing
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
                 except Exception: pass # Ignore errors here

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


    # --- Process Frames: Segmentation & Feature Extraction ---
    all_features = [] # List to store features from all frames

    print("\nProcessing frames for FA segmentation and feature extraction...")
    for t in range(num_frames):
        # Use print with carriage return to update line (optional)
        print(f" Processing frame {t+1}/{num_frames}...", end='\r')

        # Extract channels for current frame
        fak_frame = image_stack[t, fak_channel_index, :, :]
        vimentin_frame = image_stack[t, vimentin_channel_index, :, :]

        # 1. Preprocess FAK image (optional)
        fak_processed = preprocess_image(fak_frame, use_gaussian=use_gaussian_blur, sigma=gaussian_sigma)

        # 2. Segment FAK image
        fak_binary_mask = segment_image(fak_processed, method=segmentation_method,
                                        block_size=adaptive_block_size, offset=adaptive_offset,
                                        manual_thresh=manual_threshold_value)

        # 3. Clean binary mask
        fak_cleaned_mask = clean_mask(fak_binary_mask, min_area=min_fa_area,
                                      max_area=max_fa_area, remove_border=remove_border_objects)

        # 4. Label objects
        # Use connectivity=2 for 8-connectivity (usually better for diagonal touching)
        label_image, num_labels = measure.label(fak_cleaned_mask, background=0, connectivity=2, return_num=True)
        # print(f"  Found {num_labels} potential FA objects.") # Can be verbose

        # 5. Extract features for labeled objects if any were found
        if num_labels > 0:
            props_to_calculate = ('label', 'centroid', 'area', 'bbox',
                                  'mean_intensity', # This will be FAK intensity
                                  'max_intensity')

            # Calculate props using FAK intensity image
            # Use cache=False if memory is an issue, but might be slower
            features_fak = measure.regionprops_table(label_image, intensity_image=fak_frame,
                                                     properties=props_to_calculate, cache=True)
            features_df = pd.DataFrame(features_fak)
            features_df.rename(columns={'mean_intensity': 'fak_intensity_mean',
                                        'max_intensity': 'fak_intensity_max'}, inplace=True)

            # Calculate Vimentin intensity separately using the same labels
            try:
                 # Use regionprops_table directly for efficiency if possible
                 vimentin_props_table = measure.regionprops_table(label_image, intensity_image=vimentin_frame,
                                                                  properties=('label', 'mean_intensity'), cache=True)
                 vimentin_df = pd.DataFrame(vimentin_props_table)
                 vimentin_df.rename(columns={'mean_intensity': 'vimentin_intensity_mean'}, inplace=True)

                 # Merge based on label - robust way to add the column
                 features_df = pd.merge(features_df, vimentin_df, on='label', how='left')

            except Exception as e:
                 print(f"\n  Error calculating Vimentin intensity for frame {t}: {e}. Setting to NaN.")
                 features_df['vimentin_intensity_mean'] = np.nan


            # Add frame number (important for trackpy)
            features_df['frame'] = t

            # Add centroid coordinates (trackpy uses 'y', 'x')
            features_df['y'] = features_df['centroid-0']
            features_df['x'] = features_df['centroid-1']

            # Append features for this frame to the main list
            all_features.append(features_df)
        # else: print(f"  No objects found after cleaning in frame {t}.") # Optional message

        # --- Optional: Save intermediate images for debugging ---
        # if t % 20 == 0: # Save every 20 frames
        #     debug_path_mask = os.path.join(output_dir, f"DEBUG_mask_f{t:03d}.png")
        #     debug_path_labels = os.path.join(output_dir, f"DEBUG_labels_f{t:03d}.png")
        #     # Scale label image for visibility
        #     labels_display = cv2.normalize(label_image, None, 0, 255, cv2.NORM_MINMAX)
        #     cv2.imwrite(debug_path_mask, fak_cleaned_mask.astype(np.uint8) * 255)
        #     cv2.imwrite(debug_path_labels, labels_display.astype(np.uint8))
        #     # print(f"  Saved debug images for frame {t}") # Can be verbose
        # --- End Optional Debug ---

    print() # Newline after frame processing loop

    # --- Combine features from all frames ---
    if not all_features:
        print("Error: No features were extracted from any frame. Check segmentation parameters.")
        sys.exit()

    features_all_frames = pd.concat(all_features, ignore_index=True)
    print(f"\nExtracted features for {len(features_all_frames)} objects across all frames.")

    # --- Perform Tracking ---
    print(f"Linking objects into tracks using trackpy (search_range={search_range}, memory={memory})...")
    # Ensure DataFrame has 'x', 'y', 'frame' columns
    tp.quiet() # Suppress trackpy output
    try:
        tracks = tp.link(features_all_frames, search_range=search_range, memory=memory,
                         pos_columns=['y', 'x'], t_column='frame')
        # Note: tp.link_df might be deprecated or less flexible than link + filter later
        # tracks = tp.link_df(features_all_frames, search_range=search_range, memory=memory,
        #                    pos_columns=['y', 'x'], t_column='frame')

        num_raw_tracks = tracks['particle'].nunique()
        print(f"Found {num_raw_tracks} raw tracks.")

    except Exception as e:
         print(f"Error during trackpy linking: {e}")
         print("Saving features before tracking for debugging...")
         features_all_frames.to_csv(os.path.join(output_dir, f"{output_prefix}_DEBUG_features_before_tracking.csv"), index=False)
         sys.exit()


    # --- Filter Tracks ---
    print(f"Filtering tracks shorter than {min_track_length} frames...")
    # Use trackpy's filtering function
    tracks_filtered = tp.filter_stubs(tracks, threshold=min_track_length)
    # Alternatively, manual filtering:
    # track_lengths = tracks.groupby('particle')['frame'].nunique()
    # tracks_filtered = tracks[tracks['particle'].isin(track_lengths[track_lengths >= min_track_length].index)]

    num_filtered_tracks = tracks_filtered['particle'].nunique()
    print(f"Kept {num_filtered_tracks} tracks after filtering.")

    # Reset index after filtering to prevent potential issues later
    tracks_filtered = tracks_filtered.reset_index(drop=True)


    if tracks_filtered.empty:
        print("Warning: No tracks remained after filtering.")
        # Save unfiltered tracks for inspection
        tracks.to_csv(os.path.join(output_dir, f"{output_prefix}_tracks_unfiltered.csv"), index=False, float_format='%.4f')
    else:
        # --- Save Results ---
        # Save filtered tracks
        output_csv_path = os.path.join(output_dir, f"{output_prefix}_tracks_filtered.csv")
        try:
            # Select and order columns for output
            output_columns = [
                'frame', 'particle', 'label', 'y', 'x', 'area', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3',
                'fak_intensity_mean', 'fak_intensity_max', 'vimentin_intensity_mean'
            ]
            # Ensure all expected columns exist before selecting
            final_columns = [col for col in output_columns if col in tracks_filtered.columns]
            tracks_filtered[final_columns].to_csv(output_csv_path, index=False, float_format='%.4f')
            print(f"✅ Filtered track data successfully exported to {output_csv_path}")
        except Exception as e:
            print(f"Error exporting filtered track data to CSV: {e}")
            print("Attempting to save with all columns...")
            try:
                 # Ensure index is reset before saving fallback too
                 tracks_filtered.reset_index(drop=True).to_csv(output_csv_path.replace('.csv', '_all_cols.csv'), index=False, float_format='%.4f')
            except: pass


        # --- Optional: Plotting Example (Plot trajectory and intensity for a few tracks) ---
        print("Generating example plots for a few tracks...")

        # <<< Plot trajectories using Matplotlib loop >>>
        fig, ax = plt.subplots(figsize=(10, 6))
        texts = [] # <<< Initialize list to store text objects
        try:
            unique_particle_ids = tracks_filtered['particle'].unique()
            # Optional: Create a colormap for visual distinction
            # colors = plt.cm.viridis(np.linspace(0, 1, len(unique_particle_ids)))

            # Iterate through each particle track
            # for idx, particle_id in enumerate(unique_particle_ids): # Use enumerate if using colors
            for particle_id in unique_particle_ids:
                particle_data = tracks_filtered[tracks_filtered['particle'] == particle_id].sort_values('frame')
                # Plot x vs y for this particle
                # color = colors[idx] # Assign color if using colormap
                # <<< CHANGE HERE: Adjust plot parameters >>>
                ax.plot(particle_data['x'], particle_data['y'],
                        marker='o',      # Use small circles
                        linestyle='-',   # Solid line
                        markersize=2,    # Slightly larger markers
                        linewidth=0.8,   # Explicit line width
                        alpha=0.8)       # Slightly less transparent
                # <<< END CHANGE >>>

                # <<< Collect TEXT LABEL for adjustText >>>
                # Get the starting point (first frame) of the track
                if not particle_data.empty:
                    start_point = particle_data.iloc[0]
                    # Create text object and add to list
                    texts.append(ax.text(start_point['x'], start_point['y'], str(particle_id),
                                         fontsize=6, alpha=0.9)) # Store text object
                # <<< END TEXT LABEL COLLECTION >>>

            ax.set_title(f"FA Trajectories ({num_filtered_tracks} Filtered Tracks)")
            ax.set_xlim(0, frame_width)
            ax.set_ylim(frame_height, 0) # Inverted Y for image coordinates
            ax.set_xlabel("X coordinate (pixels)")
            ax.set_ylabel("Y coordinate (pixels)")
            ax.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio
            ax.grid(True, alpha=0.3)

            # <<< Call adjust_text AFTER the loop >>>
            print("Adjusting text labels to reduce overlap...")
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)) # Add arrows
            # <<< END adjust_text call >>>

            traj_plot_path = os.path.join(output_dir, f"{output_prefix}_trajectories.png")
            plt.savefig(traj_plot_path, dpi=150)
            print(f"Trajectory plot saved to {traj_plot_path}")
        except Exception as e:
            print(f"Error during manual trajectory plotting or saving: {e}")
        finally:
            plt.close(fig) # Ensure figure is closed
        # <<< END Manual Trajectory Plot >>>


        # Plot intensity for first N tracks
        num_plots_to_show = min(10, num_filtered_tracks) # Show more examples
        if num_plots_to_show > 0:
             # Get unique track IDs
             unique_tracks = tracks_filtered['particle'].unique()

             print(f"Generating individual plots for first {num_plots_to_show} tracks...")
             for i in range(num_plots_to_show):
                  track_id = unique_tracks[i]
                  # Sort values should work now index is reset
                  track_data = tracks_filtered[tracks_filtered['particle'] == track_id].sort_values('frame')

                  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
                  fig.suptitle(f"Track ID: {track_id} (Length: {len(track_data)})")

                  # Add time in seconds to x-axis if available
                  if frame_interval_sec:
                       time_sec = track_data['frame'] * frame_interval_sec
                       x_label = f'Time (seconds)'
                  else:
                       time_sec = track_data['frame'] # Fallback to frame number
                       x_label = f'Frame'


                  # Vimentin Intensity
                  axes[0].plot(time_sec, track_data['vimentin_intensity_mean'], 'r.-', label='Vimentin Mean')
                  axes[0].set_ylabel('Vimentin Intensity')
                  axes[0].legend(loc='upper left')
                  axes[0].grid(True)

                  # FAK Intensity
                  axes[1].plot(time_sec, track_data['fak_intensity_mean'], 'g.-', label='FAK Mean')
                  axes[1].set_ylabel('FAK Intensity')
                  axes[1].legend(loc='upper left')
                  axes[1].grid(True)

                  # Area
                  axes[2].plot(time_sec, track_data['area'], 'b.-', label='Area')
                  axes[2].set_ylabel('Area (pixels)')
                  axes[2].set_xlabel(x_label + f' (Frame Rate approx. {frame_rate:.3f} FPS)')
                  axes[2].legend(loc='upper left')
                  axes[2].grid(True)

                  plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
                  track_plot_path = os.path.join(output_dir, f"{output_prefix}_track_{track_id}_plots.png")
                  try:
                       plt.savefig(track_plot_path, dpi=150)
                       # print(f"Track plot saved to {track_plot_path}") # Can be verbose
                  except Exception as e: print(f"Error saving track plot {track_id}: {e}")
                  # plt.show() # Uncomment to display interactively
                  plt.close(fig) # Close figure after saving


    print("\nScript finished.")


# --- Run the main function ---
if __name__ == "__main__":
    main()
