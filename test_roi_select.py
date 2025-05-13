# test_roi_select.py - Minimal test for cv2.selectROIs

import cv2
import sys

# --- Configuration ---
# Use the same video path as your main script, but ensure the filename is correct!
video_path = 'C:/Users/Arun/Downloads/PhD/Manuscript/TIRF LIVE IMAGING/VIM_FAK_FA1.mp4' # IMPORTANT: Replace with your actual video file name

print(f"Attempting to load video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    sys.exit()

print("Reading first frame...")
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
    sys.exit()

print("\n--- Starting Interactive ROI Selection ---")
print("A window titled 'Minimal Test...' should appear. Draw a rectangle with your mouse.")
print("Try confirming with ENTER or 's'.")
print("Try canceling with 'q' or ESC.")
print("Try closing with the window's 'X' button.")
print("Report EXACTLY what happens and which keys (if any) cause the window to close.")
print("Waiting for selectROIs function...")

# --- Call the potentially problematic function ---
try:
    rois = cv2.selectROIs("Minimal Test (ENTER/s=OK, q/ESC=Cancel)", frame, showCrosshair=True, fromCenter=False)
except Exception as e:
    print(f"\n*** An error occurred during cv2.selectROIs: {e} ***")
    rois = None # Ensure rois is defined even if selectROIs crashes

# --- This part only runs if selectROIs finishes without crashing ---
print("\n--- Interactive ROI Selection Function Call Finished ---")
# Attempt to close any remaining OpenCV window forcefully
cv2.destroyAllWindows()
# Adding a small delay might help ensure windows close before script exits
cv2.waitKey(100)

if rois is not None and len(rois) > 0:
    # Check if it returned empty tuples like ((0,0,0,0),) which happens on cancel sometimes
    if len(rois) == 1 and rois[0] == (0, 0, 0, 0):
         print("Selection likely cancelled (returned ROI of zeros).")
         rois_list = []
    else:
        rois_list = [list(map(int, r)) for r in rois if r[2] > 0 and r[3] > 0]

    if rois_list:
        print(f"Successfully selected ROIs: {rois_list}")
    else:
        print("Selection finished, but no valid ROIs were returned (maybe cancelled?).")
else:
    print("Selection was cancelled, failed, or no ROIs were selected.")

cap.release()
print("Minimal test script finished.")