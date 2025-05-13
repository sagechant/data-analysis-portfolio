import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CONFIGURATION ---
video_path = 'C:/Users/Arun/Downloads/PhD/Manuscript/TIRF LIVE IMAGING/VIM_FAK_FA1.mp4'
use_interactive_roi = True
rois = [(290, 300, 40, 30), (350, 310, 40, 30)]
adaptive_threshold_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
adaptive_threshold_block_size = 11  # Must be odd
adaptive_threshold_constant = 2
smooth_window = 5

# --- INITIALIZATION ---
if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    exit()

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Frame Rate: {frame_rate:.2f} FPS, Resolution: {frame_width}x{frame_height}")

if use_interactive_roi:
    temp_rois = []
    drawing = False
    start_point = None

    def draw_roi(event, x, y, flags, param):
        global drawing, start_point, temp_rois, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            frame_copy = frame.copy()
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Define ROIs", frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if start_point:
                x1, y1 = start_point
                temp_rois.append((x1, x, y1, y))
                cv2.rectangle(frame, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Define ROIs", frame)

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_copy = frame.copy()
    cv2.namedWindow("Define ROIs")
    cv2.setMouseCallback("Define ROIs", draw_roi)
    print("Draw rectangles with the mouse. Press 's' to save, 'q' to quit.")
    while True:
        cv2.imshow("Define ROIs", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            rois = [(min(x, w), min(y, h), abs(w - x), abs(h - y)) for x, w, y, h in temp_rois]
            cv2.destroyAllWindows()
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
    cap.release()
    cap = cv2.VideoCapture(video_path)

num_rois = len(rois)
vimentin_intensity = [[] for _ in range(num_rois)]
fak_intensity = [[] for _ in range(num_rois)]
fak_area = [[] for _ in range(num_rois)]

# --- PROCESSING VIDEO FRAMES ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i, (x, y, w, h) in enumerate(rois):
        # Validate ROI bounds
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_width - x), min(h, frame_height - y)
        if w <= 0 or h <= 0:
            print(f"Warning: ROI {i+1} is invalid or out of bounds.")
            continue

        roi_frame = frame_rgb[y:y+h, x:x+w]
        red = roi_frame[:, :, 0]  # Vimentin
        green = roi_frame[:, :, 1]  # FAK
        vimentin_intensity[i].append(np.mean(red) if red.size > 0 else 0)
        fak_intensity[i].append(np.mean(green) if green.size > 0 else 0)

        thresh = cv2.adaptiveThreshold(green, 255, adaptive_threshold_method,
                                       cv2.THRESH_BINARY_INV, adaptive_threshold_block_size,
                                       adaptive_threshold_constant)
        fak_area[i].append(np.sum(thresh == 255))

cap.release()

# --- NORMALIZATION ---
normalized_vimentin_intensity = []
normalized_fak_intensity = []
for i in range(num_rois):
    max_vim = max(np.max(vimentin_intensity[i]), 1)
    max_fak = max(np.max(fak_intensity[i]), 1)
    normalized_vimentin_intensity.append([val / max_vim for val in vimentin_intensity[i]])
    normalized_fak_intensity.append([val / max_fak for val in fak_intensity[i]])

# --- SMOOTHING ---
def smooth(data, window):
    return np.convolve(data, np.ones(window)/window, mode='same')  # Changed to 'same'

# --- PLOTTING ---
for i in range(num_rois):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(smooth(vimentin_intensity[i], smooth_window), label='Vimentin', color='red')
    plt.plot(smooth(fak_intensity[i], smooth_window), label='FAK', color='green')
    plt.title(f'ROI {i+1}: Intensity & Area Over Time')
    plt.ylabel('Smoothed Intensity')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(smooth(normalized_vimentin_intensity[i], smooth_window), label='Norm. Vimentin', color='red')
    plt.plot(smooth(normalized_fak_intensity[i], smooth_window), label='Norm. FAK', color='green')
    plt.ylabel('Normalized Intensity (0-1)')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(smooth(fak_area[i], smooth_window), label='FAK Area', color='blue')
    plt.xlabel('Frame')
    plt.ylabel('Smoothed Area (Pixels)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# --- EXPORT TO CSV ---
frames = len(vimentin_intensity[0]) if vimentin_intensity else 0
data = {'Frame': list(range(frames))}
for i in range(num_rois):
    data[f'ROI{i+1}_Vimentin_Intensity'] = vimentin_intensity[i]
    data[f'ROI{i+1}_FAK_Intensity'] = fak_intensity[i]
    data[f'ROI{i+1}_FAK_Area_Pixels'] = fak_area[i]
    data[f'ROI{i+1}_Norm_Vimentin'] = normalized_vimentin_intensity[i]
    data[f'ROI{i+1}_Norm_FAK'] = normalized_fak_intensity[i]

df = pd.DataFrame(data)
df.to_csv('df.to_csv('C:/Users/Arun/Downloads/PhD/Manuscript/TIRF LIVE IMAGING/Python Analysis Results/FAK_VIM_ROI_Tracking.csv', index=False)', index=False)
print("✅ Data exported to FAK_VIM_ROI_Tracking.csv")