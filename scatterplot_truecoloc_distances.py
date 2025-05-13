import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'Zcenters_vin_vim_2peaks_edited.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Calculate the absolute Z-center differences
data['abs_diff'] = abs(data['z_center_vimentin'] - data['z_center_ch1_vinculin'])

# Filter ROIs with colocalization threshold (≤ 50 nm)
colocalized_data = data[data['abs_diff'] <= 50]

# Get Z-center data
vimentin = colocalized_data['z_center_vimentin']
vinculin = colocalized_data['z_center_ch1_vinculin']
roi_index = colocalized_data['ROI_ind']  # Use original ROI indexes

# Create the scatter plot
plt.figure(figsize=(12, 6))

# Plot Vimentin and Vinculin Z centers
plt.scatter(roi_index, vimentin, color='orange', label='Vimentin Z Centers')
plt.scatter(roi_index, vinculin, color='blue', label='Vinculin Z Centers')

# Add dashed lines to represent distances
for i in range(len(roi_index)):
    plt.plot([roi_index.iloc[i], roi_index.iloc[i]], [vimentin.iloc[i], vinculin.iloc[i]], 
             color='gray', linestyle='--', linewidth=0.8)

# Adjust Y-axis range and ticks
plt.ylim(0, 300)
plt.yticks(range(0, 301, 25))  # 0 to 300 nm with increments of 25 nm

# Adjust X-axis ticks to include all ROI indexes
plt.xticks(roi_index, rotation=45, fontsize=10)

# Customize the plot
plt.title('Vimentin and Vinculin Z Centers with True Distances (≤ 50 nm)', fontsize=14)
plt.xlabel('ROI Index', fontsize=12)
plt.ylabel('Z Centers (nm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()


