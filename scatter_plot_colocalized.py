import pandas as pd
import matplotlib.pyplot as plt

# Load your Excel file
file_path = 'Zcenters_vin_vim_2peaks_edited.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Calculate the absolute Z-distance between vimentin and vinculin
data['z_distance'] = abs(data['z_center_vimentin'] - data['z_center_ch1_vinculin'])

# Filter for truly co-localized ROIs (≤ 50 nm)
co_localized = data[data['z_distance'] <= 50]

# Scatter Plot for co-localized ROIs
plt.figure(figsize=(8, 6))
plt.scatter(co_localized['z_center_vimentin'], co_localized['z_center_ch1_vinculin'], 
            c='orange', label='Truly Co-localized ROIs (≤ 50 nm)')
plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='Identity Line (Z Vim = Z Vinc)')
plt.title('Scatter Plot of Truly Co-localized ROIs')
plt.xlabel('Z Centers - Vimentin (nm)')
plt.ylabel('Z Centers - Vinculin (nm)')
plt.xticks(range(0, 300, 25))
plt.yticks(range(0, 300, 25))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
