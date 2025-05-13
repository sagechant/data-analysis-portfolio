import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'Zcenters_vin_vim_2peaks_edited.xlsx'  # Updated file path
data = pd.read_excel(file_path)

# Filter co-localized and distinct cases based on Gaussian_peaks column
co_localized = data[data['Gaussian_peaks'] == 1]
distinct = data[data['Gaussian_peaks'] == 2]

# Scatter plot for co-localized points
plt.scatter(co_localized['z_center_vimentin'], co_localized['z_center_ch1_vinculin'],
            color='orange', label='Vimentin and Vinculin (Co-localized)', alpha=0.7)

# Scatter plot for distinct points
plt.scatter(distinct['z_center_vimentin'], distinct['z_center_vinculin_peak1'],
            color='blue', label='Vinculin Peak 1 (Distinct)', alpha=0.7)
plt.scatter(distinct['z_center_vimentin'], distinct['z_center_vinculin_peak2'],
            color='green', label='Vinculin Peak 2 (Distinct)', alpha=0.7)

# Plot formatting
plt.xlabel("Z Centers - Vimentin (nm)", fontsize=12, fontweight="bold")
plt.ylabel("Z Centers - Vinculin (nm)", fontsize=12, fontweight="bold")
plt.title("Scatter Plot of Z Centers for Vimentin and Vinculin", fontsize=14, fontweight="bold")
plt.legend()
plt.grid(True)
plt.show()
