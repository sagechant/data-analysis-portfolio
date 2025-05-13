import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'Zcenters_vin_vim_2peaks_edited.xlsx'
data = pd.read_excel(file_path)

# Prepare data (remove NaN rows for each peak)
vimentin = data['z_center_vimentin']

# Filter for vinculin_peak1
vimentin_peak1 = data['z_center_vimentin'][~data['z_center_vinculin_peak1'].isna()]
vinculin_peak1 = data['z_center_vinculin_peak1'].dropna()

# Filter for vinculin_peak2
vimentin_peak2 = data['z_center_vimentin'][~data['z_center_vinculin_peak2'].isna()]
vinculin_peak2 = data['z_center_vinculin_peak2'].dropna()

# Plot
fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

# Scatter Plot (Left)
ax[0].scatter(data['z_center_vimentin'], data['z_center_ch1_vinculin'], c='orange', label='Vimentin and Vinculin (Co-localized)')
ax[0].scatter(vimentin_peak1, vinculin_peak1, c='blue', label='Vinculin Peak 1 (Distinct)')
ax[0].scatter(vimentin_peak2, vinculin_peak2, c='green', label='Vinculin Peak 2 (Distinct)')

# Set x and y axis limits and increments
ax[0].set_xlim(0, max(data['z_center_vimentin'].max(), vinculin_peak2.max()) + 25)
ax[0].set_ylim(0, max(data['z_center_ch1_vinculin'].max(), vinculin_peak2.max()) + 25)
ax[0].set_xticks(range(0, int(max(data['z_center_vimentin'].max(), vinculin_peak2.max()) + 25), 25))
ax[0].set_yticks(range(0, int(max(data['z_center_ch1_vinculin'].max(), vinculin_peak2.max()) + 25), 25))

ax[0].set_xlabel("Z Centers - Vimentin (nm)", fontsize=12)
ax[0].set_ylabel("Z Centers - Vinculin (nm)", fontsize=12)
ax[0].set_title("Scatter Plot of Z Centers for Vimentin and Vinculin")
ax[0].legend()

# Histogram (Right)
ax[1].hist(data['z_center_vimentin'].dropna(), bins=15, alpha=0.5, label='Vimentin', color='orange')
ax[1].hist(vinculin_peak1, bins=15, alpha=0.5, label='Vinculin Peak 1', color='blue')
ax[1].hist(vinculin_peak2, bins=15, alpha=0.5, label='Vinculin Peak 2', color='green')

# Set x-axis limits and increments for histogram
ax[1].set_xlim(0, max(data['z_center_vimentin'].max(), vinculin_peak2.max()) + 25)
ax[1].set_xticks(range(0, int(max(data['z_center_vimentin'].max(), vinculin_peak2.max()) + 25), 25))

ax[1].set_xlabel("Z Centers (nm)", fontsize=12)
ax[1].set_ylabel("Frequency", fontsize=12)
ax[1].set_title("Histogram of Z Centers")
ax[1].legend()

# Show the plot
plt.tight_layout()
plt.show()


