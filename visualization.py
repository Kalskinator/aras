import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import mode

# --- Read Data and Create Datetime Index ---
data_file = "src/Data/House_A/DAY_1.txt"
# Use sep='\s+' to handle whitespace-separated values
df = pd.read_csv(data_file, sep="\s+", header=None)
# Use 's' instead of 'S' for seconds
df.index = pd.date_range(start="2025-01-01", periods=len(df), freq="s")

# --- Separate Sensor Data and Activity Labels ---
sensor_df = df.iloc[:, :20]
activity_df = df.iloc[:, 20:22]


# # --- Resample Data to 1-Minute Resolution ---
# sensor_resampled = sensor_df.resample("min").mean()


# # Updated mode function to handle scalar vs array output
# def mode_func(x):
#     m = mode(x)
#     return m.mode if np.isscalar(m.mode) else m.mode[0]


# activity_resampled = activity_df.resample("min").agg(lambda x: mode_func(x))

# --- Define Sensor and Activity Labels ---
sensor_labels = [
    "Ph1 (Wardrobe)",
    "Ph2 (Convertible Couch)",
    "Ir1 (TV receiver)",
    "Fo1 (Couch)",
    "Fo2 (Couch)",
    "Di3 (Chair)",
    "Di4 (Chair)",
    "Ph3 (Fridge)",
    "Ph4 (Kitchen Drawer)",
    "Ph5 (Wardrobe)",
    "Ph6 (Bathroom Cabinet)",
    "Co1 (House Door)",
    "Co2 (Bathroom Door)",
    "Co3 (Shower Cabinet Door)",
    "So1 (Hall)",
    "So2 (Kitchen)",
    "Di1 (Tap)",
    "Di2 (Water Closet)",
    "Te1 (Kitchen)",
    "Fo3 (Bed)",
]

activity_mapping = {
    1: "Other",
    2: "Going Out",
    3: "Preparing Breakfast",
    4: "Having Breakfast",
    5: "Preparing Lunch",
    6: "Having Lunch",
    7: "Preparing Dinner",
    8: "Having Dinner",
    9: "Washing Dishes",
    10: "Having Snack",
    11: "Sleeping",
    12: "Watching TV",
    13: "Studying",
    14: "Having Shower",
    15: "Toileting",
    16: "Napping",
    17: "Using Internet",
    18: "Reading Book",
    19: "Laundry",
    20: "Shaving",
    21: "Brushing Teeth",
    22: "Talking on the Phone",
    23: "Listening to Music",
    24: "Cleaning",
    25: "Having Conversation",
    26: "Having Guest",
    27: "Changing Clothes",
}
sensor_labels_no_id = [
    "Wardrobe",
    "Convertible Couch",
    "TV receiver",
    "Couch",
    "Couch",
    "Chair",
    "Chair",
    "Fridge",
    "Kitchen Drawer",
    "Wardrobe",
    "Bathroom Cabinet",
    "House Door",
    "Bathroom Door",
    "Shower Cabinet Door",
    "Hall",
    "Kitchen",
    "Tap",
    "Water Closet",
    "Kitchen",
    "Bed",
]

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [4, 0.5, 0.5], "hspace": 0.5}, sharex=True
)

# Sensor Heatmap (ax1)
sns.heatmap(sensor_df.T, cmap="viridis", ax=ax1, cbar=False)
ax1.set_yticklabels(sensor_labels_no_id, rotation=0)

# Format x-axis for sensor heatmap
n_readings = len(sensor_df)
increment = 3600
tick_positions = np.arange(0, n_readings, increment)
tick_labels = sensor_df.index[tick_positions].strftime("%H:%M")
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45, ha="right")
ax1.set_title("Sensor Readings")
ax1.tick_params(labelbottom=True)

# Create a custom colormap with exactly one color per activity
n_activities = 27
colors = plt.cm.tab20(np.linspace(0, 1, 20))  # Get first 20 colors
extra_colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Get extra colors
all_colors = np.vstack((colors, extra_colors))  # Combine the colors
cmap = ListedColormap(all_colors[:n_activities])

# Activity Timeline Plot - Resident 1 (ax2)
im1 = ax2.imshow(
    activity_df.iloc[:, 0:1].T, aspect="auto", cmap=cmap, vmin=1, vmax=n_activities + 1
)
ax2.set_yticks([0])
ax2.set_yticklabels(["Resident 1"])
ax2.set_xticks([])

# Activity Timeline Plot - Resident 2 (ax3)
im2 = ax3.imshow(
    activity_df.iloc[:, 1:2].T, aspect="auto", cmap=cmap, vmin=1, vmax=n_activities + 1
)
ax3.set_yticks([0])
ax3.set_yticklabels(["Resident 2"])
ax3.set_xticks(tick_positions)
ax3.set_xticklabels(tick_labels, rotation=45, ha="right")

plt.subplots_adjust(bottom=0.4, hspace=0.5)

cbar = fig.colorbar(
    im1,
    ax=[ax2, ax3],
    orientation="horizontal",
    pad=0.5,
    ticks=np.arange(1.5, n_activities + 1),
    aspect=40,
)

cbar.ax.set_xticklabels(
    [activity_mapping.get(i, str(i)) for i in range(1, n_activities + 1)], rotation=45, ha="right"
)
cbar.set_label("Activity Labels")

plt.savefig("sensor_activity_visualization.png", dpi=300, bbox_inches="tight")
plt.show()


# Remove tight_layout() as we're using subplots_adjust instead
# plt.tight_layout()  # Remove this line

# # --- Calculate and Visualize Sensor Correlations ---
# plt.figure(figsize=(12, 10))

# # Calculate correlation matrix (using correlation instead of covariance for better interpretability)
# correlation_matrix = sensor_df.corr()

# # Create heatmap with sensor labels
# sns.heatmap(
#     correlation_matrix,
#     xticklabels=sensor_labels,
#     yticklabels=sensor_labels,
#     cmap="coolwarm",
#     center=0,
#     annot=True,  # Show correlation values
#     fmt=".2f",  # Round to 2 decimal places
#     square=True,
# )

# plt.title("Sensor Correlation Matrix")
# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# # Print strongest correlations
# print("\nStrongest Sensor Relationships:")
# # Get upper triangle of correlation matrix
# upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
# # Stack and sort correlations
# strong_correlations = upper_tri.unstack()
# strong_correlations = strong_correlations.sort_values(key=abs, ascending=False)
# # Print top 10 strongest correlations
# for idx, value in strong_correlations[:10].items():
#     if not np.isnan(value):  # Skip NaN values
#         print(f"{sensor_labels[idx[0]]} <-> {sensor_labels[idx[1]]}: {value:.3f}")
