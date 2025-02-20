import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import mode

# --- Read Data and Create Datetime Index ---
data_file = 'House A\DAY_3.txt'
# Use sep='\s+' to handle whitespace-separated values
df = pd.read_csv(data_file, sep='\s+', header=None)
# Use 's' instead of 'S' for seconds
df.index = pd.date_range(start='2025-01-01', periods=len(df), freq='s')

# --- Separate Sensor Data and Activity Labels ---
sensor_df = df.iloc[:, :20]
activity_df = df.iloc[:, 20:22]

# # --- Resample Data to 1-Minute Resolution ---
# sensor_resampled = sensor_df.resample('min').mean()

# # Updated mode function to handle scalar vs array output
# def mode_func(x):
#     m = mode(x)
#     return m.mode if np.isscalar(m.mode) else m.mode[0]

# activity_resampled = activity_df.resample('min').agg(lambda x: mode_func(x))

# --- Define Sensor and Activity Labels ---
sensor_labels = [
    'Ph1 (Wardrobe)', 'Ph2 (Convertible Couch)', 'Ir1 (TV receiver)',
    'Fo1 (Couch)', 'Fo2 (Couch)', 'Di3 (Chair)', 'Di4 (Chair)',
    'Ph3 (Fridge)', 'Ph4 (Kitchen Drawer)', 'Ph5 (Wardrobe)',
    'Ph6 (Bathroom Cabinet)', 'Co1 (House Door)', 'Co2 (Bathroom Door)',
    'Co3 (Shower Cabinet Door)', 'So1 (Hall)', 'So2 (Kitchen)',
    'Di1 (Tap)', 'Di2 (Water Closet)', 'Te1 (Kitchen)', 'Fo3 (Bed)'
]

activity_mapping = {
    1: 'Other', 2: 'Going Out', 3: 'Preparing Breakfast',
    4: 'Having Breakfast', 5: 'Preparing Lunch', 6: 'Having Lunch',
    7: 'Preparing Dinner', 8: 'Having Dinner', 9: 'Washing Dishes',
    10: 'Having Snack', 11: 'Sleeping', 12: 'Watching TV',
    13: 'Studying', 14: 'Having Shower', 15: 'Toileting',
    16: 'Napping', 17: 'Using Internet', 18: 'Reading Book',
    19: 'Laundry', 20: 'Shaving', 21: 'Brushing Teeth',
    22: 'Talking on the Phone', 23: 'Listening to Music', 24: 'Cleaning',
    25: 'Having Conversation', 26: 'Having Guest', 27: 'Changing Clothes'
}

# --- Create Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

# Sensor Heatmap: Transpose so each row corresponds to a sensor
sns.heatmap(sensor_df.T, cmap='viridis', ax=ax1, cbar_kws={'label': 'Sensor Value'})
ax1.set_yticklabels(sensor_labels, rotation=0)
ax1.set_title('Sensor Readings (Aggregated by Minute)')

# # Activity Timeline Plot: Prepare a 2xN array for Resident 1 and 2
# activity_array = activity_resampled.to_numpy().T

# # Create a categorical colormap with enough colors for 27 activities
# cmap = plt.get_cmap('tab20', 28)  # 28 colors (one extra)
# im = ax2.imshow(activity_array, aspect='auto', cmap=cmap, vmin=0.5, vmax=27.5)
# ax2.set_yticks([0, 1])
# ax2.set_yticklabels(['Resident 1', 'Resident 2'])
# ax2.set_title('Activity Labels (Aggregated by Minute)')

# # Set x-axis ticks as time labels every 2 hours
# time_ticks = pd.date_range(start='2025-01-01', end='2025-01-02', freq='2H')
# time_tick_positions = [(t - df.index[0]).total_seconds() / 60 for t in time_ticks]  # convert seconds to minutes
# ax2.set_xticks(time_tick_positions)
# ax2.set_xticklabels([t.strftime('%H:%M') for t in time_ticks])

# # Add a horizontal colorbar with custom ticks/labels for activities
# cbar = fig.colorbar(im, ax=ax2, orientation='horizontal', pad=0.2, ticks=range(1, 28))
# cbar.ax.set_xticklabels([activity_mapping.get(i, str(i)) for i in range(1, 28)])
# cbar.set_label('Activity Labels')

plt.tight_layout()
plt.show()
