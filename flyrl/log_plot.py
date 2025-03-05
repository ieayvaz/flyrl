import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Load CSV file
df = pd.read_csv("flight_data2.csv")

# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude (m)")
ax.set_title("3D Flight Path Animation")

# Plot initialization
line, = ax.plot([], [], [], 'b', label='Flight Path')
scat = ax.scatter([], [], [], c=[], cmap='viridis', marker='o')

# Set axis limits
ax.set_xlim(df['Lon'].min(), df['Lon'].max())
ax.set_ylim(df['Lat'].min(), df['Lat'].max())
ax.set_zlim(0, 150)

# Animation update function
def update(num):
    line.set_data(df['Lon'][:num], df['Lat'][:num])
    line.set_3d_properties(df['Alt'][:num])
    scat._offsets3d = (df['Lon'][:num], df['Lat'][:num], df['Alt'][:num])
    scat.set_array(df['Time'][:num])
    return line, scat

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50, blit=False)

plt.legend()
plt.show()
