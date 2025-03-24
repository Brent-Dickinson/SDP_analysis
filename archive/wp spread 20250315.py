import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

# Function to calculate distance between two lat/lon points in meters
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in meters is 6,371,000
    m = 6371000 * c
    return m

# Load the waypoint data
df = pd.read_csv('test_route.csv')
print(f"Loaded {len(df)} waypoints")

# Basic statistics
print("\n===== Basic Statistics =====")
print(df.describe())

# Count by RTK status
print("\n===== RTK Status Counts =====")
rtk_counts = df['rtk_status'].value_counts()
print(rtk_counts)
print("\nRTK Status Legend:")
print("0 = No RTK correction")
print("1 = Float solution")
print("2 = Fixed solution (most accurate)")

# Count by fix type
print("\n===== Fix Type Counts =====")
fix_counts = df['fix_type'].value_counts()
print(fix_counts)
print("\nFix Type Legend:")
print("1 = Dead reckoning only")
print("2 = 2D fix")
print("3 = 3D fix")
print("4 = GNSS + dead reckoning")
print("5 = Time only fix")

# Calculate mean position
mean_lat = df['latitude'].mean()
mean_lon = df['longitude'].mean()
print(f"\nMean position: {mean_lat:.8f}, {mean_lon:.8f}")

# Calculate distances from mean point
df['distance_from_mean_m'] = df.apply(
    lambda row: haversine(row['latitude'], row['longitude'], mean_lat, mean_lon), 
    axis=1
)

# Distance statistics
print("\n===== Distance Statistics (meters) =====")
print(f"Mean distance from center: {df['distance_from_mean_m'].mean():.2f} m")
print(f"Max distance from center: {df['distance_from_mean_m'].max():.2f} m")
print(f"Standard deviation: {df['distance_from_mean_m'].std():.2f} m")
print(f"95% of points within: {df['distance_from_mean_m'].quantile(0.95):.2f} m")

# Analyze by RTK status
print("\n===== Analysis by RTK Status =====")
for rtk in sorted(df['rtk_status'].unique()):
    subset = df[df['rtk_status'] == rtk]
    if len(subset) > 0:
        status_name = "No RTK" if rtk == 0 else "Float" if rtk == 1 else "Fixed"
        print(f"\nRTK Status {rtk} ({status_name}) - {len(subset)} points:")
        print(f"  Mean distance from center: {subset['distance_from_mean_m'].mean():.2f} m")
        print(f"  Max distance from center: {subset['distance_from_mean_m'].max():.2f} m")
        print(f"  Standard deviation: {subset['distance_from_mean_m'].std():.2f} m")

# Calculate lat/lon offsets in meters for plotting
earth_radius = 6371000  # Earth radius in meters
meters_per_deg_lat = earth_radius * np.pi / 180
meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(mean_lat))

df['x_meters'] = (df['longitude'] - mean_lon) * meters_per_deg_lon
df['y_meters'] = (df['latitude'] - mean_lat) * meters_per_deg_lat

# Create plot with different colors by RTK status
plt.figure(figsize=(10, 8))

# Create custom color map for RTK status
rtk_colors = {0: 'red', 1: 'orange', 2: 'green'}
rtk_labels = {0: 'No RTK', 1: 'Float', 2: 'Fixed'}

# Plot scatter points
for rtk in sorted(df['rtk_status'].unique()):
    subset = df[df['rtk_status'] == rtk]
    if len(subset) > 0:
        plt.scatter(subset['x_meters'], subset['y_meters'], 
                   label=rtk_labels.get(rtk, f"RTK {rtk}"),
                   color=rtk_colors.get(rtk, 'blue'),
                   alpha=0.7)

# Add center point and labels
plt.scatter(0, 0, color='black', marker='x', s=100, label='Mean Position')
plt.xlabel('East-West Offset (meters)')
plt.ylabel('North-South Offset (meters)')
plt.title('GPS Point Spread - Colored by RTK Status')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Equal aspect ratio

# Add circles representing standard deviations
circle1 = plt.Circle((0, 0), df['distance_from_mean_m'].std(), 
                    color='gray', fill=False, linestyle='--', label='1σ')
circle2 = plt.Circle((0, 0), 2*df['distance_from_mean_m'].std(), 
                    color='gray', fill=False, linestyle=':', label='2σ')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

# Ensure axis limits show all points with some padding
max_offset = max(df['x_meters'].abs().max(), df['y_meters'].abs().max()) * 1.1
plt.xlim(-max_offset, max_offset)
plt.ylim(-max_offset, max_offset)

plt.savefig('gps_point_spread.png', dpi=300)
plt.show()

# Create a 2D histogram as alternative to the density plot
plt.figure(figsize=(10, 8))
h = plt.hist2d(df['x_meters'], df['y_meters'], bins=20, cmap='viridis')
plt.colorbar(h[3], label='Count')
plt.scatter(0, 0, color='red', marker='x', s=100)
plt.title('GPS Point Density (2D Histogram)')
plt.xlabel('East-West Offset (meters)')
plt.ylabel('North-South Offset (meters)')
plt.axis('equal')
plt.grid(True)
plt.savefig('gps_point_density_hist.png', dpi=300)
plt.show()

# Create a histogram of distances
plt.figure(figsize=(10, 6))
plt.hist(df['distance_from_mean_m'], bins=20, alpha=0.7, color='blue')
plt.axvline(df['distance_from_mean_m'].mean(), color='red', linestyle='--', label=f'Mean: {df["distance_from_mean_m"].mean():.2f}m')
plt.axvline(df['distance_from_mean_m'].quantile(0.95), color='green', linestyle='--', label=f'95th percentile: {df["distance_from_mean_m"].quantile(0.95):.2f}m')
plt.title('Histogram of Distances from Mean Position')
plt.xlabel('Distance (meters)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('distance_histogram.png', dpi=300)
plt.show()

# Additional analysis for fixed RTK points if available
fixed_rtk = df[df['rtk_status'] == 2]  # Fixed RTK has status 2
if len(fixed_rtk) > 0:
    print("\n===== Fixed RTK Analysis (most accurate) =====")
    print(f"Number of fixed RTK points: {len(fixed_rtk)}")
    print(f"Mean distance from center: {fixed_rtk['distance_from_mean_m'].mean():.2f} m")
    print(f"Standard deviation: {fixed_rtk['distance_from_mean_m'].std():.2f} m")
    print(f"95% of fixed RTK points within: {fixed_rtk['distance_from_mean_m'].quantile(0.95):.2f} m")