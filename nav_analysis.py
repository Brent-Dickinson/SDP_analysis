import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re
from datetime import datetime

def parse_gnss_timestamp(log_file_path):
    """Extract GNSS timestamp from log file"""
    with open(log_file_path, 'r') as file:
        for line in file:
            if "GNSS SYS TIME:" in line:
                # Extract timestamp string
                match = re.search(r'GNSS SYS TIME: ([\d-]+ [\d:]+)', line)
                if match:
                    timestamp_str = match.group(1)
                    # Convert to datetime object
                    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    # Format as string for folder name
                    return dt.strftime('%Y%m%d_%H%M')
    # Default timestamp if not found
    return datetime.now().strftime('%Y%m%d_%H%M')

def parse_nav_log(log_file_path):
    # Keep your existing parse_nav_log function, but modify to include timestamps
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    # Prepare data structures
    runs = []
    current_run = None
    waypoints = []
    
    for line in lines:
        if "ControlTask data" in line:
            try:
                # Extract timestamp
                parts = line.strip().split(',')
                timestamp = int(parts[0])
                
                # Look for position pattern using regex for more reliability
                import re
                position_match = re.search(r'ControlTask data, ([\d.]+), -([\d.]+), ([\d.]+), ([\d.]+)', line)
                
                if position_match:
                    lat = float(position_match.group(1))
                    # Longitude is negative, already captured without the "-" sign
                    lon = -float(position_match.group(2))  
                    speed = float(position_match.group(3))
                    heading = float(position_match.group(4))
                    
                    # Add to current run if active
                    if current_run is not None:
                        current_run.append({
                            'timestamp': timestamp,
                            'lat': lat,
                            'lon': lon, 
                            'speed': speed,
                            'heading': heading
                        })
            except Exception as e:
                print(f"Error parsing control data from line: {line.strip()}")
                print(f"Error details: {e}")
        
        # Detect start of a new run
        elif "NAV_CMD_START" in line:
            current_run = []
        
        # Detect end of a run
        elif "NAV_CMD_STOP" in line or "ControlTask state chg, 0" in line:
            if current_run and len(current_run) > 5:  # Only keep if has enough data points
                runs.append(current_run)
                current_run = None
        
        # Extract waypoints
        elif "NAV_CMD_ADD_WAYPOINT" in line:
            try:
                # Extract lat and lon using regex
                import re
                waypoint_match = re.search(r'NAV_CMD_ADD_WAYPOINT, ([\d.]+), -([\d.]+)', line)
                
                if waypoint_match:
                    lat = float(waypoint_match.group(1))
                    # Longitude is negative, already captured without the "-" sign
                    lon = -float(waypoint_match.group(2))
                    waypoints.append({'lat': lat, 'lon': lon})
            except Exception as e:
                print(f"Error parsing waypoint from line: {line.strip()}")
                print(f"Error details: {e}")
    
    return runs, waypoints

def plot_speed_and_heading(runs, output_dir):
    """Create plots of speed and heading over time for each run."""
    # Define colors for each run - keep consistent with GPS path plot
    colors = ['blue', 'red', 'green', 'purple']
    
    # Create figure with 2 subplots (speed and heading)
    plt.figure(figsize=(12, 10))
    
    # Speed subplot
    speed_ax = plt.subplot(2, 1, 1)
    
    for i, run in enumerate(runs):
        df = pd.DataFrame(run)
        
        # Calculate time in seconds from first timestamp
        first_timestamp = df['timestamp'].iloc[0]
        df['time_seconds'] = (df['timestamp'] - first_timestamp) / 1000.0
        
        # Plot speed over time
        speed_ax.plot(df['time_seconds'], df['speed'], 
                     color=colors[i % len(colors)], 
                     label=f'Run {i+1}')
    
    speed_ax.set_xlabel('Time (seconds)')
    speed_ax.set_ylabel('Speed (m/s)')
    speed_ax.set_title('Vehicle Speed Over Time')
    speed_ax.grid(True)
    speed_ax.legend()
    
    # Add horizontal lines for common speeds
    speed_ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    speed_ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5)
    speed_ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5)
    
    # Heading subplot
    heading_ax = plt.subplot(2, 1, 2)
    
    for i, run in enumerate(runs):
        df = pd.DataFrame(run)
        
        # Calculate time in seconds from first timestamp
        if 'time_seconds' not in df.columns:
            first_timestamp = df['timestamp'].iloc[0]
            df['time_seconds'] = (df['timestamp'] - first_timestamp) / 1000.0
        
        # Plot heading over time
        heading_ax.plot(df['time_seconds'], df['heading'], 
                       color=colors[i % len(colors)], 
                       label=f'Run {i+1}')
    
    heading_ax.set_xlabel('Time (seconds)')
    heading_ax.set_ylabel('Heading (degrees)')
    heading_ax.set_title('Vehicle Heading Over Time')
    heading_ax.grid(True)
    heading_ax.set_yticks(np.arange(0, 361, 45))  # Set ticks at cardinal/ordinal directions
    heading_ax.legend()
    
    # Add horizontal lines for cardinal directions
    heading_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    heading_ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    heading_ax.axhline(y=180, color='gray', linestyle='--', alpha=0.5)
    heading_ax.axhline(y=270, color='gray', linestyle='--', alpha=0.5)
    heading_ax.axhline(y=360, color='gray', linestyle='--', alpha=0.5)
    
    # Add cardinal direction labels
    heading_ax.text(-0.05, 0, 'N', transform=heading_ax.get_yaxis_transform(), ha='right')
    heading_ax.text(-0.05, 90, 'E', transform=heading_ax.get_yaxis_transform(), ha='right')
    heading_ax.text(-0.05, 180, 'S', transform=heading_ax.get_yaxis_transform(), ha='right')
    heading_ax.text(-0.05, 270, 'W', transform=heading_ax.get_yaxis_transform(), ha='right')
    heading_ax.text(-0.05, 360, 'N', transform=heading_ax.get_yaxis_transform(), ha='right')
    
    # Set y-axis limits to show full circle
    heading_ax.set_ylim(0, 360)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'speed_and_heading.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Speed and heading plot saved to: {output_path}")

def plot_gps_paths(runs, waypoints, output_dir):
    # Modify the add_scale_bar function to place it below the plot
    def add_scale_bar(ax, min_length=10):
        """Scale bar with individual meter ticks"""
        # Calculate conversion from degrees to meters
        lat_avg = waypoint_df['lat'].mean()
        meters_per_degree_lon = 111320 * np.cos(np.radians(lat_avg))
        
        # Calculate reasonable scale length
        x_min, x_max = ax.get_xlim()
        plot_width_degrees = x_max - x_min
        plot_width_meters = plot_width_degrees * meters_per_degree_lon
        
        # Choose scale length (~1/4 of width), but less than 20 meters
        # to keep individual meter ticks visible
        target_length = min(plot_width_meters / 4, 20)
        
        # Round to nice number
        if target_length >= 100:
            scale_length = round(target_length / 100) * 100
        elif target_length >= 10:
            scale_length = round(target_length / 10) * 10
        else:
            scale_length = round(target_length)
        
        scale_length = max(min_length, scale_length)
        
        # Convert to degrees
        scale_length_degrees = scale_length / meters_per_degree_lon
        meter_in_degrees = 1 / meters_per_degree_lon
        
        # Get y-limits
        y_min, y_max = ax.get_ylim()
        
        # Place scale bar at bottom of plot
        x_center = (x_min + x_max) / 2
        x_start = x_center - scale_length_degrees / 2
        x_end = x_center + scale_length_degrees / 2
        y_pos = y_min - (y_max - y_min) * 0.05  # Just below plot
        
        # Draw main scale bar
        ax.plot([x_start, x_end], [y_pos, y_pos], 'k-', linewidth=2)
        
        # Draw tick marks for each meter
        tick_height = (y_max - y_min) * 0.01
        for i in range(int(scale_length) + 1):
            # Position for this meter tick
            x_tick = x_start + (i * meter_in_degrees)
            
            # Draw the tick mark
            ax.plot([x_tick, x_tick], [y_pos, y_pos - tick_height], 'k-', linewidth=1)
            
            # Add label for this tick (only for 0, middle, and end)
            if i == 0 or i == int(scale_length) or i == int(scale_length // 2):
                ax.text(x_tick, y_pos - tick_height - (y_max - y_min) * 0.01, 
                        f'{i}m', 
                        horizontalalignment='center',
                        verticalalignment='top',
                        fontsize=8)
        
        # Add main label
        ax.text(x_center, y_pos - tick_height - (y_max - y_min) * 0.03, 
                f'Scale: {int(scale_length)}m', 
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=10)
        
        # Adjust plot to make room for scale bar
        plt.subplots_adjust(bottom=0.15)

    plt.figure(figsize=(10, 10))
    
    # Define colors for each run
    colors = ['blue', 'red', 'green', 'purple']
    
    # Plot each run
    for i, run in enumerate(runs):
        # Convert run to DataFrame for easier plotting
        df = pd.DataFrame(run)
        
        # Plot points as small dots for reference
        plt.scatter(df['lon'], df['lat'],
                color=colors[i % len(colors)],
                s=5,  # Small dots
                alpha=0.3,  # Semi-transparent
                zorder=1)  # Lower layer

        # Use quiver for direction arrows, but plot fewer of them for clarity
        # Select every Nth point to avoid overcrowding (adjust N as needed)
        N = max(1, len(df) // 30)  # Show ~30 arrows per run
        sample_indices = range(0, len(df), N)

        # Extract coordinates and heading components
        lons = df['lon'].iloc[sample_indices]
        lats = df['lat'].iloc[sample_indices]

        # Convert headings to unit vector components
        headings = df['heading'].iloc[sample_indices]
        u = np.sin(np.radians(headings))  # x-component
        v = np.cos(np.radians(headings))  # y-component

        # Plot the direction arrows
        q = plt.quiver(lons, lats, u, v,
                    color=colors[i % len(colors)],
                    scale=30,  # Adjust this value to control arrow size
                    width=0.004,  # Arrow width
                    alpha=0.8,
                    zorder=2)  # Upper layer

        # Add a legend entry
        plt.plot([], [], color=colors[i % len(colors)], label=f'Run {i+1}')
        
        # Mark start and end points
        plt.scatter(df['lon'].iloc[0], df['lat'].iloc[0], 
                    color=colors[i % len(colors)], marker='o', s=100, label=f'Start {i+1}')
        plt.scatter(df['lon'].iloc[-1], df['lat'].iloc[-1], 
                    color=colors[i % len(colors)], marker='x', s=100, label=f'End {i+1}')
    
    # Plot waypoints
    waypoint_df = pd.DataFrame(waypoints)
    plt.scatter(waypoint_df['lon'], waypoint_df['lat'], 
                color='black', marker='*', s=150, label='Waypoints')
    
    # Add labels for waypoints
    for i, wp in enumerate(waypoints):
        plt.annotate(f'WP{i+1}', (wp['lon'], wp['lat']), 
                    fontsize=12, ha='right', va='bottom')
    
    # Import the Ellipse patch for 2m radius visualization
    from matplotlib.patches import Ellipse
    
    # Plot 2m radius ellipses around each waypoint
    radius_added_to_legend = False
    for i, wp in enumerate(waypoints):
        # Calculate specific conversion for this waypoint's latitude
        # 1 degree latitude is approximately 111,320 meters worldwide
        meters_per_degree_lat = 111320
        # 1 degree longitude varies with latitude
        meters_per_degree_lon = 111320 * np.cos(np.radians(wp['lat']))
        
        # Convert 2 meters to degrees for this specific location
        radius_lat_degrees = 2 / meters_per_degree_lat
        radius_lon_degrees = 2 / meters_per_degree_lon
        
        # Create an ellipse to represent exactly 2m radius in both directions
        ellipse = Ellipse((wp['lon'], wp['lat']), 
                         width=radius_lon_degrees * 2,  # diameter in longitude
                         height=radius_lat_degrees * 2,  # diameter in latitude
                         color='gray', 
                         fill=True, 
                         alpha=0.2,
                         zorder=0,  # Ensure it's below other elements
                         label='2m radius' if not radius_added_to_legend else "")
        plt.gca().add_patch(ellipse)
        radius_added_to_legend = True
    
    # Get the current axis and add scale bar
    ax = plt.gca()

    # Choose a good scale bar length (about 1/4 of the plot width)
    x_range = max(waypoint_df['lon']) - min(waypoint_df['lon'])
    lat_avg = waypoint_df['lat'].mean()
    meters_per_degree_lon = 111320 * np.cos(np.radians(lat_avg))
    plot_width_meters = x_range * meters_per_degree_lon
    scale_length = round(plot_width_meters / 4, -1)  # Round to nearest 10m

    add_scale_bar(ax)

    # Set up plot appearance
    plt.grid(True)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Navigation Path with 2m Waypoint Radius')
    plt.legend()
    
    # Scale axes equally so the plot isn't distorted
    plt.axis('equal')
    
    # Save and show plot
    output_path = os.path.join(output_dir, 'navigation_path.png')
    plt.savefig(output_path, dpi=300)
    plt.show()
    
    print(f"Plot saved to: {output_path}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Parse and visualize navigation log data')
    parser.add_argument('log_file', help='Path to the navigation log file')
    args = parser.parse_args()
    
    # Get GNSS timestamp for output directory name
    timestamp = parse_gnss_timestamp(args.log_file)
    output_dir = f"nav_analysis_{timestamp}"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Parse the log file
    runs, waypoints = parse_nav_log(args.log_file)
    print(f"Found {len(runs)} autonomous runs and {len(waypoints)} waypoints")
    
    # Generate plots
    plot_gps_paths(runs, waypoints, output_dir)
    plot_speed_and_heading(runs, output_dir)

if __name__ == "__main__":
    main()