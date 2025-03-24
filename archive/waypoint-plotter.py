import matplotlib.pyplot as plt
import re
import os
import numpy as np

#python waypoint-plotter.py --log log.txt --output waypoint_plots

def extract_waypoints(log_file):
    """Extract waypoint sets from the log file"""
    
    # Dictionary to store waypoint sets (key = set number, value = list of waypoints)
    waypoint_sets = {}
    
    # Current set tracking
    current_set = []
    current_set_id = 1
    
    with open(log_file, 'r') as f:
        for line in f:
            # Skip non-data lines
            if not re.match(r'^\d+,', line):
                continue
                
            parts = line.strip().split(',', 3)
            if len(parts) < 4:
                continue
                
            timestamp, level, task, message = parts
            
            # Check for waypoint clearing
            if "CLEAR_WAYPOINTS" in message or "Processing navigation command type: 6" in message:
                # Save previous set of waypoints if we have any
                if current_set:
                    waypoint_sets[current_set_id] = current_set.copy()
                    print(f"Found waypoint set {current_set_id} with {len(current_set)} waypoints")
                
                # Start a new set
                current_set = []
                current_set_id += 1
            
            # Check for waypoint additions
            elif "Waypoint added at index" in message:
                # Look backwards in the log to find the corresponding command with the coordinates
                cmd_match = re.search(r'Processing ADD_WAYPOINT command: lat=([\d\.\-]+), lon=([\d\.\-]+)', message)
                if cmd_match:
                    lat = float(cmd_match.group(1))
                    lon = float(cmd_match.group(2))
                    current_set.append((lat, lon))
                    print(f"Added waypoint {len(current_set)-1} to set {current_set_id}: ({lat}, {lon})")
                else:
                    # Direct search in the current line
                    direct_match = re.search(r'lat=([\d\.\-]+), lon=([\d\.\-]+)', message)
                    if direct_match:
                        lat = float(direct_match.group(1))
                        lon = float(direct_match.group(2))
                        current_set.append((lat, lon))
                        print(f"Added waypoint {len(current_set)-1} to set {current_set_id}: ({lat}, {lon})")
            
            # More directed search pattern
            elif "Processing ADD_WAYPOINT command" in message:
                wp_match = re.search(r'lat=([\d\.\-]+), lon=([\d\.\-]+)', message)
                if wp_match:
                    lat = float(wp_match.group(1))
                    lon = float(wp_match.group(2))
                    current_set.append((lat, lon))
                    print(f"Added waypoint {len(current_set)-1} to set {current_set_id}: ({lat}, {lon})")
    
    # Add the final set if it has any waypoints
    if current_set:
        waypoint_sets[current_set_id] = current_set.copy()
        print(f"Found waypoint set {current_set_id} with {len(current_set)} waypoints")
    
    return waypoint_sets

def plot_waypoint_sets(waypoint_sets, output_dir):
    """Create visualizations of waypoint sets"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Colors for different waypoint sets
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink', 'olive', 'gray']
    
    # First, plot all waypoint sets together
    plt.figure(figsize=(12, 10))
    
    # Keep track of all coordinates for proper scaling
    all_lats = []
    all_lons = []
    
    for set_id, waypoints in waypoint_sets.items():
        if not waypoints:
            continue
        
        # Extract coordinates
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        
        all_lats.extend(lats)
        all_lons.extend(lons)
        
        # Plot waypoints
        color = colors[(set_id-1) % len(colors)]
        plt.plot(lons, lats, marker='x', markersize=10, linestyle='-', color=color, 
                 label=f'Set {set_id} ({len(waypoints)} points)')
        
        # Label each waypoint
        for i, (lat, lon) in enumerate(waypoints):
            plt.text(lon, lat, f' {i}', fontsize=9, color=color)
        
        # Draw waypoint detection radius circles (2 meters)
        for lat, lon in waypoints:
            radius = 2.0 / 111111.0  # Approx. conversion from 2m to degrees
            circle = plt.Circle((lon, lat), radius, fill=False, linestyle='--', 
                               edgecolor=color, alpha=0.5)
            plt.gca().add_patch(circle)
    
    if all_lats and all_lons:
        # Set axis limits with some padding
        buffer = 0.0001  # About 10 meters
        plt.xlim(min(all_lons) - buffer, max(all_lons) + buffer)
        plt.ylim(min(all_lats) - buffer, max(all_lats) + buffer)
        
        # Set aspect ratio to be correct for the latitude
        mid_lat = np.mean(all_lats)
        plt.gca().set_aspect(1.0 / np.cos(np.radians(mid_lat)))
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('RC Car Waypoint Sets')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_waypoint_sets.png'), dpi=300)
    plt.close()
    
    # Now plot each set individually for more detail
    for set_id, waypoints in waypoint_sets.items():
        if not waypoints:
            continue
        
        plt.figure(figsize=(10, 8))
        
        # Extract coordinates
        lats = [wp[0] for wp in waypoints]
        lons = [wp[1] for wp in waypoints]
        
        # Plot waypoints
        color = colors[(set_id-1) % len(colors)]
        plt.plot(lons, lats, marker='x', markersize=10, linestyle='-', color=color, 
                 label=f'Set {set_id} ({len(waypoints)} points)')
        
        # Label each waypoint
        for i, (lat, lon) in enumerate(waypoints):
            plt.text(lon, lat, f' {i}', fontsize=9, color=color)
        
        # Draw waypoint detection radius circles (2 meters)
        for lat, lon in waypoints:
            radius = 2.0 / 111111.0  # Approx. conversion from 2m to degrees
            circle = plt.Circle((lon, lat), radius, fill=False, linestyle='--', 
                               edgecolor=color, alpha=0.5)
            plt.gca().add_patch(circle)
        
        # Set axis limits with some padding
        buffer = 0.0001  # About 10 meters
        plt.xlim(min(lons) - buffer, max(lons) + buffer)
        plt.ylim(min(lats) - buffer, max(lats) + buffer)
        
        # Set aspect ratio to be correct for the latitude
        mid_lat = np.mean(lats)
        plt.gca().set_aspect(1.0 / np.cos(np.radians(mid_lat)))
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'RC Car Waypoint Set {set_id}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'waypoint_set_{set_id}.png'), dpi=300)
        plt.close()
    
    print(f"Waypoint plots saved to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RC Car Waypoint Set Plotter')
    parser.add_argument('--log', required=True, help='Path to log file')
    parser.add_argument('--output', default='waypoint_plots', help='Output directory')
    args = parser.parse_args()
    
    print(f"Extracting waypoints from log file: {args.log}")
    waypoint_sets = extract_waypoints(args.log)
    print(f"Found {len(waypoint_sets)} waypoint sets")
    
    plot_waypoint_sets(waypoint_sets, args.output)
    
    # Save the sets to a text file for reference
    with open(os.path.join(args.output, 'waypoint_sets.txt'), 'w') as f:
        f.write(f"RC Car Waypoint Sets\n")
        f.write(f"===================\n\n")
        f.write(f"Total waypoint sets: {len(waypoint_sets)}\n\n")
        
        for set_id, waypoints in waypoint_sets.items():
            f.write(f"Set {set_id}:\n")
            f.write(f"-------\n")
            f.write(f"Number of waypoints: {len(waypoints)}\n")
            
            for i, (lat, lon) in enumerate(waypoints):
                f.write(f"  Waypoint {i}: ({lat}, {lon})\n")
            
            f.write("\n")

if __name__ == "__main__":
    main()