import re
import csv
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Extract GPS data from log file')
    parser.add_argument('--log', type=str, required=True, help='Path to the log file')
    parser.add_argument('--output', type=str, default='extracted_data', help='Directory to save output files')
    return parser.parse_args()

def extract_navigation_data(log_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Open and read log file
    with open(log_path, 'r') as f:
        log_lines = f.readlines()
    
    # Skip header lines
    data_lines = []
    for line in log_lines:
        if re.match(r'^\d+,', line):
            data_lines.append(line)
    
    # Extract waypoints
    waypoints = []
    waypoint_changes = []
    
    # Extract navigation runs
    nav_runs = []
    current_run = {
        'start_time': None,
        'end_time': None,
        'waypoints': [],
        'positions': [],
        'controls': []
    }
    
    run_active = False
    
    # Process each line
    for line in data_lines:
        parts = line.strip().split(',', 3)
        if len(parts) < 4:
            continue
            
        timestamp, level, task, message = parts
        timestamp_ms = int(timestamp)
        
        # Convert timestamp to datetime for CSV
        dt = datetime.fromtimestamp(timestamp_ms / 1000)  # Convert ms to seconds
        timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Format with milliseconds
        
        # Detect navigation start
        if "Navigation started with followingWaypoints=1" in message:
            if run_active:
                # End previous run if a new one starts without explicit stop
                if current_run['start_time'] is not None:
                    nav_runs.append(current_run)
            
            # Start new run
            current_run = {
                'start_time': timestamp_ms,
                'end_time': None,
                'waypoints': [],
                'positions': [],
                'controls': []
            }
            run_active = True
        
        # Detect navigation stop
        if "Processing STOP command" in message and "autonomousMode=1" in message:
            if run_active:
                current_run['end_time'] = timestamp_ms
                nav_runs.append(current_run)
                run_active = False
        
        # Extract waypoint additions
        if "Waypoint added at index" in message:
            idx_match = re.search(r'Waypoint added at index (\d+), new count=(\d+)', message)
            if idx_match:
                index = int(idx_match.group(1))
                count = int(idx_match.group(2))
                
                # Find the corresponding ADD_WAYPOINT command
                for i in range(len(data_lines)-1, -1, -1):
                    prev_line = data_lines[i]
                    if "Processing ADD_WAYPOINT command" in prev_line:
                        coord_match = re.search(r'lat=([\d\.\-]+), lon=([\d\.\-]+)', prev_line)
                        if coord_match:
                            lat = float(coord_match.group(1))
                            lon = float(coord_match.group(2))
                            
                            waypoint_changes.append({
                                'timestamp': timestamp_ms,
                                'timestamp_str': timestamp_str,
                                'action': 'add',
                                'index': index,
                                'count': count,
                                'lat': lat,
                                'lon': lon
                            })
                            
                            # Maintain waypoints list
                            if index >= len(waypoints):
                                waypoints.append((lat, lon))
                            else:
                                waypoints[index] = (lat, lon)
                                
                            # Add to current run if active
                            if run_active:
                                current_run['waypoints'] = waypoints.copy()
                            
                            break
        
        # Extract waypoint clears
        if "CLEAR_WAYPOINTS" in message:
            waypoint_changes.append({
                'timestamp': timestamp_ms,
                'timestamp_str': timestamp_str,
                'action': 'clear',
                'index': -1,
                'count': 0,
                'lat': None,
                'lon': None
            })
            waypoints = []
            if run_active:
                current_run['waypoints'] = []
        
        # Extract control values
        control_match = re.search(r'Control values: speed=([\d\.]+), targetSpeed=([\d\.]+), heading=([\d\.]+)', message)
        if control_match:
            speed = float(control_match.group(1))
            target_speed = float(control_match.group(2))
            heading = float(control_match.group(3))
            
            if run_active:
                current_run['controls'].append({
                    'timestamp': timestamp_ms,
                    'timestamp_str': timestamp_str,
                    'speed': speed,
                    'target_speed': target_speed,
                    'heading': heading
                })
        
        # Extract GPS data
        gps_match = re.search(r'GPS data: fix=\d+, lat=([\d\.\-]+), lng=([\d\.\-]+)', message)
        if gps_match:
            lat = float(gps_match.group(1))
            lon = float(gps_match.group(2))
            
            if run_active:
                current_run['positions'].append({
                    'timestamp': timestamp_ms,
                    'timestamp_str': timestamp_str,
                    'lat': lat,
                    'lon': lon
                })
        
        # Extract waypoint transitions
        wp_match = re.search(r'Moving to next waypoint: (\d+) of (\d+)', message)
        if wp_match and run_active:
            wp_num = int(wp_match.group(1))
            wp_total = int(wp_match.group(2))
            
            # Add waypoint transition event to current run
            if 'wp_transitions' not in current_run:
                current_run['wp_transitions'] = []
                
            current_run['wp_transitions'].append({
                'timestamp': timestamp_ms,
                'timestamp_str': timestamp_str,
                'wp_num': wp_num,
                'wp_total': wp_total
            })
    
    # Add the last run if it's active
    if run_active and current_run['start_time'] is not None:
        current_run['end_time'] = timestamp_ms
        nav_runs.append(current_run)
    
    # Write waypoints to CSV
    with open(os.path.join(output_dir, 'waypoints.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Latitude', 'Longitude'])
        for i, (lat, lon) in enumerate(waypoints):
            writer.writerow([i, lat, lon])
    
    # Write waypoint changes to CSV
    with open(os.path.join(output_dir, 'waypoint_changes.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Action', 'Index', 'Count', 'Latitude', 'Longitude'])
        for change in waypoint_changes:
            writer.writerow([
                change['timestamp_str'],
                change['action'],
                change['index'],
                change['count'],
                change['lat'],
                change['lon']
            ])
    
    # Write each navigation run to separate files
    for i, run in enumerate(nav_runs):
        run_dir = os.path.join(output_dir, f'run_{i+1}')
        os.makedirs(run_dir, exist_ok=True)
        
        # Write run info
        with open(os.path.join(run_dir, 'info.txt'), 'w') as f:
            start_time = datetime.fromtimestamp(run['start_time'] / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            end_time = "N/A"
            if run['end_time']:
                end_time = datetime.fromtimestamp(run['end_time'] / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            f.write(f"Run {i+1}\n")
            f.write(f"Start time: {start_time}\n")
            f.write(f"End time: {end_time}\n")
            f.write(f"Duration: {(run['end_time'] - run['start_time'])/1000:.2f} seconds\n")
            f.write(f"Waypoints: {len(run['waypoints'])}\n")
            f.write(f"Position records: {len(run['positions'])}\n")
            f.write(f"Control records: {len(run['controls'])}\n")
        
        # Write run waypoints
        with open(os.path.join(run_dir, 'waypoints.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Index', 'Latitude', 'Longitude'])
            for i, (lat, lon) in enumerate(run['waypoints']):
                writer.writerow([i, lat, lon])
        
        # Write position data
        if run['positions']:
            with open(os.path.join(run_dir, 'positions.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Latitude', 'Longitude'])
                for pos in run['positions']:
                    writer.writerow([pos['timestamp_str'], pos['lat'], pos['lon']])
        
        # Write control data
        if run['controls']:
            with open(os.path.join(run_dir, 'controls.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Speed', 'Target Speed', 'Heading'])
                for ctrl in run['controls']:
                    writer.writerow([
                        ctrl['timestamp_str'],
                        ctrl['speed'],
                        ctrl['target_speed'],
                        ctrl['heading']
                    ])
        
        # Write waypoint transitions
        if 'wp_transitions' in run and run['wp_transitions']:
            with open(os.path.join(run_dir, 'wp_transitions.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Waypoint Number', 'Total Waypoints'])
                for trans in run['wp_transitions']:
                    writer.writerow([
                        trans['timestamp_str'],
                        trans['wp_num'],
                        trans['wp_total']
                    ])
    
    return len(waypoints), len(nav_runs)

def main():
    args = parse_args()
    num_waypoints, num_runs = extract_navigation_data(args.log, args.output)
    print(f"Extracted {num_waypoints} waypoints and {num_runs} navigation runs to {args.output}/")

if __name__ == "__main__":
    main()