import matplotlib.pyplot as plt
import re
import os
import numpy as np

def parse_runs(log_file):
    # Find all navigation runs in the log
    runs = []
    current_run = None
    run_number = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            # Skip non-data lines
            if not re.match(r'^\d+,', line):
                continue
                
            parts = line.strip().split(',', 3)
            if len(parts) < 4:
                continue
                
            timestamp, level, task, message = parts
            timestamp = int(timestamp)
            
            # Detect run starts
            if "Navigation started with followingWaypoints=1" in message:
                run_number += 1
                current_run = {
                    'id': run_number,
                    'start_time': timestamp,
                    'controls': [],
                    'waypoints': []
                }
                print(f"Found run start at timestamp {timestamp}")
            
            # Detect run stops
            elif "Processing STOP command, was autonomousMode=1" in message:
                if current_run:
                    current_run['end_time'] = timestamp
                    runs.append(current_run)
                    print(f"Found run end at timestamp {timestamp}, duration: {(timestamp - current_run['start_time'])/1000:.2f}s")
                    current_run = None
            
            # Collect control data for the current run
            elif current_run and "Control values:" in message:
                control_match = re.search(r'speed=([\d\.]+), targetSpeed=([\d\.]+), heading=([\d\.]+)', message)
                if control_match:
                    speed = float(control_match.group(1))
                    target_speed = float(control_match.group(2))
                    heading = float(control_match.group(3))
                    current_run['controls'].append({
                        'timestamp': timestamp,
                        'speed': speed,
                        'target_speed': target_speed,
                        'heading': heading
                    })
    
    # Add last run if not properly closed
    if current_run:
        runs.append(current_run)
    
    return runs

def plot_telemetry(runs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot telemetry for each run
    for run in runs:
        if not run['controls']:
            continue
            
        run_id = run['id']
        print(f"Plotting telemetry for run {run_id}")
        
        # Extract data
        timestamps = [c['timestamp'] for c in run['controls']]
        rel_time = [(t - timestamps[0])/1000 for t in timestamps]  # Time in seconds
        speeds = [c['speed'] for c in run['controls']]
        target_speeds = [c['target_speed'] for c in run['controls']]
        headings = [c['heading'] for c in run['controls']]
        
        # Calculate heading changes
        heading_changes = [0]
        for i in range(1, len(headings)):
            change = headings[i] - headings[i-1]
            # Handle angle wrapping
            if change > 180:
                change -= 360
            elif change < -180:
                change += 360
            heading_changes.append(abs(change))
        
        # Create figure with 3 subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Heading
        axs[0].plot(rel_time, headings, 'b-', linewidth=2)
        axs[0].set_ylabel('Heading (°)')
        axs[0].set_title(f'Run {run_id} Telemetry')
        axs[0].grid(True, linestyle='--', alpha=0.7)
        axs[0].axhline(y=122.03, color='r', linestyle='--', label='122.03° (Stuck Value)', alpha=0.5)
        axs[0].legend()
        
        # Plot 2: Speed
        axs[1].plot(rel_time, speeds, 'g-', linewidth=2, label='Actual Speed')
        axs[1].plot(rel_time, target_speeds, 'r--', linewidth=1, label='Target Speed')
        axs[1].set_ylabel('Speed (m/s)')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend()
        
        # Plot 3: Heading Change
        axs[2].plot(rel_time, heading_changes, 'm-', linewidth=2)
        axs[2].set_xlabel('Time (seconds)')
        axs[2].set_ylabel('|Heading Change| (°)')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].axhline(y=0.1, color='r', linestyle='--', label='Near-Zero Change Threshold', alpha=0.5)
        axs[2].legend()
        
        # Highlight potential stuck heading regions
        for i, change in enumerate(heading_changes):
            if change < 0.1:  # Very small change threshold
                axs[2].axvspan(rel_time[i]-0.1, rel_time[i]+0.1, color='r', alpha=0.1)
        
        # Check for specific stuck value (122.03°)
        stuck_122 = []
        for i, heading in enumerate(headings):
            if abs(heading - 122.03) < 0.01:
                stuck_122.append(rel_time[i])
                axs[0].plot(rel_time[i], heading, 'ro', markersize=5)  # Mark stuck points
        
        if stuck_122:
            print(f"Run {run_id} has heading stuck at 122.03° at times: {stuck_122}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'run_{run_id}_telemetry.png'), dpi=300)
        plt.close()
    
    print(f"Telemetry plots saved to {output_dir}")

def analyze_runs(runs, output_dir):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each run and write statistics to a file
    with open(os.path.join(output_dir, 'run_statistics.txt'), 'w') as f:
        f.write(f"RC Car Navigation Analysis\n")
        f.write(f"========================\n\n")
        f.write(f"Total runs analyzed: {len(runs)}\n\n")
        
        for run in runs:
            run_id = run['id']
            f.write(f"Run {run_id}:\n")
            f.write(f"--------\n")
            
            if 'start_time' in run and 'end_time' in run:
                duration = (run['end_time'] - run['start_time']) / 1000  # seconds
                f.write(f"Duration: {duration:.2f} seconds\n")
            
            if run['controls']:
                # Speed statistics
                speeds = [c['speed'] for c in run['controls']]
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
                f.write(f"Average speed: {avg_speed:.2f} m/s\n")
                f.write(f"Maximum speed: {max_speed:.2f} m/s\n")
                
                # Heading statistics
                headings = [c['heading'] for c in run['controls']]
                
                # Calculate heading changes
                heading_changes = []
                for i in range(1, len(headings)):
                    change = headings[i] - headings[i-1]
                    # Handle angle wrapping
                    if change > 180:
                        change -= 360
                    elif change < -180:
                        change += 360
                    heading_changes.append(abs(change))
                
                if heading_changes:
                    avg_change = sum(heading_changes) / len(heading_changes)
                    f.write(f"Average heading change: {avg_change:.2f}°\n")
                    
                    # Analyze stuck headings
                    stuck_count = sum(1 for c in heading_changes if c < 0.1)
                    stuck_percentage = (stuck_count / len(heading_changes)) * 100
                    f.write(f"Potentially stuck heading: {stuck_percentage:.1f}% of the time\n")
                    
                    # Check for 122.03° specifically
                    stuck_at_122 = sum(1 for h in headings if abs(h - 122.03) < 0.01)
                    if stuck_at_122 > 0:
                        f.write(f"Heading stuck at 122.03°: {stuck_at_122} times\n")
            
            f.write("\n")
    
    print(f"Analysis complete. Results saved to {os.path.join(output_dir, 'run_statistics.txt')}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='RC Car Telemetry Analyzer')
    parser.add_argument('--log', required=True, help='Path to log file')
    parser.add_argument('--output', default='rc_analysis', help='Output directory')
    args = parser.parse_args()
    
    print(f"Analyzing log file: {args.log}")
    runs = parse_runs(args.log)
    print(f"Found {len(runs)} navigation runs")
    
    plot_telemetry(runs, args.output)
    analyze_runs(runs, args.output)

if __name__ == "__main__":
    main()