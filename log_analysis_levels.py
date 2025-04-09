import re
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
import argparse

# python log_analysis_levels.py --log log.txt --ignore "RTCM age" "Websocket gnssMutex acq" "ntrip mutex acq" "ntripClientMutex hold (no data)" "gnssMutex hold" "gnssMutex take" --by-level
# python log_analysis_levels.py --log log.txt --by-level --min-duration 0 --cut-before-latency

def is_ignored(operation_name, ignored_list):
    """
    Check if an operation should be ignored using flexible matching
    that's tolerant of whitespace differences
    """
    # Normalize the operation name: convert to lowercase and normalize whitespace
    normalized_op = ' '.join(operation_name.lower().split())
    
    for ignored_pattern in ignored_list:
        # Normalize the ignored pattern as well
        normalized_pattern = ' '.join(ignored_pattern.lower().split())
        
        # Check if the normalized pattern appears anywhere in the normalized operation name
        if normalized_pattern in normalized_op:
            return True
    
    return False

def analyze_log_file(log_file_path, min_duration_ms=0, ignored_operations=None, cut_before_latency=True):

    """
    Analyze ESP32 FreeRTOS log file for timing data
    
    Args:
        log_file_path: Path to the log file
        min_duration_ms: Minimum duration to include in analysis (milliseconds)
        ignored_operations: List of operation names to ignore (default: None)
        
    Returns:
        Dictionary containing timing data organized by session and operation type
    """
    # Initialize ignored_operations as empty list if None
    if ignored_operations is None:
        ignored_operations = []
    
    # Print the ignored operations
    print("Ignored operations:")
    for op in ignored_operations:
        print(f"  - '{op}'")
    
    # Initialize data structures
    sessions = []
    current_session = {
        'start_time': None,
        'operations': defaultdict(list),
        'start_timestamp': None,
        'name': f"Session 1",
        'fusion_status_calls': [],  # Track timestamps of getFusionStatus calls
        'log_levels': {}  # Store log level for each operation
    }
    session_count = 1
    system_online = False
    
    # Regular expressions for parsing
    time_pattern = re.compile(r'(.+?)time, (\d+)')
    
        # If cutting before latency, find the first latency timestamp
    first_latency_timestamp = None
    last_latency_timestamp = None
    if cut_before_latency:
        with open(log_file_path, 'r') as f:
            for line in f:
                if 'position-control latency time' in line:
                    try:
                        parts = line.strip().split(',', 3)
                        if len(parts) >= 1:
                            ts = int(parts[0])
                            if first_latency_timestamp is None:
                                first_latency_timestamp = ts
                            last_latency_timestamp = ts  # keeps updating until last
                    except:
                        continue

    # Read the log file
    with open(log_file_path, 'r') as f:
        for line in f:
            # Skip header or empty lines
            if line.startswith('----- New Logging Session Started -----') or \
            line.startswith('Timestamp,Level,Task,Message') or \
            not line.strip():
                continue

            try:
                parts = line.strip().split(',', 3)
                if len(parts) < 4:
                    continue

                timestamp_str, level, task, message = parts
                timestamp = int(timestamp_str)

                # Always check for restart/system online markers, even if before cutoff
                if "SYSTEM RESTART DETECTED" in message:
                    if current_session['start_time'] is not None:
                        sessions.append(current_session)
                        session_count += 1
                    current_session = {
                        'start_time': timestamp,
                        'operations': defaultdict(list),
                        'start_timestamp': timestamp,
                        'name': f"Session {session_count}",
                        'fusion_status_calls': [],
                        'log_levels': {}
                    }
                    system_online = False
                    continue

                if "GNSS SYS TIME:" in message:
                    system_online = True
                    current_session['start_time'] = timestamp
                    continue

                # Skip non-init lines outside the latency window
                if cut_before_latency and (
                    (first_latency_timestamp and timestamp < first_latency_timestamp) or
                    (last_latency_timestamp and timestamp > last_latency_timestamp)
                ):
                    # Allow GNSS SYS TIME and restart detection even outside window
                    if "SYSTEM RESTART DETECTED" not in message and "GNSS SYS TIME:" not in message:
                        continue


                # Skip processing until system is online
                if not system_online:
                    continue

                # Track getFusionStatus calls
                if "about to call getFusionStatus" in message or "getFusionStatus" in message:
                    relative_time = timestamp - current_session['start_time']
                    current_session['fusion_status_calls'].append(relative_time)

                # Special case: detect ControlTask loop beginning
                if "ControlTask loop beginning" in message:
                    relative_time = timestamp - current_session['start_time']
                    current_session['operations']["ControlTask loop beginning"].append((relative_time, 0))
                    if "ControlTask loop beginning" not in current_session['log_levels']:
                        current_session['log_levels']["ControlTask loop beginning"] = level
                    continue

                # General case: match timed operations
                time_match = time_pattern.search(message)
                if time_match:
                    operation_type = message.rsplit(',', 1)[0].strip()
                    duration = int(time_match.group(2))

                    if is_ignored(operation_type, ignored_operations):
                        return

                    relative_time = timestamp - current_session['start_time']
                    current_session['operations'][operation_type].append((relative_time, duration))
                    if operation_type not in current_session['log_levels']:
                        current_session['log_levels'][operation_type] = level

            except Exception as e:
                print(f"Error processing line: {line}")
                print(f"Exception: {e}")
                continue

    # Add the last session
    if current_session['start_time'] is not None:
        sessions.append(current_session)

    for s in sessions:
        print(f"\nParsed operations in {s['name']}:")
        for op in s['operations']:
            print(f"  - {op}")

    return sessions

def generate_statistics(sessions):
    """
    Generate statistics for each operation type in each session
    
    Args:
        sessions: List of session dictionaries
        
    Returns:
        Dictionary of statistics by session and operation
    """
    statistics = {}
    
    for session in sessions:
        session_stats = {}
        
        for operation, data_points in session['operations'].items():
            if not data_points:
                continue
                
            durations = [point[1] for point in data_points]
            
            # Calculate statistics
            stats = {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'median': np.median(durations),
                'count': len(durations),
                'total_time': sum(durations),
                'level': session['log_levels'].get(operation, 'UNKNOWN')
            }
            
            session_stats[operation] = stats
            
        statistics[session['name']] = session_stats
        
    return statistics

def extract_loop_timing_table(sessions, base_op='position-control latency time'):
    """
    Extract loop-level timing data. Each row is one ControlTask loop.
    """
    for session in sessions:
        loops = []
        loop_times = sorted([t for t, _ in session['operations'].get("ControlTask loop beginning", [])])
        if not loop_times:
            print(f"No control loop boundaries found in {session['name']}")
            continue

        for i in range(len(loop_times) - 1):
            start, end = loop_times[i], loop_times[i + 1]
            loop_data = {'loop_start': start}
            for op, entries in session['operations'].items():
                vals = [v for t, v in entries if start <= t < end]
                if vals:
                    loop_data[op] = vals[-1]  # use last seen value in loop
            if base_op in loop_data:
                loops.append(loop_data)

        session['loop_table'] = loops

def debug_loop_operation_presence(sessions, base_op='position-control latency time'):
    for session in sessions:
        loops = session.get('loop_table', [])
        print(f"\n--- Debug: Loop data for {session['name']} ---")
        print(f"Total loops parsed: {len(loops)}")

        if not loops:
            continue

        op_set = set()
        for row in loops:
            op_set.update(row.keys())
        op_set.discard('loop_start')

        print(f"Operations seen in loop data: {sorted(op_set)}")

        for op in sorted(op_set):
            present = sum(1 for row in loops if op in row)
            print(f"  {op:40s}: present in {present} loops")

def compute_correlations_from_loops(sessions, base_op='position-control latency time', outfile=None, min_pairs=5):
    for session in sessions:
        loops = session.get('loop_table', [])
        if not loops:
            print(f"No loop timing data found in {session['name']}")
            continue

        all_keys = set()
        for row in loops:
            all_keys.update(row.keys())
        all_keys.discard('loop_start')
        if base_op not in all_keys:
            print(f"Base operation '{base_op}' not found in loop data for {session['name']}")
            continue
        all_keys.discard(base_op)

        results = []
        for k in sorted(all_keys):
            paired = [(row[base_op], row[k]) for row in loops if base_op in row and k in row]
            if len(paired) < min_pairs:
                print(f"Not enough data points to compute correlation between '{base_op}' and '{k}' ({len(paired)} found)")
                continue
            x, y = zip(*paired)
            corr = np.corrcoef(x, y)[0, 1]
            results.append((k, corr))

        if outfile:
            with open(outfile, 'a') as f:
                f.write(f"\nCorrelation with '{base_op}' in {session['name']} (loop-level):\n")
                if results:
                    for op, r in results:
                        f.write(f"  {op:40s}: r = {r:.3f}\n")
                else:
                    f.write("  No sufficient data for correlation analysis.\n")
        else:
            print(f"\nCorrelation with '{base_op}' in {session['name']} (loop-level):")
            if results:
                for op, r in results:
                    print(f"  {op:40s}: r = {r:.3f}")
            else:
                print("  No sufficient data for correlation analysis.")

def categorize_operations(sessions, num_categories=2):
    """
    Categorize operations by their average duration
    
    Args:
        sessions: List of session dictionaries
        num_categories: Number of categories to create (default: 2)
        
    Returns:
        Dictionary mapping each operation to its category
    """
    # Collect average durations for all operations across all sessions
    operation_avg_durations = {}
    
    for session in sessions:
        for operation, data_points in session['operations'].items():
            if not data_points:
                continue
                
            durations = [point[1] for point in data_points]
            avg_duration = sum(durations) / len(durations)
            
            if operation in operation_avg_durations:
                # Average with existing value to account for multiple sessions
                existing_avg = operation_avg_durations[operation]
                operation_avg_durations[operation] = (existing_avg + avg_duration) / 2
            else:
                operation_avg_durations[operation] = avg_duration
    
    # Sort operations by average duration
    sorted_operations = sorted(operation_avg_durations.items(), key=lambda x: x[1])
    
    # Create categories
    categories = {}
    operations_per_category = max(1, len(sorted_operations) // num_categories)
    
    for i, (operation, _) in enumerate(sorted_operations):
        category_index = min(i // operations_per_category, num_categories - 1)
        categories[operation] = category_index
    
    return categories

def plot_timing_data(sessions, output_dir, num_categories=2):
    """
    Create time series plots of operation timing data grouped by duration category
    
    Args:
        sessions: List of session dictionaries
        output_dir: Directory to save plots
        num_categories: Number of categories for grouping operations by duration
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Categorize operations by average duration
    operation_categories = categorize_operations(sessions, num_categories)
    
    # Plot each session separately
    for session in sessions:
        # Only plot operations with data points
        operations = {op: data for op, data in session['operations'].items() if data}
        
        if not operations:
            continue
            
        # Make sure all operations in this session are categorized
        for op in operations:
            if op not in operation_categories:
                # Find the average duration for this operation
                data_points = operations[op]
                durations = [point[1] for point in data_points]
                avg_duration = sum(durations) / len(durations)
                
                # Assign a category based on the average duration
                if operation_categories:
                    # Get all unique category values
                    category_values = set(operation_categories.values())
                    # Choose the middle category if possible
                    middle_category = len(category_values) // 2
                    operation_categories[op] = middle_category
                else:
                    # Assign to category 0 if no other operations categorized
                    operation_categories[op] = 0
        
        # Create a separate plot for each category
        for category in range(num_categories):
            # Only include operations that belong to this category
            category_operations = {op: data for op, data in operations.items() 
                                if op in operation_categories and operation_categories[op] == category}
            
            if not category_operations:
                continue
                
            plt.figure(figsize=(14, 8))
            
            # Debug: Print operations in this category
            print(f"Session {session['name']} - Category {category+1} operations: {list(category_operations.keys())}")
            
            # Clear any existing plots
            plt.clf()
            
            # Create empty lists to store legend entries
            lines = []
            labels = []
            
            # Plot each operation type within this category
            for operation, data_points in category_operations.items():
                times = [point[0]/1000 for point in data_points]  # Convert to seconds
                durations = [point[1] for point in data_points]
                
                # Store the line object to add to legend manually
                line, = plt.plot(times, durations, 'o-', 
                                alpha=0.7, 
                                markersize=4)
                
                lines.append(line)
                labels.append(operation)
            
            # Create legend manually from all plotted lines
            plt.legend(lines, labels, loc='upper right')
            
            # Add vertical lines for getFusionStatus calls
            for call_time in session['fusion_status_calls']:
                plt.axvline(x=call_time/1000, color='black', linestyle='--', alpha=0.5)
                plt.text(call_time/1000, plt.ylim()[1]*0.95, 'getFusionStatus', 
                         rotation=90, verticalalignment='top', alpha=0.7)
            
            plt.title(f"Operation Timing - {session['name']} - Category {category+1}")
            plt.xlabel('Time (seconds since start)')
            plt.ylabel('Duration (ms)')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_file = os.path.join(output_dir, 
                                     f"{session['name'].replace(' ', '_')}_category{category+1}_timing.png")
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Plot saved to: {plot_file}")
        
        # Create a combined plot with all operations
        plt.figure(figsize=(14, 8))
        plt.clf()  # Clear any existing plots
        
        # Create empty lists to store legend entries
        lines = []
        labels = []
        
        # Plot each operation with a unique color
        for operation, data_points in operations.items():
            times = [point[0]/1000 for point in data_points]  # Convert to seconds
            durations = [point[1] for point in data_points]
            
            # Store the line object
            line, = plt.plot(times, durations, 'o-', 
                            alpha=0.7, 
                            markersize=4)
            
            lines.append(line)
            labels.append(operation)
        
        # Create legend manually
        plt.legend(lines, labels, loc='upper right')
        
        # Add vertical lines for getFusionStatus calls
        for call_time in session['fusion_status_calls']:
            plt.axvline(x=call_time/1000, color='black', linestyle='--', alpha=0.5)
            plt.text(call_time/1000, plt.ylim()[1]*0.95, 'getFusionStatus', 
                     rotation=90, verticalalignment='top', alpha=0.7)
        
        plt.title(f"Operation Timing - {session['name']} - All Operations")
        plt.xlabel('Time (seconds since start)')
        plt.ylabel('Duration (ms)')
        plt.grid(True, alpha=0.3)
        
        # Save the combined plot
        plot_file = os.path.join(output_dir, f"{session['name'].replace(' ', '_')}_all_timing.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Combined plot saved to: {plot_file}")

def plot_timing_data_by_level(sessions, output_dir):
    """
    Create time series plots of operation timing data grouped by log level
    
    Args:
        sessions: List of session dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each session separately
    for session in sessions:
        # Only plot operations with data points
        operations = {op: data for op, data in session['operations'].items() if data}
        
        if not operations:
            continue
        
        # Group operations by log level
        level_operations = defaultdict(dict)
        for op in operations:
            level = session['log_levels'].get(op, 'UNKNOWN')  # Default to UNKNOWN if missing
            level_operations[level][op] = operations[op]
        
        # Create a separate plot for each log level
        for level, level_ops in level_operations.items():
            if not level_ops:
                continue
                
            plt.figure(figsize=(14, 8))
            
            # Debug: Print operations in this level
            print(f"Session {session['name']} - {level} operations: {list(level_ops.keys())}")
            
            # Clear any existing plots
            plt.clf()
            
            # Create empty lists to store legend entries
            lines = []
            labels = []
            
            # Plot each operation type within this level
            for operation, data_points in level_ops.items():
                times = [point[0]/1000 for point in data_points]  # Convert to seconds
                durations = [point[1] for point in data_points]
                
                # Store the line object
                line, = plt.plot(times, durations, 'o-', 
                                alpha=0.7, 
                                markersize=4)
                
                lines.append(line)
                labels.append(operation)
            
            # Create legend manually
            plt.legend(lines, labels, loc='upper right')
            
            # Add vertical lines for getFusionStatus calls
            for call_time in session['fusion_status_calls']:
                plt.axvline(x=call_time/1000, color='black', linestyle='--', alpha=0.5)
                plt.text(call_time/1000, plt.ylim()[1]*0.95, 'getFusionStatus', 
                         rotation=90, verticalalignment='top', alpha=0.7)
            
            plt.title(f"Operation Timing - {session['name']} - {level} Level")
            plt.xlabel('Time (seconds since start)')
            plt.ylabel('Duration (ms)')
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_file = os.path.join(output_dir, 
                                    f"{session['name'].replace(' ', '_')}_{level}_timing.png")
            plt.savefig(plot_file)
            plt.close()
            
            print(f"Plot saved to: {plot_file}")
        
        # Still create a combined plot with all operations
        plt.figure(figsize=(14, 8))
        plt.clf()  # Clear any existing plots
        
        # Create empty lists to store legend entries
        lines = []
        labels = []
        
        # Plot each operation with a unique color
        for operation, data_points in operations.items():
            times = [point[0]/1000 for point in data_points]  # Convert to seconds
            durations = [point[1] for point in data_points]
            
            # Store the line object
            line, = plt.plot(times, durations, 'o-', 
                            alpha=0.7, 
                            markersize=4)
            
            lines.append(line)
            labels.append(operation)
        
        # Create legend manually
        plt.legend(lines, labels, loc='upper right')
        
        # Add vertical lines for getFusionStatus calls
        for call_time in session['fusion_status_calls']:
            plt.axvline(x=call_time/1000, color='black', linestyle='--', alpha=0.5)
            plt.text(call_time/1000, plt.ylim()[1]*0.95, 'getFusionStatus', 
                     rotation=90, verticalalignment='top', alpha=0.7)
        
        plt.title(f"Operation Timing - {session['name']} - All Operations")
        plt.xlabel('Time (seconds since start)')
        plt.ylabel('Duration (ms)')
        plt.grid(True, alpha=0.3)
        
        # Save the combined plot
        plot_file = os.path.join(output_dir, f"{session['name'].replace(' ', '_')}_all_timing.png")
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Combined plot saved to: {plot_file}")

def save_statistics(statistics, output_dir, summary_filename=None):

    """
    Save statistics to a CSV file
    
    Args:
        statistics: Dictionary of statistics
        output_dir: Directory to save output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats_file = os.path.join(output_dir, f"timing_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(stats_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Session', 'Operation', 'Level', 'Min (ms)', 'Max (ms)', 'Avg (ms)', 'Median (ms)', 'Count', 'Total Time (ms)'])
        
        for session_name, session_stats in statistics.items():
            for operation, stats in session_stats.items():
                writer.writerow([
                    session_name,
                    operation,
                    stats.get('level', 'UNKNOWN'),
                    stats['min'],
                    stats['max'],
                    round(stats['avg'], 2),
                    stats['median'],
                    stats['count'],
                    stats['total_time']
                ])
    
    print(f"Statistics saved to: {stats_file}")
    
    if summary_filename is None:
        summary_filename = f"timing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    summary_file = os.path.join(output_dir, summary_filename)
    
    with open(summary_file, 'w') as f:
        f.write("ESP32 FreeRTOS Log Timing Analysis\n")
        f.write("===============================\n\n")
        
        for session_name, session_stats in statistics.items():
            f.write(f"{session_name}\n")
            f.write("=" * len(session_name) + "\n\n")
            
            # Group operations by log level
            level_operations = defaultdict(list)
            for operation, stats in session_stats.items():
                level = stats.get('level', 'UNKNOWN')
                level_operations[level].append((operation, stats))
            
            # Print operations grouped by level
            for level, operations in level_operations.items():
                f.write(f"Log Level: {level}\n")
                f.write("-" * (len(level) + 11) + "\n\n")
                
                # Sort operations by max time (descending)
                sorted_ops = sorted(operations, key=lambda x: x[1]['max'], reverse=True)
                
                for operation, stats in sorted_ops:
                    f.write(f"Operation: {operation}\n")
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Min: {stats['min']} ms\n")
                    f.write(f"  Max: {stats['max']} ms\n")
                    f.write(f"  Avg: {round(stats['avg'], 2)} ms\n")
                    f.write(f"  Median: {stats['median']} ms\n")
                    f.write(f"  Total Time: {stats['total_time']} ms\n\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    # Example: python log_analyzer.py --log your_log.txt --ignore "RTCM age" "Websocket gnssMutex acq" "ntrip mutex acq" --by-level
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Analyze ESP32 FreeRTOS log file for timing data.')
    parser.add_argument('--log', default='log.txt', help='Path to the log file')
    parser.add_argument('--min-duration', type=int, default=1, help='Minimum duration to include (ms)')
    parser.add_argument('--ignore', nargs='+', default=[], help='Operation types to ignore (space separated)')
    parser.add_argument('--by-level', action='store_true', help='Group operations by log level instead of duration')
    parser.add_argument('--cut-before-latency', action='store_true',
                    help='Ignore all log entries before first "position-control latency time" message')
    
    args = parser.parse_args()
    
    # Get current timestamp for output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"log_analysis_{timestamp}"
    
    # Use command-line arguments
    log_file_path = args.log
    min_duration_ms = args.min_duration
    ignored_operations = args.ignore
    by_level = args.by_level
    cut = args.cut_before_latency
    
    # Print startup message
    print(f"ESP32 FreeRTOS Log Analyzer")
    print(f"========================")
    
    # Check if the file exists
    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        log_file_path = input("Enter the path to your log file: ")
        
        if not os.path.exists(log_file_path):
            print(f"Log file not found: {log_file_path}")
            return
    
    print(f"Analyzing log file: {log_file_path}")
    print(f"Minimum duration: {min_duration_ms} ms")
    print(f"Grouping by: {'Log Level' if by_level else 'Duration Category'}")
    
    # Analyze the log file
    sessions = analyze_log_file(log_file_path, min_duration_ms=min_duration_ms, ignored_operations=ignored_operations, cut_before_latency=cut)
    
    if not sessions:
        print("No valid sessions found in the log file.")
        return
        
    print(f"Found {len(sessions)} session(s) in the log file.")
    
    # Report on getFusionStatus calls
    for session in sessions:
        fusion_calls = session['fusion_status_calls']
        if fusion_calls:
            print(f"{session['name']}: Found {len(fusion_calls)} getFusionStatus calls")
        else:
            print(f"{session['name']}: No getFusionStatus calls detected")
    
    # Generate statistics
    statistics = generate_statistics(sessions)
    
    summary_filename = f"timing_summary_{timestamp}.txt"
    summary_file = os.path.join(output_dir, summary_filename)
    save_statistics(statistics, output_dir, summary_filename=summary_filename)

    # Extract and compute correlations, then append to summary
    extract_loop_timing_table(sessions)
    debug_loop_operation_presence(sessions)
    compute_correlations_from_loops(sessions, outfile=summary_file)

    # Create plots based on grouping preference
    if by_level:
        plot_timing_data_by_level(sessions, output_dir)
    else:
        plot_timing_data(sessions, output_dir, num_categories=2)

    print(f"Analysis complete. Results saved to: {output_dir}")

if __name__ == "__main__":
    main()