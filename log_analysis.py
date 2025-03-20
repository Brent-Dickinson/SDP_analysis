import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime

# Customizable output suffix - change this to create different output folders
OUTPUT_SUFFIX = "20250320_firstAsync"  # Will create "figures_20250320" folder and "stats_20250320.txt"

# Define file paths
log_file = "log.txt"

# Create output folder paths
figures_folder = f"figures_{OUTPUT_SUFFIX}"
stats_file = f"stats_{OUTPUT_SUFFIX}.txt"

# Define markers for various phases
INIT_END_MARKER = "Setup complete. RTOS scheduler taking over"
GPS_FIX_MARKER = "System time set from GNSS:"  # Using this as a proxy for GPS fix since system time is set after fix type 3
RTCM_MARKER = "RTCM correction status changed: 0 -> 2"  # Another indicator of moving to operational phase
SYSTEM_TIME_SET_MARKER = "System time set from GNSS:"  # For extracting date/time

def parse_log_file(log_file):
    """Reads log file into a DataFrame, extracts useful data."""
    logs = []
    phases = {
        "init": True,       # Start in initialization phase
        "connecting": False,  # After init but before first GPS fix
        "operational": False  # After first GPS fix achieved
    }
    
    # Track fix type state
    has_fix_type_3 = False
    system_date_time = None
    
    with open(log_file, 'r') as file:
        # Skip header line if it exists
        first_line = file.readline()
        if not first_line.startswith("Timestamp"):
            file.seek(0)  # Reset to start of file if no header
        
        for line in file:
            # Skip non-data lines
            if not re.match(r'^\d+,', line):
                continue
                
            parts = line.strip().split(',', 3)  # Split into 4 parts
            if len(parts) < 4:
                continue  # Skip malformed lines
            
            timestamp, level, task, message = parts
            timestamp = int(timestamp)
            
            # Extract system date/time if present
            if SYSTEM_TIME_SET_MARKER in message:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', message)
                if date_match:
                    system_date_time = date_match.group(1)
                # System time is set only when fix type 3+ is achieved
                has_fix_type_3 = True
            
            # Check for other fix type indicators
            if "fix type" in message.lower() or "fixType" in message:
                if "3" in message or "fix type 3" in message.lower():
                    has_fix_type_3 = True
                elif "No Fix" in message or "no fix" in message.lower():
                    has_fix_type_3 = False
            
            # Detect phase transitions
            if phases["init"] and INIT_END_MARKER in message:
                phases["init"] = False
                phases["connecting"] = True
            
            # Transition to operational after first fix type 3
            if phases["connecting"] and has_fix_type_3:
                phases["connecting"] = False
                phases["operational"] = True
            
            # Determine current phase
            current_phase = "initialization" if phases["init"] else ("connecting" if phases["connecting"] else "operational")
            
            logs.append([timestamp, level, task, message, current_phase, has_fix_type_3])

    df = pd.DataFrame(logs, columns=["Timestamp", "Level", "Task", "Message", "Phase", "HasFixType3"])
    
    # Add relative time (seconds from start)
    if not df.empty:
        start_time = df["Timestamp"].min()
        df["RelativeTime"] = (df["Timestamp"] - start_time) / 1000.0  # Convert to seconds
    
    return df, system_date_time

def extract_task_timing(df):
    """Extract timing data for each task."""
    timing_data = []
    
    # Extract task timing information
    for _, row in df.iterrows():
        timestamp, task, message, phase, rel_time, has_fix_type_3 = row["Timestamp"], row["Task"], row["Message"], row["Phase"], row["RelativeTime"], row["HasFixType3"]
        
        # WebSocket task timing
        ws_total_match = re.search(r"Total (.*?) process took (\d+) ms", message)
        if ws_total_match and task == "WebSocketTask":
            event_type = f"WS {ws_total_match.group(1)}"
            duration = int(ws_total_match.group(2))
            timing_data.append([timestamp, rel_time, task, event_type, duration, phase, has_fix_type_3])
            continue
            
        ws_loop_match = re.search(r"WebSocket task loop iteration took (\d+) ms", message)
        if ws_loop_match:
            timing_data.append([timestamp, rel_time, task, "WS Loop", int(ws_loop_match.group(1)), phase, has_fix_type_3])
            continue

        # WebSocket.loop() timing
        ws_lib_loop_match = re.search(r"WebSocket\.loop\(\) took (\d+) ms", message)
        if ws_lib_loop_match:
            timing_data.append([timestamp, rel_time, task, "WebSocket.loop()", int(ws_lib_loop_match.group(1)), phase, has_fix_type_3])
            continue

        # WS GPS data update timing
        ws_gps_total_match = re.search(r"Total GPS data update process took (\d+) ms", message)
        if ws_gps_total_match:
            timing_data.append([timestamp, rel_time, task, "WS GPS Update", int(ws_gps_total_match.group(1)), phase, has_fix_type_3])
            continue
        
        # Mutex acquisition time
        mutex_acquire_match = re.search(r"mutex acquired (?:after|.+?after) (\d+) ms", message)
        if mutex_acquire_match:
            mutex_type = "GNSS" if "GNSS mutex" in message else "NTRIP" if "ntripClient mutex" in message else "Unknown"
            timing_data.append([timestamp, rel_time, task, f"{mutex_type} Mutex Acquire", int(mutex_acquire_match.group(1)), phase, has_fix_type_3])
            continue
            
        # Mutex hold time
        mutex_hold_match = re.search(r"(?:released|Released).+?held for (\d+) ms", message)
        if mutex_hold_match:
            mutex_type = "GNSS" if "GNSS mutex" in message else "NTRIP" if "ntripClient mutex" in message else "Unknown"
            timing_data.append([timestamp, rel_time, task, f"{mutex_type} Mutex Hold", int(mutex_hold_match.group(1)), phase, has_fix_type_3])
            continue
            
        # RTCM correction age
        rtcm_match = re.search(r"RTCM correction age: (\d+) ms", message)
        if rtcm_match:
            timing_data.append([timestamp, rel_time, task, "RTCM Age", int(rtcm_match.group(1)), phase, has_fix_type_3])
            continue
            
        # RTCM data push time
        rtcm_push_match = re.search(r"Pushed RTCM data to GPS module \(took (\d+) ms\)", message)
        if rtcm_push_match:
            timing_data.append([timestamp, rel_time, task, "RTCM Push", int(rtcm_push_match.group(1)), phase, has_fix_type_3])
            continue
            
        # RTCM data read time
        rtcm_read_match = re.search(r"Received \d+ bytes of RTCM data \(read took (\d+) ms\)", message)
        if rtcm_read_match:
            timing_data.append([timestamp, rel_time, task, "RTCM Read", int(rtcm_read_match.group(1)), phase, has_fix_type_3])
            continue
    
    return pd.DataFrame(timing_data, columns=["Timestamp", "RelativeTime", "Task", "EventType", "Duration", "Phase", "HasFixType3"])

def plot_operational_timing(timing_df, system_date_time):
    """Create plots focused exclusively on the operational period with Fix Type 3."""
    # Filter for operational phase with Fix Type 3
    op_df = timing_df[(timing_df["HasFixType3"] == True) & (timing_df["EventType"] != "RTCM Age")]
    
    if op_df.empty:
        print("No Fix Type 3 operational data found.")
        return None
    
    # Set up the plot
    plt.figure(figsize=(15, 10))
    title = "Operational Phase (Fix Type 3) Timing Analysis"
    if system_date_time:
        title += f" ({system_date_time})"
    plt.suptitle(title, fontsize=16)
    
    # Group data by event type
    event_types = op_df["EventType"].unique()
    
    # Plot timing by event type
    plt.subplot(2, 1, 1)
    for event_type in event_types:
        event_df = op_df[op_df["EventType"] == event_type]
        if len(event_df) > 0:
            plt.plot(event_df["RelativeTime"], event_df["Duration"], 'o-', alpha=0.7, label=event_type)
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Duration (ms)")
    plt.title("Fix Type 3 - Event Durations Over Time")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    
    # Box plot of durations by event type
    plt.subplot(2, 1, 2)
    
    # Get only event types with enough data
    event_counts = op_df["EventType"].value_counts()
    valid_events = event_counts[event_counts > 3].index
    
    if len(valid_events) > 0:
        box_data = [op_df[op_df["EventType"] == event]["Duration"] for event in valid_events]
        plt.boxplot(box_data, labels=valid_events, vert=True)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Duration (ms)")
        plt.title("Fix Type 3 - Distribution of Event Durations")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure to the custom folder
    output_path = os.path.join(figures_folder, "operational_timing_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Collect statistics
    stats = {}
    for event_type in event_types:
        event_df = op_df[op_df["EventType"] == event_type]
        if not event_df.empty:
            stats[event_type] = {
                "count": len(event_df),
                "mean": event_df["Duration"].mean(),
                "median": event_df["Duration"].median(),
                "min": event_df["Duration"].min(),
                "max": event_df["Duration"].max(),
                "std": event_df["Duration"].std()
            }
    
    return stats

def plot_rtcm_correction_age(timing_df, system_date_time):
    """Plot RTCM correction age as a completely separate chart."""
    # Filter for RTCM Age data in operational phase with Fix Type 3
    rtcm_age_df = timing_df[(timing_df["HasFixType3"] == True) & 
                           (timing_df["EventType"] == "RTCM Age")]
    
    if rtcm_age_df.empty:
        print("No RTCM Age data found for operational phase.")
        return None
    
    # Set up the plot
    plt.figure(figsize=(15, 8))
    title = "RTCM Correction Age (Operational Phase Only)"
    if system_date_time:
        title += f" ({system_date_time})"
    plt.suptitle(title, fontsize=16)
    
    # Plot RTCM correction age
    plt.plot(rtcm_age_df["RelativeTime"], rtcm_age_df["Duration"], 'o-', color='blue', alpha=0.7)
    
    # Add horizontal line at 1000ms (1 second) for reference
    plt.axhline(y=1000, color='r', linestyle='--', alpha=0.7, label='1 second threshold')
    
    # Add statistical information
    mean_age = rtcm_age_df["Duration"].mean()
    max_age = rtcm_age_df["Duration"].max()
    median_age = rtcm_age_df["Duration"].median()
    min_age = rtcm_age_df["Duration"].min()
    std_age = rtcm_age_df["Duration"].std()
    
    stats_text = f"Mean: {mean_age:.1f} ms\nMax: {max_age:.1f} ms\nMedian: {median_age:.1f} ms"
    plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("RTCM Correction Age (ms)")
    plt.title("RTCM Correction Age Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure to the custom folder
    output_path = os.path.join(figures_folder, "rtcm_correction_age.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Collect statistics
    stats = {
        "RTCM Age": {
            "count": len(rtcm_age_df),
            "mean": mean_age,
            "median": median_age,
            "min": min_age,
            "max": max_age,
            "std": std_age,
            "values_over_1s": len(rtcm_age_df[rtcm_age_df["Duration"] > 1000]),
            "percent_over_1s": len(rtcm_age_df[rtcm_age_df["Duration"] > 1000]) / len(rtcm_age_df) * 100 if len(rtcm_age_df) > 0 else 0
        }
    }
    
    return stats

def save_statistics(stats_dict, filename):
    """Save all statistics to a text file."""
    with open(filename, 'w') as f:
        f.write(f"Log Analysis Statistics - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Based on log file: {log_file}\n\n")
        
        # Overall statistics
        if "overall" in stats_dict:
            f.write("=" * 50 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("=" * 50 + "\n")
            
            for key, value in stats_dict["overall"].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # RTCM Age statistics
        if "rtcm_age" in stats_dict:
            f.write("=" * 50 + "\n")
            f.write("RTCM CORRECTION AGE STATISTICS\n")
            f.write("=" * 50 + "\n")
            
            rtcm_stats = stats_dict["rtcm_age"]["RTCM Age"]
            f.write(f"Sample count: {rtcm_stats['count']}\n")
            f.write(f"Mean: {rtcm_stats['mean']:.2f} ms\n")
            f.write(f"Median: {rtcm_stats['median']:.2f} ms\n")
            f.write(f"Min: {rtcm_stats['min']:.2f} ms\n")
            f.write(f"Max: {rtcm_stats['max']:.2f} ms\n")
            f.write(f"Standard deviation: {rtcm_stats['std']:.2f} ms\n")
            f.write(f"Values over 1 second: {rtcm_stats['values_over_1s']} ({rtcm_stats['percent_over_1s']:.2f}%)\n")
            f.write("\n")
        
        # Timing statistics
        if "timing" in stats_dict:
            f.write("=" * 50 + "\n")
            f.write("OPERATIONAL PHASE TIMING STATISTICS\n")
            f.write("=" * 50 + "\n")
            
            for event_type, stats in stats_dict["timing"].items():
                f.write(f"\n--- {event_type} ---\n")
                f.write(f"Sample count: {stats['count']}\n")
                f.write(f"Mean: {stats['mean']:.2f} ms\n")
                f.write(f"Median: {stats['median']:.2f} ms\n")
                f.write(f"Min: {stats['min']:.2f} ms\n")
                f.write(f"Max: {stats['max']:.2f} ms\n")
                f.write(f"Standard deviation: {stats['std']:.2f} ms\n")
            f.write("\n")

def main():
    # Set plot style
    plt.style.use('ggplot')
    
    # Create figures directory if it doesn't exist
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)
        print(f"Created figures directory: {figures_folder}")
    
    # Parse log file
    print(f"Parsing log file {log_file}...")
    df, system_date_time = parse_log_file(log_file)
    
    if df.empty:
        print("No log data found or could not parse log file.")
        return
    
    print(f"Log data parsed. Found {len(df)} log entries.")
    if system_date_time:
        print(f"System date/time: {system_date_time}")
    
    # Extract timing data
    timing_df = extract_task_timing(df)
    
    # Filter for only operational phase
    op_df = df[df["Phase"] == "operational"]
    op_timing_df = timing_df[timing_df["Phase"] == "operational"]
    
    # Print operational phase stats
    op_count = len(op_df)
    total_count = len(df)
    op_percentage = op_count/total_count*100 if total_count > 0 else 0
    print(f"Operational phase entries: {op_count} ({op_percentage:.1f}%)")
    
    # Initialize statistics dictionary
    all_stats = {
        "overall": {
            "total_log_entries": len(df),
            "operational_phase_entries": op_count,
            "operational_phase_percentage": op_percentage,
            "system_date_time": system_date_time if system_date_time else "Unknown"
        }
    }
    
    # Generate plots for operational phase only
    print("Generating operational phase plots...")
    
    # Plot and collect timing statistics
    timing_stats = plot_operational_timing(timing_df, system_date_time)
    if timing_stats:
        all_stats["timing"] = timing_stats
    
    # Plot and collect RTCM Age statistics
    rtcm_stats = plot_rtcm_correction_age(timing_df, system_date_time)
    if rtcm_stats:
        all_stats["rtcm_age"] = rtcm_stats
    
    # Save all statistics to file
    save_statistics(all_stats, stats_file)
    
    print(f"Analysis complete. Plots saved to {figures_folder}/ directory.")
    print(f"Statistics saved to {stats_file}")

if __name__ == "__main__":
    main()