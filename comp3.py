import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

# Create output directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"comp_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Directory with log files
log_dir = "comp"

# Operations of interest (all in lowercase for matching)
ops_input = ["position-control latency", "checkcallbacks", "checkublox", "pushrawdata"]
categories = ["Position-Control Latency", "CheckCallbacks", "CheckUblox", "PushRawData"]
width = 0.25  # bar width

# Container for summaries: {identifier: {op: {avg, med, max}}}
summaries = {}

# Process each file in the directory
for filename in os.listdir(log_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(log_dir, filename)
        with open(filepath, "r") as file:
            lines = file.readlines()

        # Use the first non-empty line as the file's identifier
        label = next((line.strip() for line in lines if line.strip()), filename)
        
        # Extract the last session's block
        session_lines = []
        for i in range(len(lines)-1, -1, -1):
            # Stop when we hit a line with "Session"
            if "session" in lines[i].lower():
                break
            session_lines.insert(0, lines[i])
        
        # Clean up whitespace and ensure all comparisons are lowercase
        session_lines = [line.rstrip() for line in session_lines]

        # Initialize stats dict for each operation
        file_stats = {op: {"avg": np.nan, "med": np.nan, "max": np.nan} for op in ops_input}
        
        # For each operation, find its corresponding block and extract numbers.
        for op in ops_input:
            op_lower = op.lower()
            # search in session_lines
            for i, line in enumerate(session_lines):
                if op_lower in line.lower():
                    # Look a few lines ahead (up to 6) for Avg, Median, and Max
                    for j in range(1, 7):
                        if i + j < len(session_lines):
                            current_line = session_lines[i+j].strip()
                            # Check each field; allow for slight variations.
                            if current_line.lower().startswith("avg:"):
                                match_avg = re.search(r"([\d.]+)", current_line)
                                if match_avg:
                                    file_stats[op]["avg"] = float(match_avg.group(1))
                            elif current_line.lower().startswith("median:"):
                                match_med = re.search(r"([\d.]+)", current_line)
                                if match_med:
                                    file_stats[op]["med"] = float(match_med.group(1))
                            elif current_line.lower().startswith("max:"):
                                match_max = re.search(r"([\d.]+)", current_line)
                                if match_max:
                                    file_stats[op]["max"] = float(match_max.group(1))
                    break  # found the op, move to next op

        summaries[label] = file_stats

# Debug: print extracted summaries
for label, stats in summaries.items():
    print("File:", label)
    for op in ops_input:
        print(f"  {op}: {stats[op]}")
    print()

# Determine number of files
labels = list(summaries.keys())
n = len(labels)
x = np.arange(len(categories))

# Plot 1: MEDIAN with Average shown as I-bars
fig1, ax1 = plt.subplots(figsize=(12, 7))

for i, label in enumerate(labels):
    medians = [summaries[label][op]["med"] for op in ops_input]
    avgs = [summaries[label][op]["avg"] for op in ops_input]
    
    # Compute an offset for each file's bars
    offset = (i - n / 2) * width + width / 2
    
    # Plot median bars
    ax1.bar(x + offset, medians, width, label=label)
    
    # Plot average I-bars
    for j, avg in enumerate(avgs):
        if not np.isnan(avg):
            # Draw a horizontal line for the average with the same width as the bar
            ax1.plot([x[j] + offset - width/2, x[j] + offset + width/2], 
                     [avg, avg], 
                     color='black', 
                     linewidth=2)
            
            # Add small vertical lines at the ends to create an I-bar
            ax1.plot([x[j] + offset - width/2, x[j] + offset - width/2], 
                     [avg - width/10, avg + width/10], 
                     color='black', 
                     linewidth=2)
            ax1.plot([x[j] + offset + width/2, x[j] + offset + width/2], 
                     [avg - width/10, avg + width/10], 
                     color='black', 
                     linewidth=2)

# Add a custom legend entry for the average I-bars
from matplotlib.lines import Line2D
custom_line = Line2D([0], [0], color='black', lw=2, label='Average')
handles, labels = ax1.get_legend_handles_labels()
handles.append(custom_line)
ax1.legend(handles=handles)

ax1.set_ylabel("Time (ms)")
ax1.set_title("Timing by Operation")
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45, ha="right")
ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

# Save figure 1 to file
fig1.tight_layout()
fig1_path = os.path.join(output_dir, "median_timings.png")
fig1.savefig(fig1_path)
print(f"Saved median timing figure to {fig1_path}")

# Plot 2: MAX Timing Histogram with theoretical maximum
fig2, ax2 = plt.subplots(figsize=(12, 7))
for i, label in enumerate(labels):
    maxs = [summaries[label][op]["max"] for op in ops_input]
    offset = (i - n / 2) * width + width / 2
    ax2.bar(x + offset, maxs, width, label=label)

# Add theoretical maximum for position control latency
theoretical_max = 769  # ms
# Add a horizontal line for theoretical maximum at position control latency (first category)
ax2.plot([0 - width, 0 + width], [theoretical_max, theoretical_max], 
         color='red', linewidth=2, linestyle='--')
# Add text annotation
ax2.annotate('Calculated maximum (769 ms) for I2C\nbased on 2.6 m/s speed limit',
             xy=(0, theoretical_max),
             xytext=(0, theoretical_max + 50),  # Offset text 50 ms above the line
             color='red',
             fontsize=9,
             ha='center',
             arrowprops=dict(arrowstyle='->',
                            color='red'))

ax2.set_ylabel("Max Time (ms)")
ax2.set_title("Max Timing by Operation")
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=45, ha="right")
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
ax2.legend()

# Save figure 2 to file
fig2.tight_layout()
fig2_path = os.path.join(output_dir, "max_timings.png")
fig2.savefig(fig2_path)
print(f"Saved max timing figure to {fig2_path}")

# Show figures on screen as well
plt.show()