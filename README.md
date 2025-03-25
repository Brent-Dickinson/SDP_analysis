# Log Analysis Tool

A Python utility for analyzing ESP32 FreeRTOS log files and visualizing operation timing data.

## Usage

1. Place your log file in the main directory
2. Run the analysis script with your desired options:

```bash
python log_analysis_levels.py --log log.txt --ignore "RTCM age" "Websocket gnssMutex acq" "ntrip mutex acq" "ntripClientMutex hold (no data)" "gnssMutex hold" "gnssMutex take" --by-level