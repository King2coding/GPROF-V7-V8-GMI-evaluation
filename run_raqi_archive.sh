#!/usr/bin/env bash

set -u

START_DATE="2021-01-01"
END_DATE="2025-12-31"

SCRIPT="/home/kkumah/Projects/Comparing_V7_V8_GPROF-GMI_data/Codes/aggregate_mrms_raqi_hourly.py"

RQI_DIR="/scratch/kkumah/MRMS/RadarQualityIndex"
OUT_DIR="/scratch/kkumah/MRMS/RAQI"

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="${LOG_DIR}/raqi_archive_${START_DATE}_to_${END_DATE}.log"
FAILED_LOG="${LOG_DIR}/raqi_failed_dates_${START_DATE}_to_${END_DATE}.log"

touch "$MAIN_LOG" "$FAILED_LOG"

current_date="$START_DATE"

echo "RAQI archive processing started: $(date)" | tee -a "$MAIN_LOG"
echo "Period: $START_DATE to $END_DATE" | tee -a "$MAIN_LOG"

while [[ "$current_date" < "$END_DATE" || "$current_date" == "$END_DATE" ]]; do

    echo "==================================================" | tee -a "$MAIN_LOG"
    echo "Processing $current_date" | tee -a "$MAIN_LOG"

    python3 "$SCRIPT" \
        --rqi-dir "$RQI_DIR" \
        --out-dir "$OUT_DIR" \
        --date "$current_date" \
        --minimum-scans 20 \
        >> "$MAIN_LOG" 2>&1

    status=$?

    if [[ $status -eq 0 ]]; then
        echo "Completed or skipped: $current_date" | tee -a "$MAIN_LOG"
    else
        echo "$current_date,status=$status" | tee -a "$FAILED_LOG"
        echo "FAILED: $current_date, status=$status" | tee -a "$MAIN_LOG"
    fi

    current_date=$(date -I -d "$current_date + 1 day")

done

echo "RAQI archive processing finished: $(date)" | tee -a "$MAIN_LOG"
echo "Main log: $MAIN_LOG"
echo "Failed-date log: $FAILED_LOG"
