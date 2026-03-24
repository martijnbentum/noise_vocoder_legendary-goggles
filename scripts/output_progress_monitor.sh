#!/bin/bash
# Poll an output directory infrequently and write a compact progress file.

set -eu

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <output_dir> <progress_file> <baseline_count> <total_files> [interval_seconds]" >&2
    exit 1
fi

output_dir="$1"
progress_file="$2"
baseline_count="$3"
total_files="$4"
interval_seconds="${5:-180}"

mkdir -p "$(dirname "$progress_file")"

write_progress() {
    current_count=$(find "$output_dir" -type f -name '*.wav' | wc -l | tr -d ' ')
    completed_count=$(( current_count - baseline_count ))
    if [ "$completed_count" -lt 0 ]; then
        completed_count=0
    fi
    if [ "$completed_count" -gt "$total_files" ]; then
        completed_count="$total_files"
    fi
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    tmp_file="${progress_file}.tmp"
    {
        echo "status: running"
        echo "timestamp: $timestamp"
        echo "output_dir: $output_dir"
        echo "wav_files_present: $current_count"
        echo "wav_files_baseline: $baseline_count"
        echo "wav_files_written: $completed_count"
        echo "total_expected: $total_files"
        if [ "$total_files" -gt 0 ]; then
            percentage=$(( completed_count * 100 / total_files ))
            echo "percent_complete: ${percentage}%"
            echo "remaining: $(( total_files - completed_count ))"
        fi
    } > "$tmp_file"
    mv "$tmp_file" "$progress_file"
}

write_final_progress() {
    current_count=$(find "$output_dir" -type f -name '*.wav' | wc -l | tr -d ' ')
    completed_count=$(( current_count - baseline_count ))
    if [ "$completed_count" -lt 0 ]; then
        completed_count=0
    fi
    if [ "$completed_count" -gt "$total_files" ]; then
        completed_count="$total_files"
    fi
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    tmp_file="${progress_file}.tmp"
    {
        echo "status: done"
        echo "timestamp: $timestamp"
        echo "output_dir: $output_dir"
        echo "wav_files_present: $current_count"
        echo "wav_files_baseline: $baseline_count"
        echo "wav_files_written: $completed_count"
        echo "total_expected: $total_files"
        if [ "$total_files" -gt 0 ]; then
            percentage=$(( completed_count * 100 / total_files ))
            echo "percent_complete: ${percentage}%"
            echo "remaining: $(( total_files - completed_count ))"
        fi
    } > "$tmp_file"
    mv "$tmp_file" "$progress_file"
}

trap 'write_final_progress; exit 0' TERM INT

while true; do
    write_progress
    sleep "$interval_seconds"
done
