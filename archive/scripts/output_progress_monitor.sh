#!/bin/bash
# Poll an output directory and write a compact progress file.

set -eu

if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
    echo "Usage: $0 <output_dir> <progress_file> <baseline_count> <total_files> [interval_seconds] [stall_seconds]" >&2
    exit 1
fi

output_dir="$1"
progress_file="$2"
baseline_count="$3"
total_files="$4"
interval_seconds="${5:-30}"
stall_seconds="${6:-360}"
full_scan_interval_seconds=360

mkdir -p "$(dirname "$progress_file")"

current_count=0
last_full_scan_epoch=0

count_wav_files() {
    find "$output_dir" -type f -name '*.wav' | wc -l | tr -d ' '
}

refresh_wav_count_if_due() {
    now_epoch=$(date '+%s')
    if [ "$last_full_scan_epoch" -eq 0 ] || \
        [ $(( now_epoch - last_full_scan_epoch )) -ge "$full_scan_interval_seconds" ]; then
        current_count=$(count_wav_files)
        last_full_scan_epoch="$now_epoch"
    fi
}

get_dir_mtime() {
    if stat -f '%m' "$1" >/dev/null 2>&1; then
        stat -f '%m' "$1"
        return
    fi
    stat -c '%Y' "$1"
}

collect_chunk_dir_stats() {
    chunk_dir_count=0
    latest_chunk_change_epoch=0
    for chunk_dir in "$output_dir"/chunk_*; do
        if [ ! -d "$chunk_dir" ]; then
            continue
        fi
        chunk_dir_count=$(( chunk_dir_count + 1 ))
        chunk_mtime=$(get_dir_mtime "$chunk_dir")
        if [ "$chunk_mtime" -gt "$latest_chunk_change_epoch" ]; then
            latest_chunk_change_epoch="$chunk_mtime"
        fi
    done
}

compute_progress_fields() {
    refresh_wav_count_if_due
    collect_chunk_dir_stats
    completed_count=$(( current_count - baseline_count ))
    if [ "$completed_count" -lt 0 ]; then
        completed_count=0
    fi
    if [ "$completed_count" -gt "$total_files" ]; then
        completed_count="$total_files"
    fi
    now_epoch=$(date '+%s')
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    seconds_since_last_change='unknown'
    if [ "$latest_chunk_change_epoch" -gt 0 ]; then
        seconds_since_last_change=$(( now_epoch - latest_chunk_change_epoch ))
    fi
    progress_status='running'
    if [ "$total_files" -gt 0 ] && [ "$completed_count" -ge "$total_files" ]; then
        progress_status='done'
    elif [ "$seconds_since_last_change" != 'unknown' ] && \
        [ "$seconds_since_last_change" -gt "$stall_seconds" ]; then
        progress_status='stalled'
    fi
}

write_progress() {
    compute_progress_fields
    tmp_file="${progress_file}.tmp"
    {
        echo "status: $progress_status"
        echo "timestamp: $timestamp"
        echo "output_dir: $output_dir"
        echo "wav_files_present: $current_count"
        echo "wav_files_baseline: $baseline_count"
        echo "wav_files_written: $completed_count"
        echo "chunk_dir_count: $chunk_dir_count"
        echo "last_wav_change_epoch: $latest_chunk_change_epoch"
        echo "seconds_since_last_wav_change: $seconds_since_last_change"
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
    current_count=$(count_wav_files)
    last_full_scan_epoch=0
    compute_progress_fields
    tmp_file="${progress_file}.tmp"
    {
        echo "status: $progress_status"
        echo "timestamp: $timestamp"
        echo "output_dir: $output_dir"
        echo "wav_files_present: $current_count"
        echo "wav_files_baseline: $baseline_count"
        echo "wav_files_written: $completed_count"
        echo "chunk_dir_count: $chunk_dir_count"
        echo "last_wav_change_epoch: $latest_chunk_change_epoch"
        echo "seconds_since_last_wav_change: $seconds_since_last_change"
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
