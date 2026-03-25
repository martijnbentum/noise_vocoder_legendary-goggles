#!/bin/bash
set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <target_dir>" >&2
    exit 1
fi

source_dir="$1"
target_dir="$2"

if [ ! -d "$source_dir" ]; then
    echo "Source directory does not exist: $source_dir" >&2
    exit 1
fi

if [ ! -d "$target_dir" ]; then
    echo "Target directory does not exist: $target_dir" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
archive_dir="$repo_root/archive"
mkdir -p "$archive_dir"

timestamp=$(date +"%Y%m%d_%H%M%S")
output_file="$archive_dir/missing_wavs_${timestamp}.txt"

while IFS= read -r source_file; do
    rel_path=${source_file#"$source_dir"/}
    if [ ! -f "$target_dir/$rel_path" ]; then
        printf '%s\n' "$rel_path" >> "$output_file"
    fi
done < <(find "$source_dir" -type f -name '*.wav' | sort)

if [ ! -f "$output_file" ]; then
    : > "$output_file"
fi

echo "$output_file"
