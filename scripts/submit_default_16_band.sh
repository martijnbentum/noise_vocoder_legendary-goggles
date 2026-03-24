#!/bin/bash
set -eu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output_dir=/scratch-shared/mbentum1/vocoded_bands-16_spidr/wav/
"$script_dir/submit_snellius_job.sh" \
    "$script_dir/sbatch_default_16_band.sh" \
    "$output_dir"
