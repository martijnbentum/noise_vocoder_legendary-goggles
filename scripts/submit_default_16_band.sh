#!/bin/bash
set -eu

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$script_dir/submit_snellius_job.sh" \
    "$script_dir/sbatch_default_16_band.sh" \
    /projects/0/prjs1489/data/spidr/vocoded_bands-16_spidr/wav
