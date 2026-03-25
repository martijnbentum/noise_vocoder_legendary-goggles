# vocoder

`vocoder` is a small noise-vocoder project for loading speech audio,
splitting it into frequency bands, extracting envelopes, generating
noise-vocoded output, and plotting the result.

## Install
```bash
pip install git+https://git@github.com/martijnbentum/noise_vocoder_legendary-goggles.git
```

## Layout
```text
.
├── AGENTS.md
├── LICENSE
├── pyproject.toml
├── README.md
├── tests
│   └── tester.py
└── vocoder
    ├── __init__.py
    ├── audio.py
    ├── batch.py
    ├── core.py
    ├── file_io.py
    ├── plot.py
    └── signal_processing.py
```

## Examples
Create a vocoded file from Python:
```python
from vocoder.vocoder import Vocoder

vocoder = Vocoder(filename='clear.wav', sample_rate=16000)
output_filename = vocoder.write_vocoded()
print(output_filename)
```

Use custom bands:
```python
import numpy as np

from vocoder.vocoder import Vocoder

frequencies = np.array([100, 500, 1500, 5000])
vocoder = Vocoder(
    filename='clear.wav',
    sample_rate=16000,
    frequencies=frequencies,
)
vocoder.plot_compare_spectrogram()
```

Run the CLI on one file:
```bash
python -m vocoder --filename clear.wav --nbands 6
```

Run the CLI on a directory:
```bash
python -m vocoder --input_dir examples --nbands 4
```

For large directory runs, the CLI now:
- runs batch jobs sequentially
- can optionally write per-file records to a JSONL metadata file with
  `--metadata_filename batch.jsonl`

## Dependencies
The code currently depends on these Python packages:
- `librosa`
- `matplotlib`
- `numpy`
- `scipy`
- `sounddevice`
- `soundfile`

## Git Hook
This repo includes a tracked `pre-commit` hook in `.githooks/pre-commit`
that bumps the patch version in `pyproject.toml` on every commit.

## Frequency Config
Standard frequency bands now live in
[`vocoder/frequency_bands.json`](/Users/martijn.bentum/vocoder/repo/vocoder/frequency_bands.json).
The CLI uses the `default_family` bands by default and also supports
`--frequency_family` and `--frequency_key`.
The `default_family` presets are approximately logarithmic and follow the
classic noise-vocoding setup described on Matt Davis's vocoder page and in
Shannon et al., "Speech Recognition with Primarily Temporal Cues"
(*Science*, 1995).

The default config file is bundled inside the installed `vocoder` package, so
`get_standard_bands()` works after `pip install`. If you saw a
`FileNotFoundError` pointing at `site-packages/config/frequency_bands.json`,
that came from an older release that still looked for a repo-level config dir.

## Snellius Scripts
Snellius helper scripts live in
[`archive/scripts/`](/Users/martijn.bentum/vocoder/repo/archive/scripts):
- `build_snellius_env.sh` creates the Python environment with the required
  module stack.
- `snellius_jobs.sh` only defines the named vocoding jobs and their output
  dirs when sourced by another script; running it directly does not submit or
  execute all jobs.
- `submit_vocode_job.sh <job_name>` resolves a named job, checks the target
  output directory, and submits the generic sbatch runner.
- `sbatch_vocode_job.sh` is the generic 64-core Slurm entrypoint that loads
  the named job configuration inside the job.
- `submit_repair_vocode_job.sh <job_name>` submits a repair run for a named
  job and reruns only source wavs whose expected outputs are still missing.
- `find_missing_vocoded_wavs.sh <job_name>` writes the missing source wav
  paths to `archive/missing_<job_name>_<slurm_job_id>.txt`.
- Slurm stdout/stderr files are written to
  [`slurm_out/`](/Users/martijn.bentum/vocoder/repo/slurm_out).
- Batch progress is written to `archive/progress_<slurm_job_id>.txt` by the
  shell-side monitor, while active file-level worker status is written by
  Python under `_worker_status/` in the output directory.
- If a running job stalls and `seconds_since_last_wav_change` exceeds 360
  while progress is still below `100%`, the main job submits one repair job
  automatically and then stops itself.

Example submissions:
```bash
# Show all available named jobs.
./archive/scripts/submit_vocode_job.sh --list

# Submit a named job through the generic submit path.
./archive/scripts/submit_vocode_job.sh default_4_band
./archive/scripts/submit_vocode_job.sh default_6_band
./archive/scripts/submit_vocode_job.sh default_8_band
./archive/scripts/submit_vocode_job.sh default_16_band
./archive/scripts/submit_vocode_job.sh speech_weighted_8_band
```

Example repair commands:
```bash
# Submit a repair job for one named batch.
./archive/scripts/submit_repair_vocode_job.sh default_4_band

# Build the missing-file list explicitly for inspection.
./archive/scripts/find_missing_vocoded_wavs.sh default_4_band

# Run the repair flow directly inside an allocation.
./archive/scripts/run_repair_vocode.sh default_4_band
```

The Snellius runner uses unbuffered Python output for startup, failures, and a
final batch summary.
