# vocoder

`vocoder` is a small noise-vocoder project for loading speech audio,
splitting it into frequency bands, extracting envelopes, generating
noise-vocoded output, and plotting the result.

## Install
```bash
pip install git+https://git@github.com/martijnbentum/vocoder.git
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
    ├── core.py
    ├── plot.py
    └── signal_processing.py
```

## Examples
Create a vocoded file from Python:
```python
from vocoder.core import Vocoder

vocoder = Vocoder(filename='clear.wav', sample_rate=16000)
output_filename = vocoder.write_vocoded()
print(output_filename)
```

Use custom bands:
```python
import numpy as np

from vocoder.core import Vocoder

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
python -m vocoder --input_dir examples --nbands 4 --nprocess 2
```

For large directory runs, the CLI now:
- preserves the input subdirectory layout inside `--output_dir`
- prints compact progress lines instead of one verbose block per file
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
[`config/frequency_bands.json`](/Users/martijn.bentum/vocoder/repo/config/frequency_bands.json).
The CLI uses the `default_family` bands by default and also supports
`--frequency_family` and `--frequency_key`.

## Snellius Scripts
Snellius helper scripts live in [`scripts/`](/Users/martijn.bentum/vocoder/repo/scripts):
- `build_snellius_env.sh` creates the Python environment with the required
  module stack.
- `snellius_jobs.sh` only defines the named vocoding jobs and their output
  dirs when sourced by another script; running it directly does not submit or
  execute all jobs.
- `submit_vocode_job.sh <job_name>` resolves a named job, checks the target
  output directory, and submits the generic sbatch runner.
- `sbatch_vocode_job.sh` is the generic 64-core Slurm entrypoint that loads
  the named job configuration inside the job.
- Slurm stdout/stderr files are written to
  [`slurm_out/`](/Users/martijn.bentum/vocoder/repo/slurm_out).

Example submissions:
```bash
# Submit a named job through the generic submit path.
./scripts/submit_vocode_job.sh default_4_band
./scripts/submit_vocode_job.sh default_6_band
./scripts/submit_vocode_job.sh default_8_band
./scripts/submit_vocode_job.sh default_16_band
./scripts/submit_vocode_job.sh speech_weighted_8_band
```

The Snellius runner uses unbuffered Python output so progress appears in the
Slurm `.out` file during the job.
