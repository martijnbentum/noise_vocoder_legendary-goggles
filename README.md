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
├── scripts
├── tests
│   └── tester.py
└── vocoder
    ├── __init__.py
    ├── audio.py
    ├── batch.py
    ├── file_io.py
    ├── plot.py
    ├── signal_processing.py
    ├── slurm_batch.py
    └── vocoder.py
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
The active Snellius entrypoints live in
[`scripts/`](/Users/martijn.bentum/vocoder/repo/scripts):
- `build_snellius_env.sh` creates the Python environment with the required
  module stack.
- `batch_vocode.sbatch` is the config-driven Slurm entrypoint for chunked
  array runs.

The array workflow is:
1. `sbatch [scripts/batch_vocode.sbatch](/Users/martijn.bentum/vocoder/repo/scripts/batch_vocode.sbatch) config.json`
2. The bootstrap job prepares `_vocode_run/` next to the output `wav/`
   directory and writes `manifest.txt` plus `run_config.json`.
3. The bootstrap job submits the real task-group array and a finalizer job.
4. Each array task gets `cpus_per_task` CPUs and runs that many Python chunk
   workers in parallel inside the task.
5. Each chunk processes one manifest slice, skips valid existing outputs,
   writes per-chunk progress/failure/input-audio logs, and prints local ETA
   lines.
6. The finalizer merges those logs and writes `summary.json`.

Dry run:
```bash
python -m vocoder.slurm_batch dry_run config/legacy_4_bands.json
```
This uses the existing manifest when present, otherwise it creates one and
prints the planned task groups, chunk ranges, total chunks, and files per
group without submitting any Slurm jobs.

Example submissions:
```bash
sbatch scripts/batch_vocode.sbatch config/speech_weighted_8_bands.json
sbatch scripts/batch_vocode.sbatch config/legacy_4_bands.json
```

The Snellius runner uses unbuffered Python output for startup, chunk-group
progress, failures, and a final run summary. The launcher job writes its
stdout/stderr to `slurm_out/launcher/`, while chunk and finalizer jobs write
to `slurm_out/`.
