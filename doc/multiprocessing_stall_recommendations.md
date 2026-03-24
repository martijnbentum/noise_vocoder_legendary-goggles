# Multiprocessing Stall Recommendations

This note captures the main ways the Snellius batch vocoder can appear
stuck while files remain, plus the most practical mitigations.

## Main Risk

The current batch pipeline can still wait indefinitely if a worker gets
stuck on one file. Typical causes include:

- a blocking read in `librosa.load`
- a blocking write in `soundfile.write`
- a filesystem stall on shared storage
- numerical work that hangs or becomes pathologically slow on one input

Because the parent process consumes results from the pool, one wedged
worker can make the job look idle for a long time even when Slurm still
shows it as running.

## Recommended Changes

1. Reduce pool chunk size for robustness.

Large chunks improve throughput in the happy path but make stalls harder
to localize and can delay worker recycling. A smaller chunksize such as
`8` or `16` is a better default for large shared-cluster runs.

2. Add a watchdog for time since the last completed result.

The parent process should track the timestamp of the most recent result.
If no result arrives within a threshold such as 10 to 20 minutes, the
job should:

- write a clear failure record
- terminate the worker pool
- exit non-zero

3. Record the current file per worker.

Before starting a file, each worker should update a small heartbeat or
status file with:

- worker PID
- input filename
- start timestamp

If a job stalls, this makes it possible to identify the blocking file
without attaching a debugger.

4. Add per-file timeout isolation for maximum robustness.

The strongest fix is to isolate each file in a short-lived subprocess
with a timeout. This adds overhead, but it prevents one pathological file
from wedging a long-running worker forever.

5. Consider a lower `maxtasksperchild` once chunking is reduced.

Restarting workers more frequently can help recover from memory growth or
native-library state issues. This matters most after reducing chunk size.

## Operational Guidance

- If the job appears stuck, inspect `archive/progress_<jobid>.txt`.
- If `processed_files` stops changing while Slurm still shows active
  CPUs, suspect a hung worker rather than a parent-process crash.
- If this pattern repeats, prioritize a watchdog and per-worker
  heartbeat logging before pursuing further throughput tuning.
