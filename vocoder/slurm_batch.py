import argparse
import json
import math
import os
import shutil
import time
from itertools import islice
from pathlib import Path

import soundfile as sf

from . import batch
from .file_io import (
    DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
    append_metadata,
    get_output_filename,
    make_output_shard_name,
)
from .vocoder import get_standard_bands

RUN_DIRNAME = '_vocode_run'
DEFAULT_FILES_PER_CHUNK = 500
DEFAULT_MAX_PARALLEL_CHUNKS = 64
DEFAULT_PROGRESS_EVERY = 25
CONFIG_KEYS = {
    'files_per_chunk',
    'frequencies',
    'input_dir',
    'match_rms',
    'max_parallel_chunks',
    'nbands',
    'output_dir',
    'overwrite',
    'sample_rate',
    'frequency_family',
}


def get_run_dir(output_dir):
    '''Return the metadata directory for one batch output dir.'''
    output_path = Path(output_dir)
    return output_path.parent / RUN_DIRNAME


def get_run_paths(output_dir):
    '''Return the standard metadata paths for one batch run.'''
    run_dir = get_run_dir(output_dir)
    return {
        'run_dir': run_dir,
        'run_config': run_dir / 'run_config.json',
        'manifest': run_dir / 'manifest.txt',
        'summary': run_dir / 'summary.json',
        'progress_dir': run_dir / 'progress',
        'failure_dir': run_dir / 'failures',
        'audio_info_dir': run_dir / 'audio_info',
        'merged_failures': run_dir / 'failures.jsonl',
        'merged_audio_info': run_dir / 'audio_info.jsonl',
    }


def load_batch_config(config_path):
    '''Load a batch config from JSON.'''
    with Path(config_path).open() as fin:
        return json.load(fin)


def normalize_batch_config(config, config_path = ''):
    '''Return a validated batch config with defaults applied.'''
    unknown_keys = set(config) - CONFIG_KEYS
    if unknown_keys:
        unknown_list = ', '.join(sorted(unknown_keys))
        raise ValueError(f'Unknown config keys: {unknown_list}')
    if 'input_dir' not in config or 'output_dir' not in config:
        raise ValueError('Config must contain input_dir and output_dir')
    normalized = {
        'config_path': str(config_path),
        'input_dir': str(config['input_dir']),
        'output_dir': str(config['output_dir']),
        'files_per_chunk': int(
            config.get('files_per_chunk', DEFAULT_FILES_PER_CHUNK)
        ),
        'max_parallel_chunks': int(
            config.get(
                'max_parallel_chunks',
                DEFAULT_MAX_PARALLEL_CHUNKS,
            )
        ),
        'sample_rate': int(config.get('sample_rate', 16000)),
        'match_rms': bool(config.get('match_rms', False)),
        'overwrite': bool(config.get('overwrite', False)),
        'frequency_family': config.get('frequency_family', 'default_family'),
        'nbands': int(config.get('nbands', 6)),
    }
    if normalized['files_per_chunk'] < 1:
        raise ValueError('files_per_chunk must be at least 1')
    if normalized['max_parallel_chunks'] < 1:
        raise ValueError('max_parallel_chunks must be at least 1')
    frequencies = config.get('frequencies')
    if frequencies is None:
        frequencies = get_standard_bands(
            n_bands=normalized['nbands'],
            family=normalized['frequency_family'],
        ).tolist()
    if len(frequencies) < 2:
        raise ValueError('frequencies must contain at least two values')
    normalized['frequencies'] = [int(value) for value in frequencies]
    normalized['n_bands'] = len(normalized['frequencies']) - 1
    return normalized


def sanitize_runtime_config(config):
    '''Drop transient fields before writing run_config.json.'''
    return {
        key: value
        for key, value in config.items()
        if key != 'overwrite'
    }


def configs_match(left, right):
    '''Return whether two normalized configs are compatible.'''
    return sanitize_runtime_config(left) == sanitize_runtime_config(right)


def build_manifest(input_dir, manifest_path):
    '''Write a sorted wav manifest and return the file count.'''
    filenames = sorted(Path(input_dir).rglob('*.wav'))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w') as fout:
        for filename in filenames:
            fout.write(f'{filename}\n')
    return len(filenames)


def cleanup_previous_run(config):
    '''Delete output wavs and run metadata for overwrite mode.'''
    output_dir = Path(config['output_dir'])
    if output_dir.exists():
        for wav_file in output_dir.rglob('*.wav'):
            wav_file.unlink()
        for chunk_dir in sorted(output_dir.glob('chunk_*'), reverse=True):
            if chunk_dir.is_dir():
                shutil.rmtree(chunk_dir)
    run_dir = get_run_dir(config['output_dir'])
    if run_dir.exists():
        shutil.rmtree(run_dir)


def read_manifest_count(manifest_path):
    '''Return the number of files in a manifest.'''
    with Path(manifest_path).open() as fin:
        return sum(1 for _ in fin)


def write_run_config(run_config_path, config):
    '''Write the normalized run config.'''
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    with run_config_path.open('w') as fout:
        json.dump(config, fout, indent=2, sort_keys=True)
        fout.write('\n')


def load_run_config(run_config_path):
    '''Load the saved run config for a batch.'''
    with Path(run_config_path).open() as fin:
        return json.load(fin)


def prepare_run(config_path):
    '''Prepare one batch run and return the saved run config.'''
    config = normalize_batch_config(load_batch_config(config_path), config_path)
    paths = get_run_paths(config['output_dir'])
    if config['overwrite']:
        cleanup_previous_run(config)
    if paths['run_config'].exists():
        existing = load_run_config(paths['run_config'])
        if not configs_match(existing, config):
            raise ValueError(
                'Existing run_config.json does not match the requested config. '
                'Use overwrite to start a fresh run.'
            )
    total_files = 0
    if paths['manifest'].exists() and not config['overwrite']:
        total_files = read_manifest_count(paths['manifest'])
    else:
        total_files = build_manifest(config['input_dir'], paths['manifest'])
    if total_files < 1:
        raise ValueError('No wav files found in input_dir')
    prepared = dict(sanitize_runtime_config(config))
    prepared['run_dir'] = str(paths['run_dir'])
    prepared['manifest_path'] = str(paths['manifest'])
    prepared['total_files'] = int(total_files)
    prepared['n_chunks'] = int(
        math.ceil(total_files / prepared['files_per_chunk'])
    )
    write_run_config(paths['run_config'], prepared)
    paths['progress_dir'].mkdir(parents=True, exist_ok=True)
    paths['failure_dir'].mkdir(parents=True, exist_ok=True)
    paths['audio_info_dir'].mkdir(parents=True, exist_ok=True)
    return prepared


def get_chunk_bounds(task_id, files_per_chunk, total_files):
    '''Return manifest bounds for one chunk index.'''
    start = task_id * files_per_chunk
    end = min(total_files, start + files_per_chunk)
    return start, end


def iter_manifest_slice(manifest_path, start, end):
    '''Yield one indexed slice from a manifest file.'''
    with Path(manifest_path).open() as fin:
        for index, line in enumerate(islice(fin, start, end), start=start):
            yield index, line.strip()


def get_chunk_progress_path(run_dir, task_id):
    '''Return the progress file path for one array task.'''
    return Path(run_dir) / 'progress' / f'chunk_{task_id:05d}.json'


def get_chunk_failure_path(run_dir, task_id):
    '''Return the failure log path for one array task.'''
    return Path(run_dir) / 'failures' / f'failures_{task_id:05d}.jsonl'


def get_chunk_audio_info_path(run_dir, task_id):
    '''Return the input-audio log path for one array task.'''
    return Path(run_dir) / 'audio_info' / f'audio_{task_id:05d}.jsonl'


def write_chunk_progress(progress_path, payload):
    '''Write one chunk progress file atomically.'''
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = progress_path.with_suffix('.tmp')
    with temp_path.open('w') as fout:
        json.dump(payload, fout, indent=2, sort_keys=True)
        fout.write('\n')
    temp_path.replace(progress_path)


def is_valid_output_file(filename):
    '''Return whether an existing output file looks usable.'''
    path = Path(filename)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        sf.info(path)
    except RuntimeError:
        return False
    return True


def get_output_shard_dir(global_index, max_files_per_output_dir):
    '''Return the shard dir name for one global manifest index.'''
    if max_files_per_output_dir < 1:
        return ''
    shard_index = global_index // max_files_per_output_dir
    return make_output_shard_name(shard_index)


def make_chunk_args(config, filename, output_shard_dir):
    '''Build one args namespace for handle_filename.'''
    return argparse.Namespace(
        filename=filename,
        sample_rate=config['sample_rate'],
        butterworth_order=4,
        match_rms=config['match_rms'],
        frequencies=config['frequencies'],
        output_dir=config['output_dir'],
        input_dir=config['input_dir'],
        output_shard_dir=output_shard_dir,
    )


def make_audio_info_record(filename):
    '''Return a compact input-audio record for one source file.'''
    path = Path(filename)
    audio_info = sf.info(path)
    return {
        'input_filename': str(path),
        'file_stem': path.stem,
        'n_samples': int(audio_info.frames),
    }


def print_chunk_progress(task_id, stats, elapsed_seconds):
    '''Print a compact progress line with ETA for one chunk.'''
    done = stats['processed'] + stats['skipped'] + stats['failed']
    assigned = stats['assigned']
    rate = done / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = assigned - done
    eta_seconds = remaining / rate if rate > 0 else 0.0
    print(
        f'chunk={task_id}',
        f'processed={done}/{assigned}',
        f'created={stats["processed"]}',
        f'skipped={stats["skipped"]}',
        f'failed={stats["failed"]}',
        f'rate={rate:.2f} files/s',
        f'eta={eta_seconds:.0f}s',
        flush=True,
    )


def process_chunk(config_path, task_id):
    '''Process one manifest chunk and write per-task logs.'''
    config = load_run_config(
        get_run_paths_from_config_path(config_path)['run_config']
    )
    start, end = get_chunk_bounds(
        task_id,
        config['files_per_chunk'],
        config['total_files'],
    )
    progress_path = get_chunk_progress_path(config['run_dir'], task_id)
    failure_path = get_chunk_failure_path(config['run_dir'], task_id)
    audio_info_path = get_chunk_audio_info_path(config['run_dir'], task_id)
    if progress_path.exists():
        progress_path.unlink()
    if failure_path.exists():
        failure_path.unlink()
    if audio_info_path.exists():
        audio_info_path.unlink()
    stats = {
        'task_id': int(task_id),
        'assigned': max(0, end - start),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'status': 'running',
    }
    started_at = time.time()
    if stats['assigned'] == 0:
        stats['status'] = 'done'
        stats['elapsed_seconds'] = 0.0
        write_chunk_progress(progress_path, stats)
        print(f'chunk={task_id} assigned=0', flush=True)
        return stats
    for local_index, (global_index, filename) in enumerate(
        iter_manifest_slice(config['manifest_path'], start, end),
        start=1,
    ):
        output_shard_dir = get_output_shard_dir(
            global_index,
            DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
        )
        output_filename = get_output_filename(
            filename,
            output_dir=config['output_dir'],
            input_dir=config['input_dir'],
            output_shard_dir=output_shard_dir,
            n_bands=config['n_bands'],
        )
        try:
            append_metadata(
                audio_info_path,
                make_audio_info_record(filename),
            )
        except RuntimeError:
            pass
        if is_valid_output_file(output_filename):
            stats['skipped'] += 1
        else:
            args = make_chunk_args(config, filename, output_shard_dir)
            try:
                batch.handle_filename(args)
                stats['processed'] += 1
            except Exception as exc:
                stats['failed'] += 1
                append_metadata(
                    failure_path,
                    batch.make_failure_result(
                        filename,
                        time.time() - started_at,
                        os.getpid(),
                        exc,
                    ),
                )
        if (
            local_index % DEFAULT_PROGRESS_EVERY == 0
            or local_index == stats['assigned']
        ):
            elapsed_seconds = time.time() - started_at
            stats['elapsed_seconds'] = float(round(elapsed_seconds, 4))
            write_chunk_progress(progress_path, stats)
            print_chunk_progress(task_id, stats, elapsed_seconds)
    elapsed_seconds = time.time() - started_at
    stats['status'] = 'done'
    stats['elapsed_seconds'] = float(round(elapsed_seconds, 4))
    write_chunk_progress(progress_path, stats)
    print(
        f'chunk={task_id} done',
        f'assigned={stats["assigned"]}',
        f'created={stats["processed"]}',
        f'skipped={stats["skipped"]}',
        f'failed={stats["failed"]}',
        f'elapsed={elapsed_seconds:.1f}s',
        flush=True,
    )
    return stats


def get_run_paths_from_config_path(config_path):
    '''Return run paths for a config file path.'''
    config = normalize_batch_config(load_batch_config(config_path), config_path)
    return get_run_paths(config['output_dir'])


def merge_failures(config):
    '''Merge per-task failure logs into one failure log.'''
    failure_dir = Path(config['run_dir']) / 'failures'
    merged_path = Path(config['run_dir']) / 'failures.jsonl'
    with merged_path.open('w') as fout:
        for failure_path in sorted(failure_dir.glob('failures_*.jsonl')):
            fout.write(failure_path.read_text())
    return merged_path


def merge_audio_info(config):
    '''Merge per-task input-audio logs into one metadata file.'''
    audio_info_dir = Path(config['run_dir']) / 'audio_info'
    merged_path = Path(config['run_dir']) / 'audio_info.jsonl'
    with merged_path.open('w') as fout:
        for audio_info_path in sorted(audio_info_dir.glob('audio_*.jsonl')):
            fout.write(audio_info_path.read_text())
    return merged_path


def count_valid_outputs(config):
    '''Count valid output files referenced by the manifest.'''
    completed = 0
    for global_index, filename in iter_manifest_slice(
        config['manifest_path'],
        0,
        config['total_files'],
    ):
        output_shard_dir = get_output_shard_dir(
            global_index,
            DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
        )
        output_filename = get_output_filename(
            filename,
            output_dir=config['output_dir'],
            input_dir=config['input_dir'],
            output_shard_dir=output_shard_dir,
            n_bands=config['n_bands'],
        )
        if is_valid_output_file(output_filename):
            completed += 1
    return completed


def finalize_run(config_path):
    '''Merge logs and write a final summary for one batch run.'''
    config = load_run_config(
        get_run_paths_from_config_path(config_path)['run_config']
    )
    progress_dir = Path(config['run_dir']) / 'progress'
    chunk_progress = []
    for progress_path in sorted(progress_dir.glob('chunk_*.json')):
        with progress_path.open() as fin:
            chunk_progress.append(json.load(fin))
    merged_failures = merge_failures(config)
    merged_audio_info = merge_audio_info(config)
    completed_outputs = count_valid_outputs(config)
    failed_files = sum(item['failed'] for item in chunk_progress)
    skipped_files = sum(item['skipped'] for item in chunk_progress)
    created_files = sum(item['processed'] for item in chunk_progress)
    failed_chunks = config['n_chunks'] - len(chunk_progress)
    summary = {
        'total_files': config['total_files'],
        'n_chunks': config['n_chunks'],
        'completed_outputs': completed_outputs,
        'created_files': created_files,
        'skipped_files': skipped_files,
        'failed_files': failed_files,
        'missing_outputs': config['total_files'] - completed_outputs,
        'failed_chunks': failed_chunks,
        'merged_failures': str(merged_failures),
        'merged_audio_info': str(merged_audio_info),
    }
    summary_path = Path(config['run_dir']) / 'summary.json'
    with summary_path.open('w') as fout:
        json.dump(summary, fout, indent=2, sort_keys=True)
        fout.write('\n')
    print('run summary:', flush=True)
    for key, value in summary.items():
        print(f'{key}: {value}', flush=True)
    return summary


def build_cli():
    parser = argparse.ArgumentParser(description='Slurm batch helper')
    subparsers = parser.add_subparsers(dest='command', required=True)

    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('config_path')

    chunk_parser = subparsers.add_parser('chunk')
    chunk_parser.add_argument('config_path')
    chunk_parser.add_argument('task_id', type=int)

    finalize_parser = subparsers.add_parser('finalize')
    finalize_parser.add_argument('config_path')
    return parser


def main():
    args = build_cli().parse_args()
    if args.command == 'prepare':
        prepared = prepare_run(args.config_path)
        print(
            prepared['n_chunks'],
            prepared['max_parallel_chunks'],
            prepared['run_dir'],
        )
        return
    if args.command == 'chunk':
        process_chunk(args.config_path, args.task_id)
        return
    finalize_run(args.config_path)


if __name__ == '__main__':
    main()
