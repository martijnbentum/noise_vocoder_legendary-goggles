import argparse
import json
import math
import os
import shutil
from pathlib import Path

from . import batch
from .file_io import get_output_filename
from .vocoder import get_standard_bands

RUN_DIRNAME = '_vocode_run'
DEFAULT_CPUS_PER_TASK = 16
DEFAULT_FILES_PER_CHUNK = 500
DEFAULT_MAX_PARALLEL_TASKS = 4
CONFIG_KEYS = {
    'files_per_chunk',
    'frequencies',
    'input_dir',
    'match_rms',
    'max_parallel_tasks',
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
        'max_parallel_tasks': int(
            config.get(
                'max_parallel_tasks',
                DEFAULT_MAX_PARALLEL_TASKS,
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
    if normalized['max_parallel_tasks'] < 1:
        raise ValueError('max_parallel_tasks must be at least 1')
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
    if paths['manifest'].exists() and not config['overwrite']:
        total_files = batch.read_manifest_count(paths['manifest'])
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
    prepared['cpus_per_task'] = DEFAULT_CPUS_PER_TASK
    prepared['n_task_groups'] = int(
        math.ceil(prepared['n_chunks'] / DEFAULT_CPUS_PER_TASK)
    )
    write_run_config(paths['run_config'], prepared)
    batch.ensure_run_directories(prepared)
    return prepared


def get_run_paths_from_config_path(config_path):
    '''Return run paths for a config file path.'''
    config = normalize_batch_config(load_batch_config(config_path), config_path)
    return get_run_paths(config['output_dir'])


def process_group(config_path, group_id):
    '''Process one Slurm task group of chunk ids.'''
    config = load_run_config(
        get_run_paths_from_config_path(config_path)['run_config']
    )
    chunk_ids = batch.get_chunk_ids_for_group(
        group_id,
        DEFAULT_CPUS_PER_TASK,
        config['n_chunks'],
    )
    nprocess = int(
        os.environ.get('SLURM_CPUS_PER_TASK', DEFAULT_CPUS_PER_TASK)
    )
    return batch.process_manifest_chunks_parallel(
        config,
        chunk_ids,
        nprocess=nprocess,
    )


def dry_run(config_path):
    '''Print the grouped chunk layout without submitting Slurm jobs.'''
    config = prepare_run(config_path)
    print('dry run:', flush=True)
    print(f'run_dir: {config["run_dir"]}', flush=True)
    print(f'manifest_path: {config["manifest_path"]}', flush=True)
    print(f'total_files: {config["total_files"]}', flush=True)
    print(f'files_per_chunk: {config["files_per_chunk"]}', flush=True)
    print(f'n_chunks: {config["n_chunks"]}', flush=True)
    print(f'cpus_per_task: {DEFAULT_CPUS_PER_TASK}', flush=True)
    print(f'n_task_groups: {config["n_task_groups"]}', flush=True)
    print(
        f'max_parallel_tasks: {config["max_parallel_tasks"]}',
        flush=True,
    )
    for group_id in range(config['n_task_groups']):
        chunk_ids = batch.get_chunk_ids_for_group(
            group_id,
            DEFAULT_CPUS_PER_TASK,
            config['n_chunks'],
        )
        start_chunk = chunk_ids[0]
        end_chunk = chunk_ids[-1]
        start_file, _ = batch.get_chunk_bounds(
            start_chunk,
            config['files_per_chunk'],
            config['total_files'],
        )
        _, end_file = batch.get_chunk_bounds(
            end_chunk,
            config['files_per_chunk'],
            config['total_files'],
        )
        print(
            f'group={group_id}',
            f'chunks={start_chunk}-{end_chunk}',
            f'n_chunks={len(chunk_ids)}',
            f'files={end_file - start_file}',
            flush=True,
        )
    return config


def merge_failures(config):
    '''Merge per-chunk failure logs into one failure log.'''
    failure_dir = Path(config['run_dir']) / 'failures'
    merged_path = Path(config['run_dir']) / 'failures.jsonl'
    with merged_path.open('w') as fout:
        for failure_path in sorted(failure_dir.glob('failures_*.jsonl')):
            fout.write(failure_path.read_text())
    return merged_path


def merge_audio_info(config):
    '''Merge per-chunk input-audio logs into one metadata file.'''
    audio_info_dir = Path(config['run_dir']) / 'audio_info'
    merged_path = Path(config['run_dir']) / 'audio_info.jsonl'
    with merged_path.open('w') as fout:
        for audio_info_path in sorted(audio_info_dir.glob('audio_*.jsonl')):
            fout.write(audio_info_path.read_text())
    return merged_path


def count_valid_outputs(config):
    '''Count valid output files referenced by the manifest.'''
    completed = 0
    for global_index, filename in batch.iter_manifest_slice(
        config['manifest_path'],
        0,
        config['total_files'],
    ):
        chunk_id = global_index // config['files_per_chunk']
        output_shard_dir = batch.get_chunk_output_dir(chunk_id)
        output_filename = get_output_filename(
            filename,
            output_dir=config['output_dir'],
            input_dir=config['input_dir'],
            output_shard_dir=output_shard_dir,
            n_bands=config['n_bands'],
        )
        if batch.is_valid_output_file(output_filename):
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
        'n_task_groups': config['n_task_groups'],
        'cpus_per_task': DEFAULT_CPUS_PER_TASK,
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

    dry_run_parser = subparsers.add_parser('dry_run')
    dry_run_parser.add_argument('config_path')

    group_parser = subparsers.add_parser('group')
    group_parser.add_argument('config_path')
    group_parser.add_argument('group_id', type=int)

    finalize_parser = subparsers.add_parser('finalize')
    finalize_parser.add_argument('config_path')
    return parser


def main():
    args = build_cli().parse_args()
    if args.command == 'prepare':
        prepared = prepare_run(args.config_path)
        print(
            prepared['n_task_groups'],
            prepared['max_parallel_tasks'],
            prepared['run_dir'],
        )
        return
    if args.command == 'dry_run':
        dry_run(args.config_path)
        return
    if args.command == 'group':
        process_group(args.config_path, args.group_id)
        return
    finalize_run(args.config_path)


if __name__ == '__main__':
    main()
