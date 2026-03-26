import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from itertools import islice
from pathlib import Path
import time
import traceback

import soundfile as sf

from .file_io import append_metadata
from .file_io import build_output_shard_map
from .file_io import DEFAULT_MAX_OUTPUT_FILES_PER_DIR
from .file_io import get_failure_path
from .file_io import get_metadata_path
from .file_io import get_output_filename
from .file_io import make_output_shard_name
from .file_io import prepare_output_dir
from .vocoder import handle_frequencies
from .vocoder import Vocoder

DEFAULT_FILES_PER_CHUNK = 500
DEFAULT_PROGRESS_EVERY = 25


def run_batch(args, filenames, output_shard_map):
    '''Run a batch sequentially with compact error reporting.'''
    processed = 0
    failures = 0
    start_time = time.time()
    metadata_path = get_metadata_path(
        getattr(args, 'output_dir', ''),
        getattr(args, 'metadata_filename', ''),
    )
    failure_path = get_failure_path(
        getattr(args, 'output_dir', ''),
        getattr(args, 'failure_filename', ''),
    )
    print(
        'batch launch:',
        'mode=sequential',
        flush=True,
    )
    if metadata_path:
        print(f'metadata_log: {metadata_path}', flush=True)
    if failure_path:
        print(f'failure_log: {failure_path}', flush=True)
    for filename in filenames:
        args.filename = filename
        args.output_shard_dir = output_shard_map.get(str(filename), '')
        try:
            result = handle_filename(args)
        except Exception as exc:
            failures += 1
            result = make_failure_result(
                filename,
                0.0,
                os.getpid(),
                exc,
            )
            append_metadata(failure_path, result)
            print(
                'file failed:',
                result['input_filename'],
                flush=True,
            )
            continue
        processed += 1
        append_metadata(metadata_path, result)
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(
        'batch complete:',
        f'processed={processed}',
        f'elapsed={elapsed:.1f}s',
        f'rate={rate:.2f} files/s',
        flush=True,
    )
    if failures:
        print(f'failed_files: {failures}', flush=True)
        raise RuntimeError(f'Batch failed for {failures} files')


def make_success_result(
    filename,
    output_filename,
    elapsed_seconds,
    signal_intensity_db,
    vocoded_intensity_db,
    n_bands,
):
    '''Create a JSON-safe success result.'''
    return {
        'status': 'ok',
        'input_filename': str(filename),
        'output_filename': str(output_filename),
        'elapsed_seconds': float(round(elapsed_seconds, 4)),
        'signal_intensity_db': float(round(signal_intensity_db, 4)),
        'vocoded_intensity_db': float(round(vocoded_intensity_db, 4)),
        'n_bands': int(n_bands),
    }


def make_failure_result(filename, elapsed_seconds, worker_pid, exc):
    '''Create a JSON-safe failure result.'''
    return {
        'status': 'error',
        'worker_pid': int(worker_pid),
        'input_filename': str(filename),
        'elapsed_seconds': float(round(elapsed_seconds, 4)),
        'error_type': exc.__class__.__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
    }


def build_parallel_config(args):
    '''Build one manifest-driven config for local chunk processing.'''
    if not args.manifest_path:
        raise ValueError('manifest_path is required for parallel batch mode')
    manifest_path = Path(args.manifest_path)
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(args.output_dir).parent / '_vocode_run'
    )
    frequencies = handle_frequencies(args)
    total_files = read_manifest_count(manifest_path)
    config = {
        'manifest_path': str(manifest_path),
        'run_dir': str(run_dir),
        'files_per_chunk': int(args.files_per_chunk),
        'sample_rate': int(args.sample_rate),
        'match_rms': bool(args.match_rms),
        'frequencies': [int(value) for value in frequencies],
        'output_dir': str(args.output_dir),
        'input_dir': str(args.input_dir),
        'total_files': int(total_files),
        'n_bands': len(frequencies) - 1,
    }
    ensure_run_directories(config)
    return config


def ensure_run_directories(config):
    '''Create the standard metadata directories for one run.'''
    run_dir = Path(config['run_dir'])
    (run_dir / 'progress').mkdir(parents=True, exist_ok=True)
    (run_dir / 'failures').mkdir(parents=True, exist_ok=True)
    (run_dir / 'audio_info').mkdir(parents=True, exist_ok=True)


def get_chunk_bounds(chunk_id, files_per_chunk, total_files):
    '''Return manifest bounds for one chunk index.'''
    start = chunk_id * files_per_chunk
    end = min(total_files, start + files_per_chunk)
    return start, end


def get_chunk_ids_for_group(group_id, chunks_per_group, n_chunks):
    '''Return the chunk ids assigned to one task group.'''
    start = group_id * chunks_per_group
    end = min(n_chunks, start + chunks_per_group)
    return list(range(start, end))


def iter_manifest_slice(manifest_path, start, end):
    '''Yield one indexed slice from a manifest file.'''
    with Path(manifest_path).open() as fin:
        for index, line in enumerate(islice(fin, start, end), start=start):
            yield index, line.strip()


def read_manifest_count(manifest_path):
    '''Return the number of files in a manifest.'''
    with Path(manifest_path).open() as fin:
        return sum(1 for _ in fin)


def get_chunk_progress_path(run_dir, chunk_id):
    '''Return the progress file path for one chunk.'''
    return Path(run_dir) / 'progress' / f'chunk_{chunk_id:05d}.json'


def get_chunk_failure_path(run_dir, chunk_id):
    '''Return the failure log path for one chunk.'''
    return Path(run_dir) / 'failures' / f'failures_{chunk_id:05d}.jsonl'


def get_chunk_audio_info_path(run_dir, chunk_id):
    '''Return the input-audio log path for one chunk.'''
    return Path(run_dir) / 'audio_info' / f'audio_{chunk_id:05d}.jsonl'


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


def get_chunk_output_dir(chunk_id):
    '''Return the wav output dir name for one processing chunk.'''
    return make_output_shard_name(chunk_id)


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


def make_audio_info_record(filename, input_filename = ''):
    '''Return a compact output-audio record for one vocoded file.'''
    path = Path(filename)
    audio_info = sf.info(path)
    return {
        'input_filename': str(input_filename) if input_filename else '',
        'output_filename': str(path),
        'file_stem': path.stem,
        'n_samples': int(audio_info.frames),
    }


def print_chunk_progress(chunk_id, stats, elapsed_seconds):
    '''Print a compact progress line with ETA for one chunk.'''
    done = stats['processed'] + stats['skipped'] + stats['failed']
    assigned = stats['assigned']
    rate = done / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining = assigned - done
    eta_seconds = remaining / rate if rate > 0 else 0.0
    print(
        f'chunk={chunk_id}',
        f'processed={done}/{assigned}',
        f'created={stats["processed"]}',
        f'skipped={stats["skipped"]}',
        f'failed={stats["failed"]}',
        f'rate={rate:.2f} files/s',
        f'eta={eta_seconds:.0f}s',
        flush=True,
    )


def process_manifest_chunk(config, chunk_id):
    '''Process one manifest chunk and write per-chunk logs.'''
    start, end = get_chunk_bounds(
        chunk_id,
        config['files_per_chunk'],
        config['total_files'],
    )
    progress_path = get_chunk_progress_path(config['run_dir'], chunk_id)
    failure_path = get_chunk_failure_path(config['run_dir'], chunk_id)
    audio_info_path = get_chunk_audio_info_path(config['run_dir'], chunk_id)
    if progress_path.exists():
        progress_path.unlink()
    if failure_path.exists():
        failure_path.unlink()
    if audio_info_path.exists():
        audio_info_path.unlink()
    print(f'failure_log: {failure_path}', flush=True)
    stats = {
        'task_id': int(chunk_id),
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
        print(f'chunk={chunk_id} assigned=0', flush=True)
        return stats
    for local_index, (global_index, filename) in enumerate(
        iter_manifest_slice(config['manifest_path'], start, end),
        start=1,
    ):
        output_shard_dir = get_chunk_output_dir(chunk_id)
        output_filename = get_output_filename(
            filename,
            output_dir=config['output_dir'],
            input_dir=config['input_dir'],
            output_shard_dir=output_shard_dir,
            n_bands=config['n_bands'],
        )
        if is_valid_output_file(output_filename):
            stats['skipped'] += 1
            try:
                append_metadata(
                    audio_info_path,
                    make_audio_info_record(
                        output_filename,
                        input_filename=filename,
                    ),
                )
            except RuntimeError:
                pass
        else:
            args = make_chunk_args(config, filename, output_shard_dir)
            try:
                result = handle_filename(args)
                stats['processed'] += 1
                try:
                    append_metadata(
                        audio_info_path,
                        make_audio_info_record(
                            result['output_filename'],
                            input_filename=filename,
                        ),
                    )
                except RuntimeError:
                    pass
            except Exception as exc:
                stats['failed'] += 1
                append_metadata(
                    failure_path,
                    make_failure_result(
                        filename,
                        time.time() - started_at,
                        os.getpid(),
                        exc,
                    ),
                )
                print(f'file failed: {filename}', flush=True)
        if (
            local_index % DEFAULT_PROGRESS_EVERY == 0
            or local_index == stats['assigned']
        ):
            elapsed_seconds = time.time() - started_at
            stats['elapsed_seconds'] = float(round(elapsed_seconds, 4))
            write_chunk_progress(progress_path, stats)
            print_chunk_progress(chunk_id, stats, elapsed_seconds)
    elapsed_seconds = time.time() - started_at
    stats['status'] = 'done'
    stats['elapsed_seconds'] = float(round(elapsed_seconds, 4))
    write_chunk_progress(progress_path, stats)
    print(
        f'chunk={chunk_id} done',
        f'assigned={stats["assigned"]}',
        f'created={stats["processed"]}',
        f'skipped={stats["skipped"]}',
        f'failed={stats["failed"]}',
        f'elapsed={elapsed_seconds:.1f}s',
        flush=True,
    )
    return stats


def _process_manifest_chunk_job(config, chunk_id):
    '''Run one chunk job in a worker process.'''
    return process_manifest_chunk(config, chunk_id)


def process_manifest_chunks_parallel(config, chunk_ids, nprocess = 1):
    '''Process multiple chunk ids serially or in parallel.'''
    ensure_run_directories(config)
    chunk_ids = list(chunk_ids)
    if not chunk_ids:
        return []
    nprocess = max(1, min(int(nprocess), len(chunk_ids)))
    print(
        'chunk group launch:',
        f'nprocess={nprocess}',
        f'chunks={chunk_ids[0]}-{chunk_ids[-1]}',
        f'count={len(chunk_ids)}',
        flush=True,
    )
    start_time = time.time()
    results = []
    if nprocess == 1:
        for chunk_id in chunk_ids:
            results.append(process_manifest_chunk(config, chunk_id))
    else:
        with ProcessPoolExecutor(max_workers=nprocess) as executor:
            future_map = {
                executor.submit(
                    _process_manifest_chunk_job,
                    config,
                    chunk_id,
                ): chunk_id
                for chunk_id in chunk_ids
            }
            for future in as_completed(future_map):
                results.append(future.result())
    results.sort(key=lambda item: item['task_id'])
    elapsed = time.time() - start_time
    processed = sum(item['processed'] for item in results)
    skipped = sum(item['skipped'] for item in results)
    failed = sum(item['failed'] for item in results)
    assigned = sum(item['assigned'] for item in results)
    rate = assigned / elapsed if elapsed > 0 else 0.0
    print(
        'chunk group complete:',
        f'assigned={assigned}',
        f'created={processed}',
        f'skipped={skipped}',
        f'failed={failed}',
        f'elapsed={elapsed:.1f}s',
        f'rate={rate:.2f} files/s',
        flush=True,
    )
    return results


def dry_run_manifest_chunks(config, chunk_ids, nprocess = 1):
    '''Print the chunk layout without processing any audio files.'''
    chunk_ids = list(chunk_ids)
    if not chunk_ids:
        print('dry run: no chunks selected', flush=True)
        return []
    nprocess = max(1, min(int(nprocess), len(chunk_ids)))
    print('dry run:', flush=True)
    print(f'run_dir: {config["run_dir"]}', flush=True)
    print(f'manifest_path: {config["manifest_path"]}', flush=True)
    print(f'total_files: {config["total_files"]}', flush=True)
    print(f'files_per_chunk: {config["files_per_chunk"]}', flush=True)
    print(f'n_chunks: {len(chunk_ids)}', flush=True)
    print(f'nprocess: {nprocess}', flush=True)
    for chunk_id in chunk_ids:
        start, end = get_chunk_bounds(
            chunk_id,
            config['files_per_chunk'],
            config['total_files'],
        )
        print(
            f'chunk={chunk_id}',
            f'start={start}',
            f'end={end}',
            f'files={max(0, end - start)}',
            flush=True,
        )
    return chunk_ids


def handle_args(args):
    if getattr(args, 'manifest_path', ''):
        config = build_parallel_config(args)
        n_chunks = int(
            math.ceil(config['total_files'] / config['files_per_chunk'])
        )
        if args.chunk_ids:
            chunk_ids = args.chunk_ids
        elif args.task_id is not None:
            chunk_ids = [args.task_id]
        else:
            chunk_ids = list(range(n_chunks))
        if args.dry_run:
            return dry_run_manifest_chunks(
                config,
                chunk_ids,
                nprocess=args.nprocess,
            )
        return process_manifest_chunks_parallel(
            config,
            chunk_ids,
            nprocess=args.nprocess,
        )
    if not args.input_dir and not args.filename:
        raise ValueError('Either input_dir or filename must be provided')
    prepare_output_dir(getattr(args, 'output_dir', ''))
    if not args.input_dir:
        return handle_filename(args)
    filenames = sorted(Path(args.input_dir).rglob('*.wav'))
    if not filenames:
        raise ValueError('No wav files found in input_dir')
    print(f'vocoding {len(filenames)} .wav files in input_dir', flush=True)
    print(
        'parallel summary:',
        f'family={getattr(args, "frequency_family", "default_family")}',
        f'key={getattr(args, "frequency_key", None)}',
        f'nbands={args.nbands}',
        'max_output_files_per_dir=' + str(
            getattr(
                args,
                'max_output_files_per_dir',
                DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
            )
        ),
        flush=True,
    )
    output_shard_map = build_output_shard_map(
        filenames,
        args.input_dir,
        getattr(
            args,
            'max_output_files_per_dir',
            DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
        ),
    )
    run_batch(args, filenames, output_shard_map)


def handle_filename(args):
    start_time = time.time()
    frequencies = handle_frequencies(args)
    vocoder = Vocoder(
        filename=args.filename,
        sample_rate=args.sample_rate,
        butterworth_order=args.butterworth_order,
        match_rms=args.match_rms,
        frequencies=frequencies,
        output_dir=args.output_dir,
        input_dir=getattr(args, 'input_dir', ''),
        output_shard_dir=getattr(args, 'output_shard_dir', ''),
    )
    output_filename = vocoder.write_vocoded()
    return make_success_result(
        args.filename,
        output_filename,
        time.time() - start_time,
        vocoder.signal_intensity,
        vocoder.vocoded_intensity,
        vocoder.n_bands,
    )


def build_parser():
    parser = argparse.ArgumentParser(description='Vocoder')
    parser.add_argument('--filename', type=str, help='audio file to vocode')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='sample rate of the audio file',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='',
        help='output directory for the vocoded file',
    )
    parser.add_argument(
        '--butterworth_order',
        type=int,
        default=4,
        help='order of the butterworth filter',
    )
    parser.add_argument(
        '--match_rms',
        action='store_true',
        help='match the rms of the vocoded signal to the original signal',
    )
    parser.add_argument(
        '--nbands',
        type=int,
        default=6,
        help='number of frequency bands to use',
    )
    parser.add_argument(
        '--frequency_family',
        type=str,
        default='default_family',
        help='frequency family from config to use',
    )
    parser.add_argument(
        '--frequency_key',
        type=str,
        default=None,
        help='frequency band key from config to use, e.g. 8_band',
    )
    parser.add_argument(
        '--frequencies',
        type=int,
        nargs='+',
        default=None,
        help='frequencies to use for the vocoder e.g. 100 300 1000',
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='',
        help='input directory for the audio files',
    )
    parser.add_argument(
        '--metadata_filename',
        type=str,
        default='',
        help='jsonl file for per-file batch metadata, relative to output_dir',
    )
    parser.add_argument(
        '--failure_filename',
        type=str,
        default='vocoder_failures.jsonl',
        help='jsonl file for per-file batch failures, relative to output_dir',
    )
    parser.add_argument(
        '--max_output_files_per_dir',
        type=int,
        default=DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
        help='maximum wav files per output directory before sharding',
    )
    parser.add_argument(
        '--manifest_path',
        type=str,
        default='',
        help='manifest file for chunked local batch processing',
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        default='',
        help='metadata directory for chunked local batch processing',
    )
    parser.add_argument(
        '--files_per_chunk',
        type=int,
        default=DEFAULT_FILES_PER_CHUNK,
        help='number of input files per chunk in parallel mode',
    )
    parser.add_argument(
        '--task_id',
        type=int,
        default=None,
        help='single chunk id to process in manifest mode',
    )
    parser.add_argument(
        '--chunk_ids',
        type=int,
        nargs='+',
        default=None,
        help='explicit chunk ids to process in manifest mode',
    )
    parser.add_argument(
        '--nprocess',
        type=int,
        default=1,
        help='worker processes to use in manifest mode',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='print the selected chunk layout without processing files',
    )
    return parser


def main():
    args = build_parser().parse_args()
    start_time = time.time()
    handle_args(args)
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds')
