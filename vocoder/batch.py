import argparse
import os
from pathlib import Path
import time
import traceback

from .file_io import append_metadata
from .file_io import build_output_shard_map
from .file_io import DEFAULT_MAX_OUTPUT_FILES_PER_DIR
from .file_io import get_failure_path
from .file_io import get_metadata_path
from .file_io import prepare_output_dir
from .vocoder import handle_frequencies
from .vocoder import Vocoder


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


def handle_args(args):
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
    return parser


def main():
    args = build_parser().parse_args()
    start_time = time.time()
    handle_args(args)
    print(f'Elapsed time: {time.time() - start_time:.2f} seconds')
