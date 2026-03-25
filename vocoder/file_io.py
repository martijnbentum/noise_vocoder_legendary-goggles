import hashlib
import json
from pathlib import Path


DEFAULT_MAX_OUTPUT_FILES_PER_DIR = 10000


def prepare_output_dir(output_dir):
    '''Create output_dir if needed and fail if it already has wav files.'''
    if not output_dir:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    existing_wavs = sorted(directory.rglob('*.wav'))
    if existing_wavs:
        raise ValueError(
            f'Output directory already contains wav files: {directory}'
        )
    return directory


def get_output_filename(
    filename,
    output_dir='',
    input_dir='',
    output_shard_dir='',
    n_bands=None,
):
    '''Return the target filename for a vocoded file.'''
    path = Path(filename)
    if output_dir:
        directory = Path(output_dir)
        if output_shard_dir:
            directory = directory / output_shard_dir
        directory.mkdir(parents=True, exist_ok=True)
        output_stem = build_output_stem(path, input_dir)
    else:
        directory = path.parent
        output_stem = path.stem
    output_filename = directory / output_stem
    if n_bands is not None:
        output_filename = f'{output_filename}_voc{n_bands}.wav'
    else:
        output_filename = f'{output_filename}_vocoded.wav'
    return str(output_filename)


def get_metadata_path(output_dir, metadata_filename):
    '''Return the metadata path for a batch run if enabled.'''
    if not metadata_filename:
        return None
    metadata_path = Path(metadata_filename)
    if metadata_path.is_absolute() or not output_dir:
        return metadata_path
    return Path(output_dir) / metadata_path


def append_metadata(metadata_path, result):
    '''Append one completed-file record to the metadata log.'''
    if not metadata_path:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open('a') as fout:
        json.dump(result, fout)
        fout.write('\n')


def get_failure_path(output_dir, failure_filename):
    '''Return the failure log path for a batch run if enabled.'''
    return get_metadata_path(output_dir, failure_filename)


def make_output_shard_name(index):
    '''Return the stable shard name for a zero-based shard index.'''
    return f'chunk_{index:05d}'


def get_relative_input_path(filename, input_dir):
    '''Return a stable relative input path when available.'''
    path = Path(filename)
    if not input_dir:
        return path.name
    try:
        return path.relative_to(Path(input_dir)).as_posix()
    except ValueError:
        return path.name


def build_output_stem(filename, input_dir=''):
    '''Return a collision-safe output stem for batch outputs.'''
    path = Path(filename)
    relative_path = get_relative_input_path(path, input_dir)
    digest = hashlib.sha1(relative_path.encode('utf-8')).hexdigest()[:8]
    return f'{digest}__{path.stem}'


def legacy_output_to_source_filename(
    output_filename,
    legacy_output_dir,
    input_dir,
    n_bands,
):
    '''Map one legacy vocoded output path back to its source input file.'''
    path = Path(output_filename)
    suffix = f'_vocoded_nbands-{n_bands}.wav'
    if not path.name.endswith(suffix):
        raise ValueError(
            f'Legacy output does not end with {suffix}: {output_filename}'
        )
    try:
        relative_path = path.relative_to(Path(legacy_output_dir))
    except ValueError as exc:
        raise ValueError(
            f'Legacy output is outside {legacy_output_dir}: {output_filename}'
        ) from exc
    source_name = path.name[:-len(suffix)] + '.wav'
    return str(Path(input_dir) / relative_path.parent / source_name)


def build_output_shard_map(
    filenames,
    input_dir,
    max_files_per_output_dir=DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
):
    '''Map input files to shard dirs for flat batch output directories.'''
    if max_files_per_output_dir < 1:
        return {}
    shard_map = {}
    for index, filename in enumerate(filenames):
        shard_index = index // max_files_per_output_dir
        shard_map[str(filename)] = make_output_shard_name(shard_index)
    return shard_map
