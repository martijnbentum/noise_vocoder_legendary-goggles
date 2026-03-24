import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np

from . import audio
from . import plot
from . import signal_processing as sp

'''
This code implements a vocoder, which is a device that encodes speech signals
into a form that can be transmitted or stored more efficiently. The vocoder
works by filtering the input signal into a number of frequency bands,
extracting the amplitude envelope of each band, and then modulating the
envelope with white noise. The resulting signal is a vocoded version of the
original signal.

The vocoder is implemented as a class, which takes an audio signal or a
filename as input. The class has methods for filtering the signal into
frequency bands, extracting the amplitude envelope of each band, and
modulating the envelope with white noise. The class also has methods for
plotting the original and vocoded signals, as well as the filtered signals
for each frequency band. The vocoder can also play the original and vocoded
signals.

Issues:
Number and bounds of frequency bands.
    currently set to 6 bands
    based on https://www.mrc-cbu.cam.ac.uk/personal/matt.davis/vocode/
    the bands are approximately logarithmically spaced

Filtering order
    currently uses a butterworth bandpass filter with order 4
    higher order results in sharper cutoffs (and more artefacts)

Envelope extraction
    approach based on Praat cited in (3)
    in the paper they mention squaring the signal, I use absolute value 
    (otherwise the envelope explodes if you no normalize or dissappears if you do)
    the envelope is extracted by convolving a kaiser 20 window of 64 ms in length
    with the signal.
    in this approach the envelope is a bit left shifted compared to the 
    alternative below which is a bit right shifted

    Alternative
    the envelope is extracted using a low-pass filter
    the cutoff frequency of the low-pass filter is set to 30 Hz
    the order of the low-pass filter is set to 2
    the envelope is smoothed using a moving average with a window size of 100

Normalizing vocoded signal
    the vocoded signal of each frequency bands is normalized to match the 
    intensity of the original signal with a window size of 100 ms based on the
    root mean square of the windowed signal.

match rms
    you can match the rms of the vocoded signal of each band to the original 
    filtered signal of a given band

carrier signal
    the vocoded signal is modulated with white noise
    the white noise is generated using the scipy white noise function
    the white noise is filtered using a bandpass filter with the same 
    cutoff frequencies as the bandpass filter used for the original signal
    the white noise is then modulated with the envelope of the original signal
    to create the vocoded signal

    alternative carrier signal could be a sine wave

references:
(0) https://www.mrc-cbu.cam.ac.uk/personal/matt.davis/vocode/
https://github.com/mdhk/speech-training/blob/main/scripts/vocoder.py
https://github.com/achabotl/vocoder/blob/master/Vocoder.ipynb

(1) Speech Recognition with Primarily Temporal Cues
https://www.science.org/doi/pdf/10.1126/science.270.5234.303?casa_token=FWxVAAH3DwAAAAAA:qb9c5mG-1mHbMd_VZzZ1FmQGez3XAkLgAeMNLQBGrNCQy0ZGd33aBhjY0Pm5hLsJ8qn35Y0v1nYkx5o

(2) Hierarchical Processing in Spoken Language Comprehension
https://www.jneurosci.org/content/jneuro/23/8/3423.full.pdf

(3) Lexical Information Drives Perceptual Learning of Distorted Speech:
Evidence From the Comprehension of Noise-Vocoded Sentences
https://pure.mpg.de/rest/items/item_2304886_3/component/file_2304918/content

(4)Effects of envelope bandwidth on the intelligibility of sine- and
noise-vocoded speech
https://pmc.ncbi.nlm.nih.gov/articles/PMC2730710/pdf/JASMAN-000126-000792_1.pdf
'''

REPO_ROOT = Path(__file__).resolve().parents[1]
FREQUENCY_CONFIG_FILENAME = REPO_ROOT / 'config' / 'frequency_bands.json'
DEFAULT_MAX_OUTPUT_FILES_PER_DIR = 10000


def load_frequency_config(filename = FREQUENCY_CONFIG_FILENAME):
    '''Load standard frequency bands from JSON config.'''
    with Path(filename).open() as fin:
        return json.load(fin)


def get_standard_bands(
    n_bands = None,
    family = 'default_family',
    key = None,
    filename = FREQUENCY_CONFIG_FILENAME,
):
    '''Return a standard band definition from config.'''
    config = load_frequency_config(filename)
    if family not in config:
        raise ValueError(f'Unknown frequency family: {family}')
    family_config = config[family]
    if key is None:
        if n_bands is None:
            raise ValueError('Either n_bands or key must be provided')
        key = f'{n_bands}_band'
    if key not in family_config:
        raise ValueError(
            f'Unknown frequency band key for {family}: {key}'
        )
    return np.array(family_config[key])

class Vocoder:
    def __init__(self, signal = None, sample_rate = 16000, frequencies = None,
        filename = None, butterworth_order = 4, match_rms = True,
        output_dir = '', input_dir = '', output_shard_dir = '',):
        if filename is None and isinstance(signal, (str, os.PathLike)):
            filename = signal
            signal = None
        if signal is None and filename is None:
            raise ValueError('Either signal or filename must be provided')
        if filename: 
            signal, sample_rate = audio.load_audio_file(filename, sample_rate)
        self.filename = filename
        self.butterworth_order = butterworth_order
        self.match_rms = match_rms
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.output_shard_dir = output_shard_dir
        self.path = Path(filename) if filename else None
        self.signal = signal
        self.white_noise = sp.white_noise(n_samples=len(signal))
        self.sample_rate = sample_rate
        self.duration = len(signal) / sample_rate
        self.info = audio.audio_info(filename) if filename else None
        self._check_info()
        if frequencies is None:
            self.frequencies = get_standard_bands(6)
        else:
            self.frequencies = frequencies
        self._filter()
        self.signal_intensity = sp.compute_praat_intensity(self.signal)
        self.vocoded_intensity = sp.compute_praat_intensity(self.vocoded_signal)

    def __repr__(self):
        m = f'Vocoder: # bands {self.n_bands}' 
        m += f' | min freq: {self.minimum_frequency}'
        m += f' | max freq: {self.maximum_frequency}'
        m += f' | {self.signal_intensity:.2f} dB'
        m += f' | {self.vocoded_intensity:.2f} dB'
        if self.filename:
            m += f' | {self.path.name}'
        return m

    def __str__(self):
        m = self.__repr__()
        m += f'\nSignal intensity:  {self.signal_intensity:.2f} dB'
        m += f'\nVocoder intensity: {self.vocoded_intensity:.2f} dB'
        if self.info:
            m += f'\nAudio info:\n'
            for k,v in self.info.items():
                m += f'\t{k.ljust(12)}: {v}\n'
        else: m += '\n'
        m += 'Frequency bands:\n'
        for band in self.bands:
            m += f'\t{band}\n'
        return m

    def _filter(self):
        self.bands = []
        for i in range(len(self.frequencies)-1):
            low_frequency = self.frequencies[i]
            high_frequency = self.frequencies[i+1]
            band = Frequency_band(low_frequency, high_frequency, self)
            self.bands.append(band)
        self.n_bands = len(self.bands)
        self.minimum_frequency = self.frequencies[0]
        self.maximum_frequency = self.frequencies[-1]

    def _check_info(self):
        if not self.info:
            self.checked = False
            return
        if self.info['sample_rate'] != self.sample_rate:
            m = f'Message: sample rate mismatch: '
            m += f'{self.info["sample_rate"]} != {self.sample_rate}'
            m += 'librosa resampled the signal'
        if self.info['n_channels'] != 1:
            raise NotImplementedError('Only mono files are supported')
        self.duration_delta =  abs(self.info['duration'] - self.duration)
        if self.duration_delta > 0.1:
            m = f'Error: duration mismatch: '
            m += f'{self.info["duration"]} != {self.duration}'
            raise ValueError(m)
        self.checked = True

    def plot_signal(self, show_envelope = True):
        '''plot the signal'''
        envelope = self.envelope if show_envelope else None
        plot.plot_signal(self.signal, self.sample_rate, title=self.path.name,
            envelope = envelope)

    @property
    def envelope(self):
        if hasattr(self, '_envelope'): return self._envelope
        '''
        x = sp.extract_envelope(self.signal, cutoff=30, order=2, 
            sample_rate=self.sample_rate, smoothing=True)
        '''
        x = sp.extract_kaiser_20_envelope(self.signal, kaiser_beta=20, 
            sample_rate=self.sample_rate)
        self._envelope = x
        return self._envelope

    @property
    def vocoded_signal(self):
        '''return the vocoded signal'''
        if hasattr(self, '_vocoded_signal'): return self._vocoded_signal
        x = np.zeros_like(self.signal)
        for band in self.bands:
            x += band.vocoded_signal
        self._vocoded_signal = x
        return self._vocoded_signal

    def plot_compare_spectrogram(self):
        '''plot the spectrogram of the original and vocoded signal'''
        plot.compare_spectrogram(self.signal, self.vocoded_signal, 
            sample_rate=self.sample_rate, names = ['Original', 'Vocoded'])

    def plot_stacked_filtered_sigals(self, show_envelope = True):
        plot_name = f'{self.path.name} - filtered signals'
        signals = [b.filtered_signal for b in self.bands]
        names = [b.__repr__() for b in self.bands]
        envelopes = [b.envelope for b in self.bands] if show_envelope else None
        plot.plot_stacked_sigals(signals, names, envelopes, 
            sample_rate=self.sample_rate, title=plot_name)

    def plot_stacked_vocoded_sigals(self, show_envelope = True):
        plot_name = f'{self.path.name} - vocoded signals'
        signals = [b.vocoded_signal for b in self.bands]
        names = [b.__repr__() for b in self.bands]
        envelopes = [b.envelope for b in self.bands] if show_envelope else None
        plot.plot_stacked_sigals(signals, names, envelopes, 
            sample_rate=self.sample_rate, title=plot_name)

    def plot_grid_filtered_vocoded_sigals(self, show_envelope = True):
        plot_name = f'{self.path.name} - filtered and vocoded signals'
        left_side_signals = [b.filtered_signal for b in self.bands]
        right_side_signals = [b.vocoded_signal for b in self.bands]
        left_side_names = ['filtered ' + b.__repr__() for b in self.bands]
        right_side_names = ['vocoded ' + b.__repr__() for b in self.bands]
        left_envelopes = [b.envelope for b in self.bands] if show_envelope else None
        right_envelopes = [b.envelope for b in self.bands] if show_envelope else None
        plot.plot_grid_signals(left_side_signals, right_side_signals,
            left_side_names, right_side_names, 
            left_side_envelopes = left_envelopes,
            right_side_envelopes = right_envelopes, title=plot_name,
            sample_rate=self.sample_rate)

    def plot_original_vocoded_signals(self):
        plot_name = f'{self.path.name} - original and vocoded signals'
        left_side_signals = [self.signal]
        right_side_signals = [self.vocoded_signal]
        left_side_names = [f'original (Intensity: {self.signal_intensity:.2f} dB)']
        right_side_names = [f'vocoded (Intensity: {self.vocoded_intensity:.2f} dB)']
        left_envelopes = [self.envelope]
        right_envelopes = [self.envelope]
        plot.plot_grid_signals(left_side_signals, right_side_signals,
            left_side_names, right_side_names, 
            left_side_envelopes = left_envelopes,
            right_side_envelopes = right_envelopes, title=plot_name,
            sample_rate=self.sample_rate, figsize=(10, 5))

    def play_original(self):
        '''play the original signal'''
        audio.play_audio(self.signal, self.sample_rate)

    def play_vocoded(self):
        '''play the vocoded signal'''
        audio.play_audio(self.vocoded_signal, self.sample_rate)

    def write_vocoded(self, filename = None):
        '''write the vocoded signal to a file'''
        if filename is None and self.filename is None:
            raise ValueError('Either filename or self.filename must be provided')
        if filename is None: filename = self.filename
        filename = get_output_filename(
            filename,
            output_dir=self.output_dir,
            input_dir=self.input_dir,
            output_shard_dir=self.output_shard_dir,
            n_bands=self.n_bands,
        )
        audio.write_audio(self.vocoded_signal, filename, self.sample_rate)
        return filename

        
            


class Frequency_band:
    def __init__(self, low_frequency, high_frequency, parent):
        self.low_frequency = low_frequency
        self.high_frequency = high_frequency
        self.parent = parent
        self._filter()
        self._extract_envelope()

    def __repr__(self):
        m = f'Frequency band: {self.low_frequency} - {self.high_frequency} Hz'
        return m

    def _filter(self):
        '''apply a bandpass butterworth filter to a signal'''
        y = sp.butterworth_bandpass_filter(self.parent.signal, 
            self.low_frequency, self.high_frequency, self.parent.sample_rate,
            order = self.parent.butterworth_order)
        self.filtered_signal = y

    def _extract_envelope(self, cutoff=30, order=2, sample_rate=None, 
        smoothing = True):
        '''extract the amplitude envelope of a signal using a low-pass filter.
        x                  the signal
        cutoff             the cutoff frequency of the low-pass filter
        order              the order of the low-pass filter
        sample_rate        the sample rate of the signal
        smoothing          whether to apply a moving average to the envelope
                            to smooth it 
        '''
        if sample_rate is None:
            sample_rate = self.sample_rate
        self.envelope = sp.extract_envelope(self.filtered_signal, cutoff, order, 
            sample_rate, smoothing)

    def plot_compare_spectrogram(self):
        '''plot the spectrogram of the original and vocoded signal'''
        n = self.__repr__()
        plot.compare_spectrogram(self.filtered_signal, self.vocoded_signal, 
            sample_rate=self.sample_rate, names = [f'Filtered {n}', 'Vocoded'])

    def plot_filtered_signal(self, show_envelope = True):
        '''plot the signal'''
        envelope = self.envelope if show_envelope else None
        plot.plot_signal(self.filtered_signal, self.sample_rate, 
            title=f'{self.path.name} - {self.__repr__()} filtered',
            envelope = envelope)

    def plot_vocoded_signal(self, show_envelope = True):
        '''plot the signal'''
        envelope = self.envelope if show_envelope else None
        plot.plot_signal(self.vocoded_signal, self.sample_rate, 
            title=f'{self.path.name} - {self.__repr__()} vocoded',
            envelope = envelope)

    def plot_original_vocoded_signals(self):
        plot_name = f'{self.path.name} - filtered and vocoded signals'
        left_side_signals = [self.signal]
        right_side_signals = [self.vocoded_signal]
        left_side_names = [f'filtered {self.__repr__()}']
        right_side_names = [f'vocoded {self.__repr__()}']
        left_envelopes = [self.envelope]
        right_envelopes = [self.envelope]
        plot.plot_grid_signals(left_side_signals, right_side_signals,
            left_side_names, right_side_names, 
            left_side_envelopes = left_envelopes,
            right_side_envelopes = right_envelopes, title=plot_name,
            sample_rate=self.sample_rate, figsize=(10, 5))

    @property
    def sample_rate(self):
        return self.parent.sample_rate

    @property
    def path(self):
        return self.parent.path

    @property
    def white_noise(self):
        return self.parent.white_noise

    @property
    def signal(self):
        return self.parent.signal
            
    @property
    def vocoded_signal(self):
        '''return the vocoded signal'''
        if hasattr(self, '_vocoded_signal'): return self._vocoded_signal
        carrier = sp.butterworth_bandpass_filter(
            self.white_noise,
            self.low_frequency, self.high_frequency, self.parent.sample_rate)
        x = self.envelope * carrier
        if self.parent.match_rms:
            x = sp.match_rms_by_window(self.filtered_signal, x)
        self._vocoded_signal = x
        return self._vocoded_signal

def handle_nbands(args):
    return get_standard_bands(
        n_bands=args.nbands,
        family=getattr(args, 'frequency_family', 'default_family'),
        key=getattr(args, 'frequency_key', None),
    )


def handle_frequencies(args):
    if not args.frequencies:
        return handle_nbands(args)
    return np.array(args.frequencies)


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
    output_dir = '',
    input_dir = '',
    output_shard_dir = '',
    n_bands = None,
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


def build_output_stem(filename, input_dir = ''):
    '''Return a collision-safe output stem for batch outputs.'''
    path = Path(filename)
    relative_path = get_relative_input_path(path, input_dir)
    digest = hashlib.sha1(relative_path.encode('utf-8')).hexdigest()[:8]
    return f'{digest}__{path.stem}'


def build_output_shard_map(
    filenames,
    input_dir,
    max_files_per_output_dir = DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
):
    '''Map input files to shard dirs for flat batch output directories.'''
    if max_files_per_output_dir < 1:
        return {}
    shard_map = {}
    for index, filename in enumerate(filenames):
        shard_index = index // max_files_per_output_dir
        shard_map[str(filename)] = make_output_shard_name(shard_index)
    return shard_map


def log_progress(processed, total, start_time, last_result = None):
    '''Print a compact batch progress line.'''
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(total - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    message = (
        f'progress: {processed}/{total} files '
        f'({rate:.2f} files/s, elapsed={elapsed:.1f}s)'
    )
    if eta_seconds is not None:
        message += f' eta={eta_seconds / 3600:.2f}h'
    if last_result:
        message += f' latest={Path(last_result["output_filename"]).name}'
    print(message, flush=True)


def run_parallel_batch(args, argss, total_files):
    '''Run a batch with compact progress reporting.'''
    start_time = time.time()
    processed = 0
    failures = 0
    progress_every = max(1, getattr(args, 'progress_every', 100))
    chunksize = max(1, min(50, total_files // (args.nprocess * 4) or 1))
    metadata_path = get_metadata_path(
        getattr(args, 'output_dir', ''),
        getattr(args, 'metadata_filename', ''),
    )
    failure_path = get_failure_path(
        getattr(args, 'output_dir', ''),
        getattr(args, 'failure_filename', ''),
    )
    print(
        'pool launch:',
        f'workers={args.nprocess}',
        f'chunksize={chunksize}',
        f'maxtasksperchild={args.worker_max_tasks}',
        flush=True,
    )
    if metadata_path:
        print(f'metadata_log: {metadata_path}', flush=True)
    if failure_path:
        print(f'failure_log: {failure_path}', flush=True)
    with multiprocessing.Pool(
        args.nprocess,
        maxtasksperchild=args.worker_max_tasks,
    ) as pool:
        for result in pool.imap_unordered(
            handle_task,
            argss,
            chunksize=chunksize,
        ):
            processed += 1
            if result['status'] == 'ok':
                append_metadata(metadata_path, result)
            else:
                failures += 1
                append_metadata(failure_path, result)
                print(
                    'file failed:',
                    result['input_filename'],
                    flush=True,
                )
            if processed == 1 or processed % progress_every == 0:
                log_progress(processed, total_files, start_time, result)
    log_progress(processed, total_files, start_time)
    if failures:
        print(f'failed_files: {failures}', flush=True)


def build_worker_config(args):
    '''Create the shared worker config for a batch run.'''
    frequencies = handle_frequencies(args)
    return {
        'sample_rate': args.sample_rate,
        'butterworth_order': args.butterworth_order,
        'match_rms': args.match_rms,
        'output_dir': args.output_dir,
        'input_dir': getattr(args, 'input_dir', ''),
        'frequencies': frequencies.tolist(),
    }


def iter_batch_tasks(filenames, worker_config, output_shard_map = None):
    '''Yield lightweight tasks for the process pool.'''
    if output_shard_map is None:
        output_shard_map = {}
    for filename in filenames:
        yield {
            'filename': str(filename),
            'output_shard_dir': output_shard_map.get(str(filename), ''),
            'worker_config': worker_config,
        }


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
    fn = sorted(Path(args.input_dir).rglob('*.wav'))
    if not fn:
        raise ValueError('No wav files found in input_dir')
    print(f'vocoding {len(fn)} .wav files in input_dir', flush=True)
    print(
        'parallel summary:',
        f'nprocess={args.nprocess}',
        f'family={getattr(args, "frequency_family", "default_family")}',
        f'key={getattr(args, "frequency_key", None)}',
        f'nbands={args.nbands}',
        'max_output_files_per_dir='
        f'{getattr(args, "max_output_files_per_dir", DEFAULT_MAX_OUTPUT_FILES_PER_DIR)}',
        f'progress_every={getattr(args, "progress_every", 100)}',
        flush=True,
    )
    output_shard_map = build_output_shard_map(
        fn,
        args.input_dir,
        getattr(
            args,
            'max_output_files_per_dir',
            DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
        ),
    )
    if args.nprocess == 1:
        for filename in fn:
            args.filename = filename
            args.output_shard_dir = output_shard_map.get(str(filename), '')
            handle_filename(args)
        return
    worker_config = build_worker_config(args)
    tasks = iter_batch_tasks(fn, worker_config, output_shard_map)
    run_parallel_batch(args, tasks, len(fn))
    
        
def handle_filename(args):
    start_time = time.time()
    worker_pid = os.getpid()
    frequencies = handle_frequencies(args)
    vocoder = Vocoder(filename=args.filename, sample_rate=args.sample_rate,
        butterworth_order=args.butterworth_order, match_rms=args.match_rms,
        frequencies=frequencies, output_dir=args.output_dir,
        input_dir=getattr(args, 'input_dir', ''),
        output_shard_dir=getattr(args, 'output_shard_dir', ''))
    output_filename = vocoder.write_vocoded()
    return make_success_result(
        args.filename,
        output_filename,
        time.time() - start_time,
        vocoder.signal_intensity,
        vocoder.vocoded_intensity,
        vocoder.n_bands,
    )


def handle_task(task):
    '''Process one pooled batch task and return a JSON-safe result.'''
    start_time = time.time()
    worker_pid = os.getpid()
    filename = task['filename']
    output_shard_dir = task.get('output_shard_dir', '')
    config = task['worker_config']
    try:
        vocoder = Vocoder(
            filename=filename,
            sample_rate=config['sample_rate'],
            butterworth_order=config['butterworth_order'],
            match_rms=config['match_rms'],
            frequencies=np.array(config['frequencies']),
            output_dir=config['output_dir'],
            input_dir=config['input_dir'],
            output_shard_dir=output_shard_dir,
        )
        output_filename = vocoder.write_vocoded()
        result = make_success_result(
            filename,
            output_filename,
            time.time() - start_time,
            vocoder.signal_intensity,
            vocoder.vocoded_intensity,
            vocoder.n_bands,
        )
        result['worker_pid'] = int(worker_pid)
        return result
    except Exception as exc:
        return make_failure_result(
            filename,
            time.time() - start_time,
            worker_pid,
            exc,
        )


def build_parser():
    parser = argparse.ArgumentParser(description='Vocoder')
    parser.add_argument('--filename', type=str, help='audio file to vocode')
    parser.add_argument('--sample_rate', type=int, default=16000,
        help='sample rate of the audio file')
    parser.add_argument('--output_dir', type=str, default='',
        help='output directory for the vocoded file')
    parser.add_argument('--butterworth_order', type=int, default=4,
        help='order of the butterworth filter')
    parser.add_argument('--match_rms', action='store_true',
        help='match the rms of the vocoded signal to the original signal')
    parser.add_argument('--nbands', type=int, default=6,
        help='number of frequency bands to use')
    parser.add_argument('--frequency_family', type=str,
        default='default_family',
        help='frequency family from config to use')
    parser.add_argument('--frequency_key', type=str, default=None,
        help='frequency band key from config to use, e.g. 8_band')
    parser.add_argument('--frequencies', type=int, nargs='+',
        default=None, 
        help='frequencies to use for the vocoder e.g. 100 300 1000')
    parser.add_argument('--input_dir', type=str, default='',
        help='input directory for the audio files')
    parser.add_argument('--nprocess', type=int, default=1,
        help='number of processes to use for vocoding')
    parser.add_argument('--progress_every', type=int, default=100,
        help='print one progress line after this many completed files')
    parser.add_argument('--worker_max_tasks', type=int, default=100,
        help='restart worker processes after this many files')
    parser.add_argument('--metadata_filename', type=str,
        default='',
        help='jsonl file for per-file batch metadata, relative to output_dir')
    parser.add_argument('--failure_filename', type=str,
        default='vocoder_failures.jsonl',
        help='jsonl file for per-file batch failures, relative to output_dir')
    parser.add_argument('--max_output_files_per_dir', type=int,
        default=DEFAULT_MAX_OUTPUT_FILES_PER_DIR,
        help='maximum wav files per output directory before sharding')
    return parser


def main():
    args = build_parser().parse_args()
    start = time.time()
    handle_args(args)
    print(f'Elapsed time: {time.time() - start:.2f} seconds')


if __name__ == '__main__':
    main()
