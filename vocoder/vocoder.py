import json
import os
from pathlib import Path

import numpy as np

from . import audio
from .file_io import get_output_filename
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
    (otherwise the envelope explodes if you no normalize or dissappears
    if you do)
    the envelope is extracted by convolving a kaiser 20 window of 64 ms
    in length
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
https://www.science.org/doi/pdf/10.1126/science.270.5234.303

(2) Hierarchical Processing in Spoken Language Comprehension
https://www.jneurosci.org/content/jneuro/23/8/3423.full.pdf

(3) Lexical Information Drives Perceptual Learning of Distorted Speech:
Evidence From the Comprehension of Noise-Vocoded Sentences
https://pure.mpg.de/rest/items/item_2304886_3/component/file_2304918/content

(4)Effects of envelope bandwidth on the intelligibility of sine- and
noise-vocoded speech
https://pmc.ncbi.nlm.nih.gov/articles/PMC2730710/pdf/JASMAN-000126-000792_1.pdf
'''

FREQUENCY_CONFIG_FILENAME = Path(__file__).resolve().with_name(
    'frequency_bands.json'
)

class Vocoder:
    def __init__(self, signal = None, sample_rate = 16000, frequencies = None,
        filename = None, butterworth_order = 4, match_rms = True,
        output_dir = '', input_dir = '', output_shard_dir = '',
        carrier_type = 'noise',):
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
        if carrier_type not in ('noise', 'sine'):
            raise ValueError(
                "carrier_type must be 'noise' or 'sine'"
            )
        self.carrier_type = carrier_type
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
        from . import plot
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
        from . import plot
        plot.compare_spectrogram(self.signal, self.vocoded_signal, 
            sample_rate=self.sample_rate, names = ['Original', 'Vocoded'])

    def plot_stacked_filtered_sigals(self, show_envelope = True):
        from . import plot
        plot_name = f'{self.path.name} - filtered signals'
        signals = [b.filtered_signal for b in self.bands]
        names = [b.__repr__() for b in self.bands]
        envelopes = [b.envelope for b in self.bands] if show_envelope else None
        plot.plot_stacked_sigals(signals, names, envelopes, 
            sample_rate=self.sample_rate, title=plot_name)

    def plot_stacked_vocoded_sigals(self, show_envelope = True):
        from . import plot
        plot_name = f'{self.path.name} - vocoded signals'
        signals = [b.vocoded_signal for b in self.bands]
        names = [b.__repr__() for b in self.bands]
        envelopes = [b.envelope for b in self.bands] if show_envelope else None
        plot.plot_stacked_sigals(signals, names, envelopes, 
            sample_rate=self.sample_rate, title=plot_name)

    def plot_grid_filtered_vocoded_sigals(self, show_envelope = True):
        from . import plot
        plot_name = f'{self.path.name} - filtered and vocoded signals'
        left_side_signals = [b.filtered_signal for b in self.bands]
        right_side_signals = [b.vocoded_signal for b in self.bands]
        left_side_names = ['filtered ' + b.__repr__() for b in self.bands]
        right_side_names = ['vocoded ' + b.__repr__() for b in self.bands]
        if show_envelope:
            left_envelopes = [b.envelope for b in self.bands]
            right_envelopes = [b.envelope for b in self.bands]
        else:
            left_envelopes = None
            right_envelopes = None
        plot.plot_grid_signals(
            left_side_signals,
            right_side_signals,
            left_side_names,
            right_side_names,
            left_side_envelopes=left_envelopes,
            right_side_envelopes=right_envelopes,
            title=plot_name,
            sample_rate=self.sample_rate)

    def plot_original_vocoded_signals(self):
        from . import plot
        plot_name = f'{self.path.name} - original and vocoded signals'
        left_side_signals = [self.signal]
        right_side_signals = [self.vocoded_signal]
        left_side_names = [
            f'original (Intensity: {self.signal_intensity:.2f} dB)'
        ]
        right_side_names = [
            f'vocoded (Intensity: {self.vocoded_intensity:.2f} dB)'
        ]
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
            raise ValueError(
                'Either filename or self.filename must be provided'
            )
        if filename is None:
            filename = self.filename
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
        self.envelope = sp.extract_envelope(
            self.filtered_signal,
            cutoff,
            order,
            sample_rate,
            smoothing,
        )

    def plot_compare_spectrogram(self):
        '''plot the spectrogram of the original and vocoded signal'''
        from . import plot
        n = self.__repr__()
        plot.compare_spectrogram(self.filtered_signal, self.vocoded_signal, 
            sample_rate=self.sample_rate, names = [f'Filtered {n}', 'Vocoded'])

    def plot_filtered_signal(self, show_envelope = True):
        '''plot the signal'''
        from . import plot
        envelope = self.envelope if show_envelope else None
        plot.plot_signal(self.filtered_signal, self.sample_rate, 
            title=f'{self.path.name} - {self.__repr__()} filtered',
            envelope = envelope)

    def plot_vocoded_signal(self, show_envelope = True):
        '''plot the signal'''
        from . import plot
        envelope = self.envelope if show_envelope else None
        plot.plot_signal(self.vocoded_signal, self.sample_rate, 
            title=f'{self.path.name} - {self.__repr__()} vocoded',
            envelope = envelope)

    def plot_original_vocoded_signals(self):
        from . import plot
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
    def center_frequency(self):
        return np.sqrt(self.low_frequency * self.high_frequency)
            
    @property
    def vocoded_signal(self):
        '''return the vocoded signal'''
        if hasattr(self, '_vocoded_signal'): return self._vocoded_signal
        if self.parent.carrier_type == 'noise':
            carrier = sp.butterworth_bandpass_filter(
                self.white_noise,
                self.low_frequency,
                self.high_frequency,
                self.parent.sample_rate,
            )
        else:
            carrier = sp.sine_wave(
                self.center_frequency,
                len(self.signal),
                sample_rate=self.parent.sample_rate,
            )
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


if __name__ == '__main__':
    main()
