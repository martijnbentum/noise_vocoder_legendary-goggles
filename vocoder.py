import audio
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import plot
import scipy
import signal_processing as sp

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
https://www.mrc-cbu.cam.ac.uk/personal/matt.davis/vocode/
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



six_bands= np.array([50,229,558,1161,2265,4290,7999])

class Vocoder:
    def __init__(self, signal = None, sample_rate = 16000, frequencies = None,
        filename = None, butterworth_order = 4, match_rms = True,):
        if signal is None and filename is None:
            raise ValueError('Either signal or filename must be provided')
        if filename: 
            signal, sample_rate = audio.load_audio_file(filename, sample_rate)
        self.filename = filename
        self.butterworth_order = butterworth_order
        self.match_rms = match_rms
        self.path = Path(filename) if filename else None
        self.signal = signal
        self.white_noise = sp.white_noise(n_samples=len(signal))
        self.sample_rate = sample_rate
        self.duration = len(signal) / sample_rate
        self.info = audio.soxinfo_to_dict(
            audio.soxi_info(filename)) if filename else None
        self._check_info()
        if frequencies is None: self.frequencies = six_bands
        else: self.frequencies = frequencies
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
            raise NotImplemented('Only mono files are supported')
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

    def _extract_envelope(self, cutoff=30, order=2, sample_rate=16000, 
        smoothing = True):
        '''extract the amplitude envelope of a signal using a low-pass filter.
        x                  the signal
        cutoff             the cutoff frequency of the low-pass filter
        order              the order of the low-pass filter
        sample_rate        the sample rate of the signal
        smoothing          whether to apply a moving average to the envelope
                            to smooth it 
        '''
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
        x = self.filtered_signal * self.white_noise
        x = sp.butterworth_bandpass_filter(x,
            self.low_frequency, self.high_frequency, self.parent.sample_rate)
        if self.parent.match_rms:
            x = sp.match_rms_by_window(self.filtered_signal, x)
        self._vocoded_signal = x
        return self._vocoded_signal
        
