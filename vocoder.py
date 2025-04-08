import audio
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import plot
import scipy
import signal_processing as sp

six_bands= np.array([50,229,558,1161,2265,4290,7999])

class Vocoder:
    def __init__(self, signal = None, sample_rate = 16000, frequencies = None,
        filename = None):
        if signal is None and filename is None:
            raise ValueError('Either signal or filename must be provided')
        if filename: 
            signal, sample_rate = audio.load_audio_file(filename, sample_rate)
        self.filename = filename
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
            m = f'Error: sample rate mismatch: '
            m += f'{self.info["sample_rate"]} != {self.sample_rate}'
            raise ValueError(m)
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
        x = sp.extract_envelope(self.signal, cutoff=30, order=2, 
            sample_rate=self.sample_rate, smoothing=True)
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
            self.low_frequency, self.high_frequency, self.parent.sample_rate)
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
        x = sp.match_rms_by_window(self.filtered_signal, x)
        self._vocoded_signal = x
        return self._vocoded_signal
        
