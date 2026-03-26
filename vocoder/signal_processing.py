import numpy as np
import scipy

def extract_kaiser_20_envelope(x, kaiser_beta = 20, sample_rate = 16000):
    # x = x**2
    x = np.abs(x)
    window_size = int(sample_rate * 0.064)
    window = np.kaiser(window_size, kaiser_beta)
    window /= np.sum(window)
    return np.convolve(x, window, mode='same') 
    

def extract_envelope(x, cutoff=30, order=1, sample_rate=16000, 
    smoothing = True):
    '''extract the amplitude envelope of a signal using a low-pass filter.
    x                  the signal
    cutoff             the cutoff frequency of the low-pass filter
    order              the order of the low-pass filter
    sample_rate        the sample rate of the signal
    smoothing          whether to apply a moving average to the envelope
                       to smooth it 
    '''
    envelope = np.abs(x)
    b, a = scipy.signal.butter(order, cutoff * 2 / sample_rate)
    envelope = scipy.signal.lfilter(b, a, envelope)
    if smoothing:
        envelope = moving_average(envelope, 100)
    return envelope

def moving_average(x, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(x, window, mode='same')

def butterworth_bandpass_filter(x, low_frequency, high_frequency, 
    sample_rate = 16000, order = 4):
    '''apply a bandpass butterworth filter to a signal
    x                  the signal
    low_frequency      the low frequency of the bandpass filter
    high_frequency     the high frequency of the bandpass filter
    sample_rate        the sample rate of the signal
    order              the order of the bandpass filter
    '''
    low_high = np.array([low_frequency, high_frequency])
    b, a =scipy.signal.butter(order, low_high * 2 / sample_rate, 
        btype="bandpass")
    return scipy.signal.lfilter(b, a, x)

def compute_fft(x, sample_rate = 16000):
    '''compute the fast fourier transform of a signal
    x                  the signal
    sample_rate        the sample rate of the signal
    returns only the positive frequencies
    frequencies         a list of frequencies corresponding to the fft_result
    fft_result          a list of complex numbers -> fourier decomposition
                        of the signal
    '''

    fft_result= np.fft.fft(x)
    frequencies = np.fft.fftfreq(len(x), 1.0/sample_rate)
    frequencies = frequencies[:int(len(frequencies)/2)]
    fft_result = fft_result[:int(len(fft_result)/2)]
    return frequencies, fft_result

def compute_power_spectrum(x, sample_rate = 16000):
    '''compute the power spectrum of a signal
    x                  the signal
    sample_rate        the sample rate of the signal
    '''
    frequencies, fft_result = compute_fft(x, sample_rate)
    # the factor of 4 is to account for the fact that we only use the positive
    # frequencies
    power_spectrum = 10 * np.log10(4 * np.abs(fft_result)**2)
    return frequencies, power_spectrum


def white_noise(n_samples = None, duration = None, sample_rate = 16000):
    '''generate white noise
    duration           the duration of the noise in seconds
    sample_rate        the sample rate of the noise
    '''
    if n_samples is None and duration is None:
        raise ValueError('Either n_samples or duration must be specified')
    if n_samples is None:
        n_samples = int(duration * sample_rate)
    white_noise = np.random.normal(0, 1, n_samples)
    return white_noise


def sine_wave(
    frequency,
    n_samples,
    sample_rate = 16000,
    phase = 0.0,
):
    '''Generate a sine wave with one fixed carrier frequency.'''
    time = np.arange(n_samples) / sample_rate
    return np.sin(2 * np.pi * frequency * time + phase)


def log_spaced_band_frequencies(
    low_frequency,
    high_frequency,
    n_frequencies = 3,
    edge_fraction = 0.2,
):
    '''Return interior log-spaced carrier frequencies for one band.'''
    if n_frequencies < 1:
        raise ValueError('n_frequencies must be at least 1')
    if low_frequency <= 0 or high_frequency <= low_frequency:
        raise ValueError('Band frequencies must be positive and increasing')
    log_low = np.log(low_frequency)
    log_high = np.log(high_frequency)
    log_span = log_high - log_low
    inner_low = log_low + edge_fraction * log_span
    inner_high = log_high - edge_fraction * log_span
    return np.exp(np.linspace(inner_low, inner_high, n_frequencies))


def normalize_rms(signal, target_rms = 1.0):
    '''Rescale one signal to the requested RMS level.'''
    rms = np.sqrt(np.mean(signal ** 2))
    if rms == 0:
        return signal
    return signal * (target_rms / rms)

def moving_rms(signal, window_size = 1000):
    """Centered moving RMS."""
    pad = window_size // 2
    padded = np.pad(signal, (pad, pad), mode='reflect')
    squared = padded**2
    window = np.ones(window_size) / window_size
    x = np.sqrt(np.convolve(squared, window, mode='valid'))
    return x[:len(signal)]

def match_rms_by_window(source, target, window_size = 1000):
    """Rescale `target` so its moving RMS matches that of `source`."""
    eps = 1e-10  # avoid divide by zero
    source_rms = moving_rms(source, window_size)
    target_rms = moving_rms(target, window_size)
    gain = source_rms / (target_rms + eps)
    return target * gain

def compute_praat_intensity(signal):
    '''
    compute the intensity of a signal using praat's intensity algorithm
    '''
    baseline = 4 * 10 ** -10
    power = np.mean(signal ** 2)
    return 10 * np.log10(power / baseline)

def find_min_max_for_signal_list(signals):
    minimum = min(signals[0])
    maximum = max(signals[0])
    for signal in signals:
        temp = np.min(signal)
        if temp < minimum:
            minimum = temp
        temp = np.max(signal)
        if temp > maximum:
            maximum = temp
    return minimum, maximum
    
