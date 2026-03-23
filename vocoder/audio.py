import librosa
import soundfile as sf

def _load_sounddevice():
    '''Import sounddevice only when playback is requested.'''
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            'Audio playback requires sounddevice with a working PortAudio '
            'installation.'
        ) from exc
    return sd


def load_audio_file(file_path, sample_rate = 16000, start = 0.0, end = None):
    '''load an audio file and return the signal and sample rate'''
    if end:
        duration = end - start
    else: duration = None
    signal, sr = librosa.load(file_path, sr=sample_rate, offset=start,
        duration=duration)
    return signal, sr

def time_to_samples(time, sr):
    '''convert time to samples'''
    return int(time * sr)

def select_samples(signal, sr, start, end):
    start = time_to_samples(start, sr)
    end = time_to_samples(end, sr)
    return signal[start:end]

def audio_info(filename):
    '''Return basic audio metadata for a file.'''
    info = sf.info(filename)
    d = {}
    d['filename'] = str(filename)
    d['n_channels'] = info.channels
    d['sample_rate'] = info.samplerate
    d['duration'] = info.frames / info.samplerate
    return d

def play_audio(signal, sample_rate = 16000):
    '''play audio signal'''
    sd = _load_sounddevice()
    sd.play(signal, sample_rate)
    sd.wait()

def write_audio(signal, file_path, sample_rate = 16000):
    '''write audio signal to file'''
    sf.write(file_path, signal, sample_rate)
    return file_path
