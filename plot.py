from matplotlib import pyplot as plt
import numpy as np
import signal_processing as sp

def compare_spectrogram(x, y, sample_rate = 16000, 
    names = ['Original', 'Modified']):
    '''Plots spectrogram of x, the original, and y, the modified signal.
    x                  the original signal
    y                  the modified signal
    sample_rate        the sample rate of the signal
    '''
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 4))

    p_x, *_, im  = axes[0].specgram(x, cmap='Blues', Fs=sample_rate)
    vmin, vmax = 10 * np.log10([np.min(p_x), np.max(p_x)])
    _ = axes[1].specgram(y, cmap='Blues', Fs=sample_rate, vmin=vmin, vmax=vmax)
    axes[0].set_title(names[0])
    axes[1].set_title(names[1])
    for ax in axes:
        ax.set_xlabel('Time (s)')
        ax.grid(False)
    axes[0].set_ylabel('Frequency (Hz)')
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Magnitude (dB)')
    fig.tight_layout()

def plot_power_spectrum(x, sample_rate = 16000):
    '''plot the power spectrum of a signal
    x                  the signal
    sample_rate        the sample rate of the signal
    '''
    frequencies, power_spectrum = sp.compute_power_spectrum(x, sample_rate)
    plt.ion()
    plt.clf()
    plt.plot(frequencies, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(alpha=0.3)
    plt.show()

def plot_signal(x, sample_rate = 16000, title = 'Signal', envelope = None,
    ax = None, show_legend = True, show_xaxis = True, show_yaxis = True,
    ylim = None):
    '''plot the signal
    x                  the signal
    sample_rate        the sample rate of the signal
    title             the title of the plot
    '''
    plt.ion()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    seconds = np.arange(len(x))/ sample_rate
    ax.plot(seconds, x, color='royalblue', alpha= 0.5, label='Signal')
    if envelope is not None:
        ax.plot(np.arange(len(x)) / sample_rate, envelope, color='red', 
            alpha=0.9, label='Envelope')
    ax.grid(True, axis = 'both', linestyle = '--',alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_xaxis:
        ax.set_xlabel('Time (s)')
    else: ax.set_xticklabels([])
    if show_yaxis:
        ax.set_ylabel('Amplitude')
    else: ax.set_yticklabels([])
    if show_legend:
        ax.legend(loc='upper right')
    ax.set_title(title)
    plt.show()

def plot_stacked_sigals(signals, names, envelopes = None, title = '',
    sample_rate = 16000):
    '''plot the signals in a stacked manner
    signals            the signals to plot
    names              the names of the signals
    envelopes          the envelopes of the signals
    sample_rate        the sample rate of the signals
    '''
    ylim = sp.find_min_max_for_signal_list(signals)
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 4))
    for i, signal in enumerate(signals):
        show_legend = True if i == 0 else False
        show_xaxis = True if i == len(signals) - 1 else False
        show_yaxis = True if i == len(signals) - 1 else False
        plot_signal(signal, sample_rate, title=names[i], envelope=envelopes[i],
            ax=axes[i], show_legend = show_legend, show_xaxis = show_xaxis,
            show_yaxis = show_yaxis, ylim = ylim, )
    plt.suptitle(title)
    plt.show()

