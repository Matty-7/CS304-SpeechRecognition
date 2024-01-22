import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import *

def plot_waveform(filename):
    
    with wave.open(filename, 'rb') as wave_file:
        # Extract Raw Audio 
        signal = wave_file.readframes(-1)
        # Convert binary data to integers
        signal = np.frombuffer(signal, dtype='int16')
        
        # Get the frame rate
        framerate = wave_file.getframerate()
        
        # Time axis in seconds
        time = np.linspace(0, len(signal) / framerate, num=len(signal))

        plt.style.use('dark_background')  # Set the background theme
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, color='cyan')
        plt.title('Waveform of the audio')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.xlim(left=time[0], right=time[-1])
        plt.savefig("waveform.png")
        plt.show()

def plot_spectrogram_from_mfcc(mfccs, sample_rate, num_mel_bins=40, n_fft=512, filename = 'spectrogram.png'):
    """Plots a spectrogram from the MFCCs.

    Parameters:
    - mfccs: The MFCCs of the audio signal.
    - sample_rate: The sampling rate of the audio signal.
    - num_mel_bins: The number of Mel bins used to compute the MFCCs.
    - n_fft: The number of points used in the FFT transform.

    """

    # Compute the inverse DCT to convert the MFCCs back to the log Mel spectrum
    log_mel_spectra = idct(mfccs, type=2, n=num_mel_bins, axis=-1, norm='ortho')

    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_spectra.T, aspect='auto', origin='lower',
               extent=[0, mfccs.shape[0], 0, sample_rate / 2])
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig(filename)
    plt.show()

def plot_cepstrum(cepstra, sample_rate, num_ceps, filename='cepstrum.png'):
    """
    Plot the cepstrum of an audio signal.
    :param cepstra: The MFCCs of the audio signal.
    :param sample_rate: The sample rate of the audio signal.
    :param num_ceps: Number of cepstral coefficients to plot.
    :param filename: The file name to save the plot.
    """

    # Use IDCT to convert cepstra back to log spectrum
    log_spectrum = idct(cepstra, type=2, axis=1, norm='ortho')[:num_ceps]

    # Generate time axis for the frames
    time_frames = np.arange(log_spectrum.shape[0])
    
    # Generate cepstral coefficient axis
    cepstrum_coeffs = np.arange(log_spectrum.shape[1])

    plt.figure(figsize=(12, 8))
    plt.imshow(log_spectrum.T, aspect='auto', origin='lower',
               extent=[time_frames.min(), time_frames.max(), cepstrum_coeffs.min(), cepstrum_coeffs.max()])
    
    plt.title('Cepstrum')
    plt.ylabel('Cepstral Coefficients')
    plt.xlabel('Frame number')
    plt.colorbar(label='Amplitude')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()