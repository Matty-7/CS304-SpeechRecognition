import matplotlib.pyplot as plt
import numpy as np
import wave

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
        
        # Plotting the waveform
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, color='cyan')
        plt.title('Waveform of the audio')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')
        plt.xlim(left=time[0], right=time[-1])
        plt.show()

def plot_spectrogram_from_mfcc(mfccs, sample_rate, num_mel_bins=40, n_fft=512):
    """Plots a spectrogram from the MFCCs.

    Parameters:
    - mfccs: The MFCCs of the audio signal.
    - sample_rate: The sampling rate of the audio signal.
    - num_mel_bins: The number of Mel bins used to compute the MFCCs.
    - n_fft: The number of points used in the FFT transform.

    This is a placeholder function and will need to be validated with actual data.
    """
    import scipy.fftpack
    import matplotlib.pyplot as plt

    # Compute the inverse DCT to convert the MFCCs back to the log Mel spectrum
    log_mel_spectra = scipy.fftpack.idct(mfccs, type=2, n=num_mel_bins, axis=-1, norm='ortho')

    # Optional: Convert the log Mel spectrum back to the log frequency spectrum (not implemented here)

    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(log_mel_spectra.T, aspect='auto', origin='lower',
               extent=[0, mfccs.shape[0], 0, sample_rate / 2])
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(format='%+2.0f dB')
    plt.show()