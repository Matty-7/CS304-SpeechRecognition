import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import *
from PIL import Image

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

def plot_segment(frames,i,name):
    a=np.array(range(len(frames[i])))

    plt.style.use('dark_background')  # Set the background theme
    plt.figure(figsize=(10, 4))
    plt.plot(a,frames[i],label='Waveform', color = 'cyan')
    plt.title(f'{i+1}th Frame of the {name}')
    plt.ylabel("Amplitude")
    plt.xlabel('Sample Number')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.show()

def plot_spectrum(frames,i, name):
    a=np.array(range(len(frames[i])))

    plt.style.use('dark_background')  # Set the background theme
    plt.figure(figsize=(10, 4))
    plt.plot(a,frames[i],label='Spectrum', color = 'cyan')
    plt.title(f'{name} Spectrum of the {i+1}th Frame')
    plt.ylabel("Energy")
    plt.xlabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name} Spectrum of the {i+1}th Frame.png")
    plt.show()

def plot_mel_spectrum(filter_banks, i, name):
    frame = filter_banks[i, :]
    mel_points = np.array(range(len(frame)))

    plt.style.use('dark_background')  # Set the background theme
    plt.figure(figsize=(10, 4))
    plt.plot(mel_points, frame, color = 'cyan')
    plt.title(f'{name} Spectrum of the {i+1}th Frame')
    plt.xlabel('Mel Filter Bank')
    plt.ylabel('Magnitude (dB)')
    plt.tight_layout()
    plt.savefig(f'{name} Spectrum of the {i+1}th Frame')
    plt.grid(True)
    plt.show() 

def plot_mel_cepstrum(mfcc, i):

    mfcc_frame = mfcc[0, :13]

     # Plotting the Mel Cepstrum coefficients
    plt.figure(figsize=(12, 6))
    plt.scatter(range(13), mfcc_frame, alpha=0.5, color = 'yellow')
    plt.axhline(0, color='limegreen', lw=1)  # Add a horizontal line at y=0
    plt.title(f'Mel Cepstrum Coefficients of the {i+1}th Frame')
    plt.ylabel('Cepstral Coefficients')
    plt.xlabel('Frame Index')

    # Add vertical lines from each scatter point to y=0 in cyan color
    for j in range(13):
        plt.vlines(j, ymin=0, ymax=mfcc_frame[j], color='cyan')

    plt.grid(True)
    plt.savefig(f"Mel Cepstrum of the {i+1}th Frame")
    plt.show()

def plot_merge():
     
    image_files = ['original segment.png', 
                   'emphasized segment.png', 
                   'windowed segment.png', 
                   'padded segment.png', 
                   'Power Spectrum of the 1th Frame.png', 
                   '40 Point Spectrum of the 1th Frame.png', 
                   'Log Spectrum of the 1th Frame.png', 
                   'Mel Cepstrum of the 1th Frame.png']

    # Open the images
    images = [Image.open(img_file) for img_file in image_files]

    # Determine the size of the merged image
    widths, heights = zip(*(i.size for i in images))

    # Total dimensions for the merged image
    total_width = sum(widths[:4])
    total_height = sum(heights[0:2])

    # Create a new blank image with the total dimensions
    merged_image = Image.new('RGB', (total_width, total_height))

    # Initialize the starting position
    x_offset = 0
    y_offset = 0

    # Place images in two rows
    for i, img in enumerate(images):
        # For the second row, reset x_offset and set y_offset to the height of the first image
        if i == 4:
            x_offset = 0
            y_offset = heights[0]
        
        # Paste the current image into the merged image
        merged_image.paste(img, (x_offset, y_offset))
        x_offset += img.size[0]

    merged_image.save('merged_image.png')

def plot_spectrogram_from_mfcc(mfccs, sample_rate, num_mel_bins_list=[40, 30, 25], n_fft=512):
    """Plots a spectrogram from the MFCCs.

    Parameters:
    - mfccs: The MFCCs of the audio signal.
    - sample_rate: The sampling rate of the audio signal.
    - num_mel_bins: The number of Mel bins used to compute the MFCCs.
    - n_fft: The number of points used in the FFT transform.

    """

    for num_mel_bins in num_mel_bins_list:

        # Compute the inverse DCT to convert the MFCCs back to the log Mel spectrum
        log_mel_spectra = idct(mfccs, type=2, n=num_mel_bins, axis=-1, norm='ortho')

        plt.figure(figsize=(10, 4))
        plt.imshow(log_mel_spectra.T, aspect='auto', origin='lower',
                   extent=[0, mfccs.shape[0], 0, sample_rate / 2])
        plt.title(f'Spectrogram with {num_mel_bins} Mel Bins')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(f"spectrogram_{num_mel_bins}_mel_bins.png")
        plt.show()

def plot_cepstrum(cepstra, sample_rate, num_ceps):
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
    plt.savefig("cepstrum.png")
    plt.show()


