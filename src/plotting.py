import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import *
from PIL import Image
import os

def plot_waveform(filename):
    
    full_filepath = os.path.join(os.pardir, "recordings", filename)

    with wave.open(full_filepath, 'rb') as wave_file:
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
        plt.xlabel('Time (s)')
        plt.xlim(left=time[0], right=time[-1])

        file_path = os.path.join(os.pardir, "plots", "waveform.png")
        plt.savefig(file_path)
        # plt.show()
        plt.close()

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

    file_path = os.path.join(os.pardir, "plots", f"{name}.png")

    plt.savefig(file_path)
    # plt.show()
    plt.close()

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

    file_path = os.path.join(os.pardir, "plots", f"{name} Spectrum of the {i+1}th Frame.png")
    plt.savefig(file_path)
    # plt.show()
    plt.close()

def plot_mel_spectrum(filter_banks, i, name):

    frame = filter_banks[i, :]
    mel_points = np.array(range(len(frame)))

    plt.style.use('dark_background')  # Set the background theme
    plt.figure(figsize=(10, 4))
    plt.plot(mel_points, frame, color = 'cyan')
    plt.title(f'{name} Spectrum of the {i+1}th Frame')
    plt.xlabel('Mel Filter Bank')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(os.pardir, "plots", f'{name} Spectrum of the {i+1}th Frame')
    plt.savefig(file_path)
    
    # plt.show() 
    plt.close()

def plot_mel_cepstrum(mfcc, i):

    mfcc_frame = mfcc[0, :13]

     # Plotting the Mel Cepstrum coefficients
    plt.figure(figsize=(12, 6))
    plt.scatter(range(13), mfcc_frame, alpha=0.5, color = 'yellow')
    plt.axhline(0, color='limegreen', lw=1)  # Add a horizontal line at y=0
    plt.title(f'Mel Cepstrum Coefficients of the {i+1}th Frame')
    plt.ylabel('Cepstral Coefficients')
    plt.xlabel('Quefrency')

    # Add vertical lines from each scatter point to y=0 in cyan color
    for j in range(13):
        plt.vlines(j, ymin=0, ymax=mfcc_frame[j], color='cyan')

    plt.grid(True)

    file_path = os.path.join(os.pardir, "plots", f"Mel Cepstrum of the {i+1}th Frame.png")
    plt.savefig(file_path)
    # plt.show()
    plt.close()

def plot_merge():
     
    image_files = [os.path.join(os.pardir, "plots", img_file) for img_file in [
        'original segment.png',
        'emphasized segment.png',
        'windowed segment.png',
        'padded segment.png',
        'Power Spectrum of the 1th Frame.png',
        '40 Point Spectrum of the 1th Frame.png',
        'Log Spectrum of the 1th Frame.png',
        'Mel Cepstrum of the 1th Frame.png'
    ]]

    # Open the images
    images = [Image.open(img_file) for img_file in image_files]

    # Find the smallest width and height among all images
    min_width = min(img.size[0] for img in images)
    min_height = min(img.size[1] for img in images)

    # Find the size to which we can scale all images so that they have the same dimension
    # and are as large as possible within the constraints of the smallest image
    target_size = (min_width, min_height)

    # Scale all images to the target size
    scaled_images = [img.resize(target_size, Image.LANCZOS) for img in images]

    # The total width and height of the merged image (two rows of four images each)
    total_width = target_size[0] * 4
    total_height = target_size[1] * 2

    # Create a new blank image with the total dimensions
    merged_image = Image.new('RGB', (total_width, total_height))

    # Place images in two rows of four
    for i, img in enumerate(scaled_images):
        # Calculate x and y coordinates for the image
        x_offset = (i % 4) * target_size[0]
        y_offset = (i // 4) * target_size[1]
        
        # Paste the current image into the merged image
        merged_image.paste(img, (x_offset, y_offset))

    merged_image_path = os.path.join(os.pardir, "plots", "merged_image.png")
    merged_image.save(merged_image_path)

    plt.imshow(merged_image)
    plt.axis('off')
    # plt.show()
    plt.close()

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
        plt.xlabel('Time [0.01s]')
        plt.colorbar(format='%+2.0f dB')

        file_path = os.path.join(os.pardir, "plots", f"spectrogram_{num_mel_bins}_mel_bins.png")
        plt.savefig(file_path)
        plt.show()

def plot_mfccs(mfccs, title='MFCC Coefficient'):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs.T, aspect='auto', origin='lower', cmap='jet')
    plt.title(title)
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time(0.01s)')
    plt.colorbar()

    file_path = os.path.join(os.pardir, "plots", title)
    plt.savefig(file_path)
    plt.tight_layout()
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
    log_spectrum = idct(cepstra[:, 1:num_ceps], type=2, axis=1, norm='ortho')

    # Generate time axis for the frames
    time_frames = np.arange(log_spectrum.shape[0])
    
    # Generate cepstral coefficient axis
    cepstrum_coeffs = np.arange(log_spectrum.shape[1])

    plt.figure(figsize=(10, 2))
    plt.imshow(log_spectrum.T, aspect='auto', origin='lower',
               extent=[time_frames.min(), time_frames.max(), cepstrum_coeffs.min(), cepstrum_coeffs.max()])
    plt.title('Cepstrum')
    plt.ylabel('Cepstral Coefficients')
    plt.xlabel('Time(0.01s)')
    plt.tight_layout()
    plt.colorbar(label='Amplitude')

    file_path = os.path.join(os.pardir, "plots", "cepstrum.png")
    
    plt.savefig(file_path)
    plt.show()

def plot_pruning_threshold_vs_accuracy(prune_thresholds, accuracies, name):

    plt.style.use('dark_background')  # Set the background theme

    plt.figure(figsize=(10, 6))
    plt.plot(prune_thresholds, accuracies, marker='o', color='cyan')

    plt.title(f'Pruning Threshold vs Recognition Accuracy for {name}')
    plt.xlabel('Pruning Threshold')
    plt.ylabel('Recognition Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(os.pardir, "plots", f"Pruning Threshold vs Recognition Accuracy for {name}.png"))
    plt.show()




