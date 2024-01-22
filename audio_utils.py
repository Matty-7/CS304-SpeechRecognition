import math
import scipy.fftpack
import numpy as np
import wave

def get_info(filename):
    # Open the audio file with wave module
    with wave.open(filename, 'rb') as wave_file:
        sample_rate = wave_file.getframerate()  # Get the sample rate
        n_frames = wave_file.getnframes()  # Get the number of frames
        duration = n_frames / sample_rate  # Calculate the duration of the audio

    return sample_rate, n_frames, duration

def compute_energy(data):
    energy=10*math.log(sum(sample**2 for sample in data))
    return energy

def classifyFrame(audioframe,level,background):
    forgetfactor=2
    adjustment=0.05
    threshold=20
    current = compute_energy(audioframe)
    print(f'current: {current}')
    isSpeech = False
    level = ((level * forgetfactor) + current) / (forgetfactor + 1)
    
    if (current < background):
        background = current
    else:
        background += (current - background) * adjustment
    print(f'level: {level}')
    print(f'background: {background}')
    if (level < background):
        level = background
        
    if (level - background > threshold):
        
        isSpeech = True
    print(f'level1: {level}')
    print(f'background1: {background}')
    print(isSpeech)
    return isSpeech,level,background

def compute_mfcc(signal, sample_rate):

    # Pre-Emphasis
    emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])

    # Framing
    frame_length = 0.025  # Frame size in seconds
    frame_step = 0.01  # Frame stride in seconds
    frame_length_samples = int(round(frame_length * sample_rate))
    frame_step_samples = int(round(frame_step * sample_rate))
    num_frames = int(np.ceil(float(np.abs(len(emphasized_signal) - frame_length_samples)) / frame_step_samples))

    # Zero Padding
    pad_signal_length = num_frames * frame_step_samples + frame_length_samples
    z = np.zeros(pad_signal_length - len(emphasized_signal))
    padded_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length_samples), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step_samples, frame_step_samples), (frame_length_samples, 1)).T
    frames = padded_signal[indices.astype(np.int32, copy=False)]

    # Window
    frames *= np.hamming(frame_length_samples)

    # FFT and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    # Filter Banks
    low_freq_mel = 2595 * np.log10(1 + (133.33 / 700))
    high_freq_mel = 2595 * np.log10(1 + (6855.4976 / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, 40 + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((40, int(np.floor(NFFT / 2)) + 1))
    for m in range(1, 41):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # MFCCs
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:14]

    # Mean normalization
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc

