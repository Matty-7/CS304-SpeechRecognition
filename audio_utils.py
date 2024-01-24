import math
import scipy.fftpack
import numpy as np
import wave
from plotting import *

def get_info(filename):
    # Open the audio file with wave module
    with wave.open(filename, 'rb') as wave_file:
        sample_rate = wave_file.getframerate()  
        n_frames = wave_file.getnframes()  
        duration = n_frames / sample_rate  

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

# Enhance higher frequencies
def emphasize(signal):
    emphasized_signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    return emphasized_signal

# Framing into short frames
def segmenting(signal,segment_time,step_time,sample_rate):
    segment_samples = int(round(segment_time * sample_rate))
    segment_step_samples = int(round(step_time * sample_rate))
    num_segment = int(np.ceil(float(np.abs(len(signal) - segment_samples)) / segment_step_samples))
    segmented_signal=[]
    signal=np.array(signal)
    for i in range(num_segment):
        b=signal[i*segment_step_samples:i*segment_step_samples+segment_samples]
        segmented_signal.append(b)
    segmented_signal=np.array(segmented_signal)
    return segmented_signal


def zero_padding(signal,target_length):
    current_length = len(signal)
    if current_length >= target_length:
        return signal
    padding_length = target_length - current_length
    z = np.zeros(padding_length)
    padded_signal = np.append(signal, z)
    return padded_signal

# Minimize the discontinuities at the beginning and end of each frame
def windowing(segment,mode="Hamming"):
    n=np.array(range(0,len(segment)))
    M=len(segment)
    if mode=="Hamming":
        return segment*(0.54 - 0.46 * np.cos(2 * np.pi * n / M))
    elif mode=="Hanning":
        return segment*(0.5 - 0.5 * np.cos(2 * np.pi * n / M))
    elif mode=="Blackman":
        return segment*( 0.42 - 0.5 * np.cos(2 * np.pi * n / M) + 0.08 * np.cos(4 * np.pi * n / M))

def compute_mfcc(signal, sample_rate):

    # Framing
    frames=segmenting(signal,0.025,0.01,sample_rate)
    plot_segment(frames,0,"original segment")

    # Pre-Emphasis
    emphasized_signal=emphasize(signal)
    emphasized_frames=segmenting(emphasized_signal,0.025,0.01,sample_rate)
    plot_segment(emphasized_frames,0,"emphasized segment")

    # Windowing
    windowed_frames=[windowing(frame) for frame in emphasized_frames]
    plot_segment(windowed_frames,0,"windowed segment")
    
    # FFT and Power Spectrum
    NFFT = 512
    print(len(windowed_frames[0]))
    padded_frames=[zero_padding(frame,NFFT) for frame in windowed_frames]
    plot_segment(padded_frames,0,"padded segment")
    mag_frames = [np.absolute(np.fft.rfft(frame, NFFT)) for frame in padded_frames]
    
    pow_frames = [((1.0 / NFFT) * (frame ** 2)) for frame in mag_frames]
    plot_spectrum(mag_frames,0)
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
    
    
    plot_mel_spectrum(filter_banks,0)

    # MFCCs
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:14]

    # Mean normalization
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc

