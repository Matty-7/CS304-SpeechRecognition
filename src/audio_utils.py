import math
import scipy.fftpack
import numpy as np
import wave
from plotting import *
from config import *
import os

def get_info(filename):

    full_filepath = os.path.join(os.pardir, "recordings", filename)

    # Open the audio file with wave module
    with wave.open(full_filepath, 'rb') as wave_file:
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
def segmenting(signal, segment_time, step_time, sample_rate):
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

# Pads a given signal with zeros to reach a specified target length
def zero_padding(signal, target_length):
    current_length = len(signal)
    if current_length >= target_length:
        return signal
    padding_length = target_length - current_length
    z = np.zeros(padding_length)
    padded_signal = np.append(signal, z)
    return padded_signal


def preprocessing(signal, sample_rate):

    # Framing
    frames=segmenting(signal,0.025,0.0125,sample_rate)
    plot_segment(frames,0,"original segment")

    # Pre-Emphasis
    emphasized_signal=emphasize(signal)
    emphasized_frames=segmenting(emphasized_signal,0.025,0.01,sample_rate)
    plot_segment(emphasized_frames,0,"emphasized segment")

    # Windowing
    windowed_frames=[windowing(frame) for frame in emphasized_frames]
    plot_segment(windowed_frames,0,"windowed segment")
    
    # Zero padding
    NFFT = 512
    padded_frames=[zero_padding(frame,NFFT) for frame in windowed_frames]
    plot_segment(padded_frames,0,"padded segment")
    return padded_frames

def power_spectrum(padded_frames):
    # FFT
    mag_frames = [np.absolute(np.fft.rfft(frame, NFFT)) for frame in padded_frames]
    pow_frames = [((1.0 / NFFT) * (frame ** 2)) for frame in mag_frames]
    plot_spectrum(pow_frames,0, "Power")
    return pow_frames

def mel_filter_banks(sample_rate,pow_frames):
    # Mel Filter Banks
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
    plot_mel_spectrum(filter_banks, 0, "40 Point")

    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    plot_mel_spectrum(filter_banks,0, "Log")
    return filter_banks
    
def mfcc(sample_rate,signal):
    # DCT
    filter_banks=mel_filter_banks(sample_rate,power_spectrum(preprocessing(signal,sample_rate)))

    #mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:14]
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')

    # Mean normalization
    # mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    normalized_mfccs = normalize_features(mfcc)

    plot_mel_cepstrum(normalized_mfccs, 0)

    return normalized_mfccs

def calculate_delta(mfccs):

    delta_mfccs = np.diff(mfccs, axis=0)
    delta_mfccs = np.concatenate([np.zeros((1, mfccs.shape[1])), delta_mfccs])
    normalized_delta_mfccs = normalize_features(delta_mfccs)

    return normalized_delta_mfccs

def calculate_delta_delta(delta_mfccs):
    # Compute delta-delta MFCC
    delta_delta_mfccs = np.diff(delta_mfccs, axis=0)
    delta_delta_mfccs = np.concatenate([np.zeros((1, delta_mfccs.shape[1])), delta_delta_mfccs])

    # Normalize delta-delta MFCC
    normalized_delta_delta_mfccs = normalize_features(delta_delta_mfccs)

    return normalized_delta_delta_mfccs

def normalize_features(features):
    
    mean_subtracted = features - np.mean(features, axis=0)
    normalized = mean_subtracted / (np.std(features, axis=0) + 1e-8)
    return normalized

def integrate_mfccs(sample_rate, signal):
    mfccs = mfcc(sample_rate, signal)
    delta_mfccs = calculate_delta(mfccs)
    delta_delta_mfccs = calculate_delta_delta(delta_mfccs)

    # Combine MFCC, delta MFCC, and delta-delta MFCC
    combined_features = np.concatenate((mfccs, delta_mfccs, delta_delta_mfccs), axis=1)

    return combined_features
    
def dtw(features1, features2):
    
    n = len(features1)
    m = len(features2)
    dtw_matrix = np.zeros((n+1, m+1))

    # 初始化无穷大的值
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf

    # 初始化第一个元素为0
    dtw_matrix[0, 0] = 0

    # 动态规划填表
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = np.linalg.norm(features1[i-1] - features2[j-1])
            # 取最小的DTW路径
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # 插入
                                          dtw_matrix[i, j-1],    # 删除
                                          dtw_matrix[i-1, j-1])  # 替换
    return dtw_matrix[n, m]

