import pyaudio
import numpy as np
import librosa
import sys

# Constants for the audio stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def start_audio_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    return audio, stream

def compute_energy(data_fl):
    # Dummy energy computation. Replace with actual energy computation logic
    return np.sum(data_fl**2) / len(data_fl)

def classifyFrame(data_fl, level, background):
    # Assuming compute_energy returns an energy level for the frame
    current_level = compute_energy(data_fl)
    isSpeech = current_level > 1.5 * background
    print(f"Current level: {current_level}, Background: {background}, Is speech: {isSpeech}")
    return isSpeech, current_level, background

def capture_audio(stream):
    print("Hit Enter to start recording")
    input()
    print("Recording in progress...")
    frames = []
    
    count = 0
    sum_energy = 0
    level = 0
    background = 0
    isRecording = True
    NoSpeechAfterSpeech = False
    silent_chunks = 0

    while isRecording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        data_fl = np.frombuffer(data, dtype=np.int16) if sys.platform == 'darwin' else np.array([struct.unpack('h', data[i:i+2])[0] for i in range(0, len(data), 2)])  

        if count < 10:
            sum_energy += compute_energy(data_fl)
            count += 1
        elif count == 10:
            background = sum_energy / 10
            count += 1
            print(f'Background level: {background}')

        if count > 10:
            isSpeech, level, background = classifyFrame(data_fl, level, background)
            
            if isSpeech:
                frames.append(data)
                silent_chunks = 0  # Reset silent_chunks counter on speech detection
                NoSpeechAfterSpeech = True
            else:
                if NoSpeechAfterSpeech:
                    frames.append(data)
                    silent_chunks += 1  # Increment silent_chunks counter on silence detection
                    print(f"Silent chunks count: {silent_chunks}")
                    
                    if silent_chunks > 10:  # Reduced threshold for quicker detection
                        print("Detected end of speech based on silence. Stopping recording.")
                        isRecording = False

    frames = frames[:-10]  # Adjust to match the new threshold if needed
    frames = np.hstack([np.frombuffer(frame, dtype=np.int16) for frame in frames])
    normalized_frames = frames.astype(np.float32) / np.iinfo(np.int16).max
    mfccs = librosa.feature.mfcc(y=normalized_frames, sr=RATE, n_mfcc=13)
        
    return frames, mfccs

def calculate_delta(mfccs):
    # Calculate the first derivative (delta) of MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    return delta_mfccs

def calculate_delta_delta(mfccs):
    # Calculate the second derivative (delta-delta) of MFCCs
    delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
    return delta_delta_mfccs

def normalize_features(features):
    # Normalize features to have zero mean and unit variance
    mean_subtracted = features - np.mean(features, axis=0)
    normalized = mean_subtracted / (np.std(features, axis=0) + 1e-8)
    return normalized

def integrate_mfccs(sample_rate, signal):
    # Compute MFCCs from the signal
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    
    # Calculate first and second derivatives of MFCCs
    delta_mfccs = calculate_delta(mfccs)
    delta_delta_mfccs = calculate_delta_delta(mfccs)

    # Combine MFCC, delta MFCC, and delta-delta MFCC along the feature axis
    combined_features = np.concatenate((mfccs, delta_mfccs, delta_delta_mfccs), axis=0)
    
    # Transpose the combined feature matrix to have frames as rows and features as columns
    combined_features_transposed = combined_features.T
    
    # Normalize the combined feature set
    normalized_combined_features = normalize_features(combined_features_transposed)

    return normalized_combined_features

# Example usage for live audio recording and feature extraction
if __name__ == "__main__":
    audio, stream = start_audio_stream()
    print("Please start speaking into the microphone.")
    frames, mfccs = capture_audio(stream)
    
    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # The mfccs computed from capture_audio are already in the shape (n_mfcc, n_frames),
    # so you can directly proceed with calculating deltas and combining features
    delta_mfccs = calculate_delta(mfccs)
    delta_delta_mfccs = calculate_delta_delta(mfccs)
    
    # Combine and normalize features
    combined_normalized_features = integrate_mfccs(RATE, frames.astype(np.float32) / np.iinfo(np.int16).max)
    
    print("Shape of the combined feature set:", combined_normalized_features.shape)
    # Expected output: (N, 39) where N is the number of frames
    
    # Ensure there are frames recorded before attempting to print features
    if combined_normalized_features.shape[0] > 0:
        print("First frame's combined features (39 dimensions):", combined_normalized_features[0, :])
    else:
        print("No audio frames were recorded.")