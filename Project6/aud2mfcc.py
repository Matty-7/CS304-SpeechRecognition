import os
import numpy as np
import librosa

def calculate_delta(features):
    """
    Calculate the first derivative of the features (MFCCs) using librosa.
    """
    return librosa.feature.delta(features)

def calculate_delta_delta(features):
    """
    Calculate the second derivative (delta-delta) of the features (MFCCs) using librosa.
    """
    return librosa.feature.delta(features, order=2)

def normalize_features(features):
    """
    Normalize features to have zero mean and unit variance.
    """
    mean_subtracted = features - np.mean(features, axis=0)
    normalized = mean_subtracted / (np.std(features, axis=0) + 1e-8)
    return normalized

def extract_mfcc_features(file_path, n_mfcc=13):
    """
    Extract MFCC and its derivatives from an audio file, and then normalize these features.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfccs = calculate_delta(mfccs)
    delta_delta_mfccs = calculate_delta_delta(mfccs)

    combined_features = np.concatenate((mfccs, delta_mfccs, delta_delta_mfccs), axis=0)
    normalized_features = normalize_features(combined_features.T)  # Transpose to have frames as rows and features as columns

    # Print the shape of the MFCC sequence
    print(f'Shape of MFCC sequence for {os.path.basename(file_path)}: {normalized_features.shape}')

    return normalized_features

def process_and_save_features(source_folder, target_folder):
    """
    Process all .wav files in the source_folder, extract MFCCs and their derivatives,
    normalize, and save them as .npy files in target_folder.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file in os.listdir(source_folder):
        if file.endswith('.wav'):
            file_path = os.path.join(source_folder, file)
            features = extract_mfcc_features(file_path)
            npy_filename = os.path.splitext(file)[0] + '.npy'
            npy_path = os.path.join(target_folder, npy_filename)
            np.save(npy_path, features)
            print(f'Saved MFCC features to {npy_path}')

if __name__ == "__main__":
    source_folder = 'new_recordings'  # Folder where the .wav files are located
    target_folder = 'features'  # Folder where the .npy files will be saved
    process_and_save_features(source_folder, target_folder)
