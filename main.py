from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

def main():
    audio, stream = start_audio_stream()
    frames, mfccs = capture_audio(stream)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    filename = 'captured_speech.wav'
    save_audio(frames, filename)
    print(f"Recording stopped and saved to '{filename}'")

    # Call the get_info function from audio_utils.py
    sample_rate, n_frames, duration = get_info(filename)

    # Print out the audio file properties
    print(f"Recording stopped and saved to '{filename}'")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Number of frames: {n_frames}")
    print(f"Duration: {duration:.2f} seconds")

    plot_waveform(filename)

    plot_spectrogram_from_mfcc(mfccs, RATE, num_mel_bins_list=[40, 30, 25])
    plot_cepstrum(mfccs, RATE, num_ceps=13)
 

if __name__ == '__main__':
    main()
    
