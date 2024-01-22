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

    plot_waveform(filename)
    plot_spectrogram_from_mfcc(mfccs, RATE, num_mel_bins_list=[40, 30, 25])
    plot_cepstrum(mfccs, RATE, num_ceps=13)
 

if __name__ == '__main__':
    main()
    
