# main.py
from config import *
from audio_capture import *
from plotting import *
from audio_utils import *

# Handle the audio capture process
def main():
    audio, stream = start_audio_stream()
    frames, mfccs = capture_audio(stream)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded frames as a WAV file
    filename = 'captured_speech.wav'
    
    save_audio(frames, filename)
    print(f"Recording stopped and saved to '{filename}'")

    plot_waveform(filename)

if __name__ == '__main__':
    main()
    
