from ...src import audio_capture
def main():
    audio,stream=audio_capture.start_audio_stream()
    frames,mfcc=audio_capture.capture_audio(stream)
    audio_capture.save_audio(frames,"Recorded Audio")

def __init__():
    main()