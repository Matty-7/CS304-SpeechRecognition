import pyaudio
import sys

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 if sys.platform == 'darwin' else 16000
CHUNK = 1024  # Buffer size for audio chunks
