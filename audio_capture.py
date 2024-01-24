import pyaudio
import wave
import struct
from config import *
from audio_utils import *

# Initialize PyAudio and start the stream
def start_audio_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    return audio, stream

def capture_audio(stream):
    print("Hit Enter to start recording")
    input()
    print("Recording in progress...")
    frames = []
    
    count=0
    sum=0
    level=0
    background=0
    isRecording=True
    NoSpeechAfterSpeech=False
    while isRecording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        print(len(data))
        #frames.append(data)
        #data_fl = np.frombuffer(data, dtype=np.int16)
        data_fl = np.frombuffer(data, dtype=np.int16) if sys.platform == 'darwin' else [struct.unpack('h', data[i:i+2])[0] for i in range(0, len(data), 2)]  
        print(len(data_fl))
        if count==0:   
            level=compute_energy(data_fl)
        if count<=9:
            sum=sum+compute_energy(data_fl)
            count=count+1
        if count==10:
            background=sum//10
            count=count+1
            print(f'background0: {background}')
        if count>10:
            isSpeech,level,background=classifyFrame(data_fl,level,background)
            #print(f'level: {level}')
            #print(f'background: {background}')
            #print(f'isSpeech{isSpeech}')
            print(f'background0: {background}')
            
            if isSpeech:
                frames.append(data)
                silent_chunks = 0 
                NoSpeechAfterSpeech=True 
                silent_chunks=0# reset the silent_chunks counter as we've detected speech
            else:
                if isRecording:
                    if NoSpeechAfterSpeech:
                        frames.append(data)
            # If we've started recording and encounter silence, increment the counter
                        silent_chunks += 1
                        print(silent_chunks)
                    #print(silent_chunks)
            # If we've hit the silence threshold, consider it the end of speech
                        if silent_chunks > 20:
                            isRecording=False

    frames = np.array([np.frombuffer(frame, dtype=np.int16) for frame in frames])
    # Flatten the array to 1D before passing to compute_mfcc if needed
    frames = frames.flatten()
    print(frames)
    mfccs = compute_mfcc(frames, RATE)
        
    return frames, mfccs

def save_audio(frames, filename):
    wave_file = wave.open(filename, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(pyaudio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
