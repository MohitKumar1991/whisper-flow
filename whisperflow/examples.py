
import numpy as np
import torch
import os
from datetime import datetime, timedelta
from queue import Queue

#from whisper_real_time
def split_on_time(audio_model, data_queue):
    phrase_time = None
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    now = datetime.now()
    phrase_timeout = 30
    transcription = ['']
    if not data_queue.empty():
        phrase_complete = False
        # If enough time has passed between recordings, consider the phrase complete.
        # Clear the current working audio buffer to start over with the new data.
        if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
            phrase_complete = True
        # This is the last time we received new audio data from the queue.
        phrase_time = now
        
        # Combine audio data from queue
        audio_data = b''.join(data_queue.queue)
        data_queue.queue.clear()
        
        # Convert in-ram buffer to something the model can use directly without needing a temp file.
        # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
        # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Read the transcription.
        result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = result['text'].strip()

        # If we detected a pause between recordings, add a new item to our transcription.
        # Otherwise edit the existing one.
        if phrase_complete:
            transcription.append(text)
        else:
            transcription[-1] = text

        # Clear the console to reprint the updated transcription.
        os.system('cls' if os.name=='nt' else 'clear')
        for line in transcription:
            print(line)
        # Flush stdout.
        print('', end='', flush=True)


