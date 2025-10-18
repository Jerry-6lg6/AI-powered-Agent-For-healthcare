import sounddevice as sd
import torch
import whisper
import numpy as np
import queue
import threading
import keyboard
import pyaudio
from TTS.api import TTS
import time
# # Microphone testing
# duration = 5  # seconds
# samplerate = 16000
# print("Recording...")
# audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
# sd.wait()
# print("Done.")



# CUDA checking
# print("CUDA available:", torch.cuda.is_available())
# print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


# Initialize Whisper
model = whisper.load_model("medium", device="cuda")

# Create an audio queue
q = queue.Queue()

samplerate = 16000
block_size = 1024  # smaller block = lower latency, higher CPU load

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# Start streaming from microphone
stream = sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback)
stream.start()

def transcribe_stream():
    buffer = np.zeros((0, 1), dtype=np.float32)
    segment_duration = 2  # seconds
    segment_samples = samplerate * segment_duration
    while True:
        # Collect small chunks
        while not q.empty():
            buffer = np.concatenate((buffer, q.get()), axis=0)

        # Only process if enough samples are accumulated
        if len(buffer) >= segment_samples:
            # Take exactly 2 seconds
            segment = buffer[:segment_samples]
            buffer = buffer[segment_samples:]  # remove processed part

            # Flatten (1D) for Whisper
            segment = segment.flatten()

            # Transcribe safely
            try:
                result = model.transcribe(segment, fp16=False, language='en')
                text = result["text"].strip()
                print("You said:", result["text"])
            except Exception as e:
                print("Transcription error:", e)

# def on_press(key):
#     try:
#         if key.char == "r" or key.char == 'R':
#             thread = threading.Thread(target=transcribe_stream, daemon=True)
#             thread.start()
#     except AttributeError:
#         pass
#
# def start_kb_listener():
#     with keyboard.Listener(on_press=on_press) as listener:
#         listener.join()


try:
    print("Press 'R' to start recording. Press Ctrl+C to exit.\n")
    while True:
        if keyboard.is_pressed('r'):
            print("🎙️ Starting transcription thread...")
            t = threading.Thread(target=transcribe_stream, daemon=True)
            t.start()

            # Wait until thread finishes (or user interrupts)
            t.join()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Exiting cleanly.")
    sd.stop()