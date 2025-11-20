import wave
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
import pygame
import os
from faster_whisper import WhisperModel
class beeper:
    def __init__(self, sample_rate=44100, buffer_size=1024, cache_dir=r"audio\beep"):
        self.filename = "beep.wav"
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        filename = os.path.join(self.cache_dir, self.filename)
        pygame.mixer.pre_init(sample_rate, -16, 1, buffer_size)
        pygame.init()

        self.sample_rate = sample_rate
        self.buffer_size = buffer_size



    def generate_beep(self, frequency=440, duration=0.5, volume=0.5, save_to_cache=True):
        n_samples = int(round(duration * self.sample_rate))
        buf = np.zeros((n_samples, 1), dtype=np.float32)

        for i in range(n_samples):
            fade_in = min(1.0, i / (self.sample_rate * 0.01))   # 10ms fade-in
            fade_out = min(1.0, (n_samples - i) / (self.sample_rate * 0.01))  # 10ms fade-out
            fade = fade_in * fade_out
            buf[i][0] = volume * fade * np.sin(2.0 * np.pi * frequency * i / self.sample_rate)

        # Convert to 16-bit signed integer
        buf_int16 = (buf * 32767.0).astype(np.int16)
        buf_stereo = np.repeat(buf_int16, 2, axis=1)
        sound = pygame.sndarray.make_sound(buf_stereo)

        if save_to_cache:
            filename = os.path.join(self.cache_dir, self.filename)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(buf_int16.tobytes())

        return sound


    def play_beep(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=2, buffer=self.buffer_size)
        filename = os.path.join(self.cache_dir, self.filename)

        if os.path.exists(filename):

            print(filename)
            try:
                sound = pygame.mixer.Sound(filename)
            except Exception as e:
                print(f"Loading audio file fail: {e}")
                return None
        else:
            sound = self.generate_beep()

        sound.play()
        return 0



class speechRecognizer:
    def __init__(self, model_name="whisper", model_size="medium", device="cuda"):
        self.model_name = model_name
        print(f"Loading moodel{model_name}_{model_size}")
        if model_name == "whisper":
            self.model = whisper.load_model(model_size, device=device)
            print("Model loaded successfully")
        if model_name == "faster_whisper":
            self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.beeper = beeper()


    def record_audio(self, duration=10, samplerate=16000):
        print(f"Recording for {duration} seconds... Speak after hearing the beep!")
        time.sleep(0.5)
        self.beeper.play_beep()
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
        sd.wait()
        print("Recording finished.")
        text = ""
        audio = audio.flatten()
        time_0 = 0.0
        time_1 = 0.0

        if self.model_name == "whisper":
            try:
                # Flatten it to 1D for whisper

                result = self.model.transcribe(audio, language='en')
                text = result["text"].strip()
                print("🗣 You said:", text)
            except Exception as e:
                print("❌ Transcription error:", e)

        if self.model_name == "faster_whisper":
            segments, _ = self.model.transcribe(audio, word_timestamps=True, language="en")
            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                time_0 = segment.start
                time_1 = segment.end
                text = segment.text

        return text, time_0, time_1



if __name__ == "__main__":
    # b = beeper()
    # b.play_beep()
    # time.sleep(3)

    sr = speechRecognizer(model_name="faster_whisper")
    print("Press 'R' to start recording. Press Ctrl+C to exit.\n")

    try:
        while True:
            if keyboard.is_pressed('r'):
                print("🎙️ Starting transcription thread...")
                t = threading.Thread(target=sr.record_audio, daemon=True)
                t.start()
                t.join()
                time.sleep(0.5)  # debounce
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Exiting cleanly.")
        sd.stop()