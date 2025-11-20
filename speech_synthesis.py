import os
import wave
import librosa
import sounddevice as sd
import soundfile as sf
import torch
import whisper
import numpy as np
import queue
import threading
from TTS.api import TTS
import pyaudio

class speechSynthesize:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True, output_dir="audio/question"):
        self.model_name = model_name
        self.gpu = gpu

        self.output_dir = output_dir

        # Loading TTS model
        print(f"Loading TTS model: {model_name} (GPU={gpu})...")
        self.tts_api = TTS(model_name, gpu=gpu)
        # Ensure the loading address
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        print(f"Output directory set to: {os.path.abspath(self.output_dir)}")

    def synthesize(self, text, filename="output.wav", speed=1.0, adjust_speed=0, speaker="en_1"):
        output_path = os.path.join(self.output_dir, filename)
        print(self.tts_api.config)
        self.tts_api.tts_to_file(
            speaker="Ana Florence",
            text=text,
            file_path=output_path,
            language="en",
            speed=speed
        )
        if adjust_speed != 0:
            y, sr = librosa.load(output_path, sr=None)
            y_stretched = librosa.effects.time_stretch(y, rate=speed)
            sf.write(output_path, y_stretched, sr)

        return filename

    def play_audio(self, filename="output.wav", playback_speed=1.0, text=None, is_synthesize=False):
        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath) or is_synthesize:
            filename_0 = self.synthesize(text=text, filename=filename, speed=playback_speed)
            filepath = os.path.join(self.output_dir, filename_0)

        wf = wave.open(filepath, 'rb')
        p = pyaudio.PyAudio()

        # Obtain audio parameters
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()

        # Calculate playing speed
        adjusted_rate = int(sample_rate * 1.0)

        stream = p.open(format=p.get_format_from_width(sample_width),
                        channels=n_channels,
                        rate=adjusted_rate,
                        output=True)

        chunk = 1024
        data = wf.readframes(chunk)

        print(f"▶️ Playing audio at {playback_speed}x speed...")
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

if __name__ == "__main__":

    tts = speechSynthesize(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.play_audio(
        text="Hi, my name is Jerry. Let's test the playback speed.",
        filename="output_new_2.wav",
        playback_speed=2,
    )

    tts.play_audio(
        text="Hi, my name is Jerry. Let's test the playback speed.",
        filename="output_new_0.1.wav",
        playback_speed=0.1,
    )