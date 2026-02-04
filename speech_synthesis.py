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

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        print(f"Output directory set to: {os.path.abspath(self.output_dir)}")

    def synthesize(self, text, filename="output.wav", speed=1.0, speaker=None):
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            filename: Output filename
            speed: Playback speed (0.5-2.0 recommended)
            speaker: Speaker ID (if supported by model)
        """
        output_path = os.path.join(self.output_dir, filename)

        # Check if XTTS model is being used
        if "xtts" in self.model_name.lower():
            # XTTS v2 specific parameters
            self.tts_api.tts_to_file(
                text=text,
                file_path=output_path,
                speaker="Ana Florence",  # or use speaker parameter
                language="en",
                speed=speed
            )
        else:
            # For other TTS models
            self.tts_api.tts_to_file(
                text=text,
                file_path=output_path,
                speed=speed
            )

        return filename

    def play_audio(self, filename="output.wav", playback_speed=1.0, text=None, is_synthesize=False):
        """
        Play audio file with adjustable speed.

        Args:
            filename: Audio file name
            playback_speed: Speed multiplier (0.5 = half speed, 2.0 = double speed)
            text: Text to synthesize if file doesn't exist
            is_synthesize: Force re-synthesis even if file exists
        """
        filepath = os.path.join(self.output_dir, filename)

        # Synthesize if file doesn't exist or forced
        if not os.path.exists(filepath) or is_synthesize:
            if text is None:
                raise ValueError("Text must be provided for synthesis")
            filename = self.synthesize(text=text, filename=filename, speed=playback_speed)
            filepath = os.path.join(self.output_dir, filename)
        else:
            # Apply speed adjustment to existing file if needed
            if playback_speed != 1.0:
                temp_filename = f"temp_speed_{playback_speed}_{filename}"
                temp_filepath = os.path.join(self.output_dir, temp_filename)

                # Load and adjust speed
                y, sr = librosa.load(filepath, sr=None)

                # Time stretch for speed adjustment
                y_stretched = librosa.effects.time_stretch(y, rate=1.0 / playback_speed)

                # Save temporary file
                sf.write(temp_filepath, y_stretched, sr)
                filepath = temp_filepath

        # Play the audio
        wf = wave.open(filepath, 'rb')
        p = pyaudio.PyAudio()

        # Obtain audio parameters
        n_channels = wf.getnchannels()
        original_rate = wf.getframerate()
        sample_width = wf.getsampwidth()

        # Calculate adjusted sample rate for playback speed
        adjusted_rate = int(original_rate * playback_speed)

        # Ensure adjusted rate is within valid range
        adjusted_rate = max(8000, min(adjusted_rate, 48000))  # Typical valid range

        stream = p.open(format=p.get_format_from_width(sample_width),
                        channels=n_channels,
                        rate=wf.getframerate(),
                        output=True)

        chunk = 1024
        data = wf.readframes(chunk)

        print(f"▶️ Playing audio at {playback_speed}x speed (Rate: {adjusted_rate}Hz)...")

        try:
            while len(data) > 0:
                stream.write(data)
                data = wf.readframes(chunk)
        except Exception as e:
            print(f"Error during playback: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
    def synthesize_and_save(self, text, filename="output.wav", speed=1.0):
        """Convenience method to only synthesize without playing."""
        return self.synthesize(text, filename, speed)


if __name__ == "__main__":

    tts = speechSynthesize(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.play_audio(
        text="Hi, my name is Jerry. Let's test the playback speed.",
        filename="output_new_1.wav",
        playback_speed=1,
    )
    speed_new = 1.1
    tts.play_audio(
        text="Hi.., my name... is Jerry. Let's test... the playback speed.",
        filename=f"output_new_{speed_new}.wav",
        playback_speed=float(speed_new),
        is_synthesize=True
    )