import wave
from collections import deque
import sounddevice as sd
import torch
import whisper
import numpy as np
import threading
import keyboard
import time
import pygame
from faster_whisper import WhisperModel
from clearvoice import ClearVoice
import os
os.environ['TORCH_HOME'] = r'C:\torch_cache'

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
            fade_in = min(1.0, i / (self.sample_rate * 0.01))  # 10ms fade-in
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
        torch.set_num_threads(1)
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )

    def listen_keyword(self, keywords, timeout=1, samplerate=16000):
        audio = sd.rec(int(timeout * samplerate), samplerate=samplerate,
                       channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        if np.abs(audio).mean() < 0.015:
            return None

        if self.model_name == "faster_whisper":
            segments, _ = self.model.transcribe(audio, language="en")

            for seg in segments:
                print("[%.2fs -> %.2fs] %s" % (seg.start, seg.end, seg.text))
                text = seg.text.lower()
                for k in keywords:
                    if k in text:
                        return k
        return None

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


class RobustSpeechRecognizer(speechRecognizer):
    def __init__(self, *args, plugins=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.plugins = plugins or []

    def apply_plugins(self, audio, sr):
        for plugin in self.plugins:
            audio = plugin.process(audio, sr)
        return audio

    def record_audio(self, max_duration=10, samplerate=16000, silence_threshold=1.5, vad_free_duration=4.0):
        text, t0, t1 = super().record_audio(max_duration, samplerate)
        return text, t0, t1


class StreamRobustRecognizer(speechRecognizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # silero vad
        self.vad_model = load_silero_vad()

        self.sample_rate = 16000
        self.chunk_duration = 0.5
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        self.energy_threshold = 0.01
        self.silence_limit = 1.2

        self.vote_window = deque(maxlen=3)

        self.rn_model = ClearVoice(task='speech_enhancement', model_names=['MossFormerGAN_SE_16K'])

    def record_stream(self, max_duration=10):

        print("🎙️ Streaming...")

        audio_buffer = []
        silence_counter = 0
        total_duration = 0

        while total_duration < max_duration:

            chunk = sd.rec(
                self.chunk_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            chunk = chunk.flatten()

            total_duration += self.chunk_duration

            # ① Energy Gate
            if np.abs(chunk).mean() < self.energy_threshold:
                silence_counter += self.chunk_duration
                if silence_counter > self.silence_limit:
                    break
                continue

            # ② VAD
            speech_dict = get_speech_timestamps(
                torch.from_numpy(chunk),
                self.vad_model,
                sampling_rate=self.sample_rate
            )

            if len(speech_dict) == 0:
                silence_counter += self.chunk_duration
                if silence_counter > self.silence_limit:
                    break
                continue

            silence_counter = 0

            # ③ Noise Reduction
            chunk = self.dynamic_batch_process(chunk)

            audio_buffer.append(chunk)

        if len(audio_buffer) == 0:
            return "", 0, 0

        full_audio = np.concatenate(audio_buffer)

        return self.transcribe_with_filter(full_audio)

    def transcribe_with_filter(self, audio):

        segments, info = self.model.transcribe(
            audio,
            language="en",
            word_timestamps=True
        )

        final_text = ""
        avg_logprob = []

        for seg in segments:
            final_text += seg.text
            avg_logprob.append(seg.avg_logprob)

        if len(avg_logprob) == 0:
            return "", 0, 0

        confidence = np.mean(avg_logprob)

        # 🔴 Confidence level filtering
        if confidence < -1.2:
            return "", 0, 0

        return final_text.strip(), 0, 0

    def listen_interrupt_stream(self, keywords):

        chunk = sd.rec(
            self.chunk_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        chunk = chunk.flatten()

        if np.abs(chunk).mean() < self.energy_threshold:
            return None

        text, _, _ = self.transcribe_with_filter(chunk)

        detected = 0
        for k in keywords:
            if k in text.lower():
                detected = 1

        self.vote_window.append(detected)

        if sum(self.vote_window) >= 2:
            self.vote_window.clear()
            return text

        return None


if __name__ == "__main__":
    # b = beeper()
    # b.play_beep()
    # time.sleep(3)

    sr = RobustSpeechRecognizer(model_name="faster_whisper")
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
