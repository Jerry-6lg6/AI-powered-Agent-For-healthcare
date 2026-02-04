from abc import ABC, abstractmethod
import numpy as np
import librosa
from scipy.signal import wiener
from pyrnnoise import rnnoise

class AudioPlugin(ABC):
    """
    Base class for all audio enhancement plugins
    """
    @abstractmethod
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        pass

class SpectralSubtraction(AudioPlugin):
    def __init__(self, noise_frames=10, alpha=1.5):
        self.noise_frames = noise_frames
        self.alpha = alpha

    def process(self, audio, sr):
        stft = librosa.stft(audio)
        mag, phase = np.abs(stft), np.angle(stft)

        noise_mag = np.mean(mag[:, :self.noise_frames], axis=1, keepdims=True)
        clean_mag = np.maximum(mag - self.alpha * noise_mag, 0)

        clean_stft = clean_mag * np.exp(1j * phase)
        return librosa.istft(clean_stft)

class WienerFilter(AudioPlugin):
    def process(self, audio, sr):
        return wiener(audio)

class RNNoisePlugin(AudioPlugin):
    def __init__(self, target_sr=48000):
        self.state = rnnoise.create()
        self.target_sr = target_sr
        self.frame_size = 480  # 10 ms @ 48 kHz

    def process(self, audio, sr):
        # -------- Mono --------
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # -------- Resample --------
        if sr != self.target_sr:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sr,
                target_sr=self.target_sr
            )

        audio = audio.astype(np.float32)

        # -------- Padding --------
        remainder = len(audio) % self.frame_size
        if remainder != 0:
            pad = self.frame_size - remainder
            audio = np.pad(audio, (0, pad))

        # -------- Frame-wise RNNoise --------
        out = np.zeros_like(audio, dtype=np.float32)
        for i in range(0, len(audio), self.frame_size):
            frame = audio[i:i + self.frame_size]

            # Convert to int16 for RNNoise
            frame_int16 = (frame * 32768).astype(np.int16)

            processed = rnnoise.process_frame(self.state, frame_int16)

            # If wrapper returns tuple/list
            if isinstance(processed, (tuple, list)):
                processed = processed[0]

            processed = np.asarray(processed, dtype=np.float32) / 32768.0

            out[i:i + self.frame_size] = processed

        out = np.clip(out, -1.0, 1.0)

        return out




