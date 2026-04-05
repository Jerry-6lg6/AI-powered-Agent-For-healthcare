import os
import wave
import soundfile as sf
import numpy as np
import threading
import pyaudio
from kokoro import KPipeline


def _wsola_time_stretch(x, speed, sr,
                        frame_ms=40,
                        search_ms=15,
                        overlap_ms=20):
    """
    Lightweight WSOLA time-stretch (speech-optimized).
    Designed for real-time TTS playback with minimal CPU cost.
    """
    if speed == 1.0:
        return x

    frame_len = int(sr * frame_ms / 1000)
    search_len = int(sr * search_ms / 1000)
    overlap_len = int(sr * overlap_ms / 1000)

    hop_in = frame_len - overlap_len
    hop_out = int(hop_in / speed)

    window = np.hanning(frame_len)
    output = np.zeros(int(len(x) / speed) + frame_len)

    in_pos = 0
    out_pos = 0

    # first frame direct copy
    output[:frame_len] = x[:frame_len] * window

    while in_pos + frame_len + search_len < len(x):
        in_pos += hop_in

        ref = output[out_pos:out_pos + overlap_len]
        search_region = x[in_pos - search_len: in_pos + search_len]

        # cross-correlation search (WSOLA)
        corr = np.correlate(search_region[:2 * search_len], ref, mode='valid')
        best_offset = np.argmax(corr)
        best_pos = in_pos - search_len + best_offset

        frame = x[best_pos:best_pos + frame_len] * window
        output[out_pos:out_pos + frame_len] += frame
        out_pos += hop_out

    return output[:out_pos]


class speechSynthesize:
    def __init__(self,
                 model_name="kokoro",
                 gpu=True,
                 output_dir="audio/question",
                 voice="af_bella",
                 lang_code="a"):
        """
        Args:
            model_name: Use "kokoro" to use Kokoro TTS (replaces XTTS v2)
            gpu: Use CUDA GPU acceleration
            output_dir: Directory to save generated audio files
            voice: Kokoro voice name. Options include:
                   American English (lang_code="a"):
                     af_bella, af_sarah, af_nicole, af_sky
                     am_adam, am_michael
                   British English (lang_code="b"):
                     bf_emma, bf_isabella, bm_george, bm_lewis
            lang_code: "a" for American English, "b" for British English
        """
        self.model_name = model_name
        self.gpu = gpu
        self.output_dir = output_dir
        self.voice = voice
        self.lang_code = lang_code
        self.is_playing = threading.Event()
        self.stop_signal = threading.Event()

        # Loading Kokoro TTS pipeline
        device = "cuda" if gpu else "cpu"
        print(f"Loading Kokoro TTS (voice={voice}, device={device})...")
        self.pipeline = KPipeline(lang_code=lang_code, device=device)
        print("Kokoro TTS loaded successfully.")

        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        print(f"Output directory set to: {os.path.abspath(self.output_dir)}")

    def synthesize(self, text, filename="output.wav", speed=1.0, speaker=None):
        """
        Synthesize speech from text using Kokoro TTS.

        Args:
            text:     Text to synthesize
            filename: Output filename (.wav)
            speed:    Speech speed (0.5–2.0 recommended)
            speaker:  Override default voice (optional)
        """
        output_path = os.path.join(self.output_dir, filename)
        voice = speaker if speaker else self.voice

        # Kokoro pipeline returns a generator of audio chunks (24kHz)
        audio_chunks = []
        generator = self.pipeline(text, voice=voice, speed=speed)
        for _, _, audio in generator:
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError(f"Kokoro produced no audio for text: '{text}'")

        # Concatenate all chunks into a single waveform
        full_audio = np.concatenate(audio_chunks)

        # Kokoro outputs float32 at 24kHz — save as WAV
        sf.write(output_path, full_audio, samplerate=24000)

        return filename

    def play_audio(self, filename="output.wav", playback_speed=1.0,
                   text=None, is_synthesize=False):
        """
        Synthesize (if needed) and play audio.

        Args:
            filename:       Output .wav filename
            playback_speed: Speed multiplier for playback (WSOLA)
            text:           Text to synthesize (required if file doesn't exist)
            is_synthesize:  Force re-synthesis even if file exists
        """
        filepath = os.path.join(self.output_dir, filename)
        self.is_playing.set()
        self.stop_signal.clear()

        # Synthesize if file doesn't exist or re-synthesis requested
        if not os.path.exists(filepath) or is_synthesize:
            if text is None:
                raise ValueError("Text must be provided for synthesis")
            filename = self.synthesize(
                text=text,
                filename=filename,
                speed=1.0
            )
            filepath = os.path.join(self.output_dir, filename)

        # Apply WSOLA time-stretch if playback speed differs from 1.0
        if playback_speed != 1.0:
            y, sr = sf.read(filepath)

            if len(y.shape) > 1:
                y = y.mean(axis=1)

            y_stretch = _wsola_time_stretch(y, playback_speed, sr)

            temp_filepath = os.path.join(
                self.output_dir,
                f"__tmp_{playback_speed}_{filename}"
            )
            sf.write(temp_filepath, y_stretch, sr)
            filepath = temp_filepath

        # Play audio via PyAudio
        wf = wave.open(filepath, 'rb')
        p = pyaudio.PyAudio()

        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        chunk = 1024
        data = wf.readframes(chunk)

        try:
            while data:
                if self.stop_signal.is_set():
                    break
                stream.write(data)
                data = wf.readframes(chunk)
        finally:
            self.is_playing.clear()
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()

    def interrupt(self):
        """Stop currently playing audio."""
        self.stop_signal.set()

    def synthesize_and_save(self, text, filename="output.wav", speed=1.0):
        """Convenience method to only synthesize without playing."""
        return self.synthesize(text, filename, speed)


if __name__ == "__main__":
    tts = speechSynthesize(
        model_name="kokoro",
        voice="af_bella",
        lang_code="a",
        gpu=True
    )
    tts.play_audio(
        text="Hello, my name is Jennet, your personal medical assistant.",
        filename="test_kokoro.wav",
        playback_speed=1.0,
        is_synthesize=True
    )
    tts.play_audio(
        text="Let's test the slow playback speed for accessibility.",
        filename="test_kokoro_slow.wav",
        playback_speed=0.85,
        is_synthesize=True
    )
