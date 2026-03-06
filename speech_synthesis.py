import os
import wave
# import librosa
# import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
from TTS.api import TTS
import pyaudio


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

        # cross-correlation search (THIS is WSOLA magic)
        corr = np.correlate(search_region[:2 * search_len], ref, mode='valid')
        best_offset = np.argmax(corr)

        best_pos = in_pos - search_len + best_offset

        frame = x[best_pos:best_pos + frame_len] * window

        output[out_pos:out_pos + frame_len] += frame

        out_pos += hop_out

    return output[:out_pos]


class speechSynthesize:
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=True, output_dir="audio/question"):
        self.model_name = model_name
        self.gpu = gpu
        self.output_dir = output_dir
        self.is_playing = threading.Event()
        self.stop_signal = threading.Event()

        # Loading TTS model
        print(f"Loading TTS model: {model_name} (GPU={gpu})...")
        self.tts_api = TTS(model_name)
        if self.gpu:
            self.tts_api.to("cuda")

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
                speed=speed,
            )
        else:
            # For other TTS models
            self.tts_api.tts_to_file(
                text=text,
                file_path=output_path,
                speed=speed
            )

        return filename

    def play_audio(self, filename="output.wav", playback_speed=1.0,
                   text=None, is_synthesize=False):

        filepath = os.path.join(self.output_dir, filename)
        self.is_playing.set()
        self.stop_signal.clear()
        # If audio do not exists or need to synthesize => Synthesize new audio
        if not os.path.exists(filepath) or is_synthesize:
            if text is None:
                raise ValueError("Text must be provided for synthesis")
            filename = self.synthesize(
                text=text,
                filename=filename,
                speed=1.0
            )
            filepath = os.path.join(self.output_dir, filename)

        # If need to change playback speed => Utilize the WSOLA function to change
        if playback_speed != 1.0:
            # y, sr = librosa.load(filepath, sr=None, mono=True)
            #
            # y = librosa.effects.preemphasis(y)
            #
            # y = librosa.effects.time_stretch(y,
            #                                  rate=playback_speed,
            #                                  n_fft=2048,
            #                                  hop_length=256)
            # y = librosa.effects.deemphasis(y)

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

        # print(f"▶️ Playback speed: {playback_speed}x (timbre preserved)")

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
        self.stop_signal.set()

    def synthesize_and_save(self, text, filename="output.wav", speed=1.0):
        """Convenience method to only synthesize without playing."""
        return self.synthesize(text, filename, speed)


if __name__ == "__main__":
    tts = speechSynthesize(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    tts.play_audio(
        text="Hi, my name is Jerry 6. Let's test the playback speed.",
        filename="output_new_1.wav",
        playback_speed=1,
        is_synthesize=True
    )
    speed_new = 0.8
    tts.play_audio(
        text="Hi, my name is Jerry. Let's test the playback speed.",
        filename=f"output_new_{speed_new}.wav",
        playback_speed=float(speed_new),
        is_synthesize=True
    )
