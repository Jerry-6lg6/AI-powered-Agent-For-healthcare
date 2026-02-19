# file convertion
# import librosa
# import soundfile as sf
# import glob
# import os
#
# src = r"D:/MS_SNSD_Hospital/CleanSpeech/LibriSpeech/train-clean-100"
# dst = r"D:/MS_SNSD_Hospital/clean_wav"
# os.makedirs(dst, exist_ok=True)
#
# for flac in glob.glob(src + "/**/*.flac", recursive=True):
#     y, sr = librosa.load(flac, sr=16000)
#     name = os.path.basename(flac).replace(".flac", ".wav")
#     sf.write(os.path.join(dst, name), y, 16000)
import json
import numpy as np
import time
import rnnoise # RNN based network for
import jiwer
import os
import soundfile as sf
from tqdm import tqdm
from speech_Recognition import RobustSpeechRecognizer
from tools import SpectralSubtraction, RNNoisePlugin, WienerFilter


def compute_wer(ref, hyp):
    return jiwer.wer(ref, hyp)


def parse_utt_id(noisy_wav_name):
    """
    2952-408-0026_SNRdb_0_noisy.wav
        -> 2952-408-0026
    """
    return noisy_wav_name.split("_SNRdb_")[0]


def save_trans_dict_json(trans_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(trans_dict, f, ensure_ascii=False, indent=2)


def load_librispeech_transcriptions(librispeech_root):
    """
    遍历 LibriSpeech，建立 utt_id -> text 的映射
    """
    trans_dict = {}

    for root, _, files in os.walk(librispeech_root):
        for f in files:
            if f.endswith(".trans.txt"):
                with open(os.path.join(root, f), "r", encoding="utf-8") as fin:
                    for line in fin:
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            utt_id, text = parts
                            trans_dict[utt_id] = text.lower()
    save_trans_dict_json(
        trans_dict,
        r"D:\MS_SNSD_Hospital\cache\librispeech_transcriptions.json"
    )

    return trans_dict


def load_trans_dict_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate_asr(
        recognizer,
        wav_dir=r"D:\MS_SNSD_Hospital\MS-SNSD\NoisySpeech_training",
        cache_dir=r"D:\MS_SNSD_Hospital\cache",
        sr=16000,
        debug=True
):
    transcriptions_file = os.path.join(cache_dir, "librispeech_transcriptions.json")
    with open(transcriptions_file, "r", encoding="utf-8") as f:
        tran_dict = json.load(f)

    wers = []
    durations = []  # 存储每条音频处理时间

    def rms(x):
        return np.sqrt(np.mean(x ** 2))

    for wav_file in tqdm(os.listdir(wav_dir)):
        if not wav_file.endswith(".wav"):
            continue

        start_time = time.time()  # 计时开始

        audio_path = os.path.join(wav_dir, wav_file)
        audio, fs = sf.read(audio_path)

        # ===== Level 0: 原始音频检查 =====
        if debug:
            if fs != sr:
                print(f"[SR Mismatch] {wav_file}: {fs} != {sr}")
            if np.isnan(audio).any() or np.isinf(audio).any():
                print(f"[Invalid Audio] {wav_file}")
                continue

        # ===== Apply denoise plugin =====
        audio_dn = recognizer.apply_plugins(audio, fs)

        # ===== Level 1: 时序 / 能量检查 =====
        if debug:
            dur_diff = abs(len(audio_dn) - len(audio)) / fs
            if dur_diff > 0.1:
                print(f"[Time Drift] {wav_file}: {dur_diff:.2f}s")

            energy_ratio = rms(audio_dn) / (rms(audio) + 1e-8)
            if energy_ratio < 0.3:
                print(f"[Over-suppression] {wav_file}, ratio={energy_ratio:.2f}")

        # ===== ASR =====
        if recognizer.model_name == "faster_whisper":
            segments, _ = recognizer.model.transcribe(audio_dn, language="en")
            hyp = " ".join([s.text for s in segments]).lower()
        else:
            hyp = recognizer.model.transcribe(audio_dn)["text"].lower()

        # ===== GT 对齐 =====
        base_id = wav_file.split("_")[0]
        gt_text = tran_dict.get(base_id, "").lower()

        if not gt_text:
            print(f"[GT Missing] {wav_file} -> {base_id}")
            continue

        # ===== Level 3: WER 结构诊断 =====
        wer_transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords()
        ])

        out = jiwer.process_words(
            gt_text,
            hyp,
            reference_transform=wer_transform,
            hypothesis_transform=wer_transform
        )

        wer = out.wer
        wers.append(wer)

        if debug and wer > 0.5:
            print(f"\n🚨 High WER detected: {wav_file}")
            print(f"WER: {wer:.2f}")
            print(f"S/D/I: {out.substitutions} / {out.deletions} / {out.insertions}")
            print(f"GT : {gt_text}")
            print(f"HYP: {hyp[:200]}")

        # ===== 记录处理时间 =====
        durations.append(time.time() - start_time)

    # ===== 输出平均 WER & 平均处理用时 =====
    if wers:
        print(f"\nAverage WER: {sum(wers) / len(wers):.4f}")
    else:
        print("No WER calculated, please check dataset and transcriptions.")

    if durations:
        avg_time = sum(durations) / len(durations)
        print(f"Average processing time per audio: {avg_time:.2f}s")


if __name__ == "__main__":
    noisy_root = r"D:\MS_SNSD_Hospital\MS-SNSD\NoisySpeech_training"
    librispeech_root = r"D:\MS_SNSD_Hospital\CleanSpeech\LibriSpeech"
    # load_librispeech_transcriptions(librispeech_root)
    gt_dict = load_trans_dict_json(
        r"D:\MS_SNSD_Hospital\cache\librispeech_transcriptions.json"
    )

    print(dir(rnnoise))

    baseline = RobustSpeechRecognizer(
        model_name="faster_whisper",
        plugins=[]
    )
    evaluate_asr(baseline)

    denoise = RobustSpeechRecognizer(
        model_name="faster_whisper",
        plugins=[
            SpectralSubtraction(),
        ]
    )
    evaluate_asr(denoise)

