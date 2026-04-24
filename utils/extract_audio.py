# utils/extract_audio.py
# Convert an input AUDIO or VIDEO file to mono 16 kHz WAV and return temp path.

from __future__ import annotations
import os
import tempfile
import subprocess

import numpy as np
import librosa

try:
    import soundfile as sf  # pip install soundfile
except Exception:  # tiny fallback via scipy if soundfile missing
    sf = None
    from scipy.io import wavfile as wavwrite  # type: ignore


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v"}


def _tmp_wav_path() -> str:
    fd, p = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    return p


def _write_wav(y: np.ndarray, sr: int, out_path: str) -> None:
    y = y.astype(np.float32)
    if sf is not None:
        sf.write(out_path, y, sr)
    else:
        # scipy requires int16; simple scale
        y16 = np.clip(y, -1.0, 1.0)
        y16 = (y16 * 32767.0).astype(np.int16)
        wavwrite.write(out_path, sr, y16)  # type: ignore


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        return False


def extract_wav_mono(input_path: str, target_sr: int = 16000) -> str | None:
    """
    Returns path to a temp mono 16 kHz WAV made from input (audio OR video).
    On failure returns None.
    """
    if not os.path.isfile(input_path):
        return None

    ext = os.path.splitext(input_path)[1].lower()
    out_wav = _tmp_wav_path()

    # ---- If it's a video and ffmpeg exists, use ffmpeg directly (fast & robust)
    if ext in VIDEO_EXTS and _has_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vn",             # no video
            "-ac", "1",        # mono
            "-ar", str(target_sr),
            "-f", "wav", out_wav
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return out_wav
        except Exception:
            pass  # fall through to librosa path

    # ---- Librosa path (works for audio; and video if ffmpeg not present but some decoders available)
    try:
        y, sr = librosa.load(input_path, sr=target_sr, mono=True)
        _write_wav(y, target_sr, out_wav)
        return out_wav
    except Exception:
        return None
