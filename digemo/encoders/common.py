"""Tiện ích dùng chung: key diaX_uttY, đọc/ghi pkl, quét file, đọc audio từ mp4 (ffmpeg), đọc csv MELD."""
import glob
import os
import pickle
import re
import shutil
import subprocess

import numpy as np

KEY = re.compile(r"dia(\d+)_utt(\d+)")
AUDIO_EXT = (".mp4", ".wav", ".flac", ".mp3", ".ogg", ".m4a")
# nhãn MELD theo paper: {0: neutral, 1: surprise, 2: fear, 3: sadness, 4: joy, 5: disgust, 6: anger}
EMO_NAMES = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
EMO = {n: i for i, n in enumerate(EMO_NAMES)}
SR, MAX_SEC = 16000, 15


def key(d, u):
    return f"dia{int(d)}_utt{int(u)}"


def parse_key(s):
    m = KEY.search(str(s))
    return (int(m.group(1)), int(m.group(2))) if m else None


def save_pkl(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]


def collect_files(root, exts=AUDIO_EXT):
    """quét đệ quy -> {'diaX_uttY': path}"""
    files = {}
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if p.lower().endswith(exts):
            k = parse_key(os.path.basename(p))
            if k:
                files[key(*k)] = p
    return files


def read_split_csv(path):
    import pandas as pd

    df = pd.read_csv(path)
    df["key"] = [key(d, u) for d, u in zip(df["Dialogue_ID"], df["Utterance_ID"])]
    if "Utterance" in df:
        df["Utterance"] = df["Utterance"].fillna("").astype(str).str.strip()
    return df


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise RuntimeError("Không tìm thấy ffmpeg: cài ffmpeg vào PATH hoặc pip install imageio-ffmpeg")


def decode_audio(path):
    """mp4/wav/... -> (float32 mono 16 kHz, tối đa MAX_SEC giây, returncode)"""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
           "-t", str(MAX_SEC), "-f", "s16le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0, r.returncode
