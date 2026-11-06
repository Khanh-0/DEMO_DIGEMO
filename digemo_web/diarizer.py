"""Speaker diarization with pyannote/speaker-diarization-community-1 for the DiGemo video pipeline.

Put next to video_pipeline.py. Two ways to run pyannote:

1. In-process (default): needs `pip install pyannote.audio` in the same venv.
2. Separate venv (recommended if pip complains about torch versions): set
       DIARIZE_PYTHON=C:\\path\\to\\venv_pyannote\\Scripts\\python.exe
   and this module calls itself as a worker in that interpreter.

Hugging Face token: set env HF_TOKEN (after accepting the model conditions on the model page).
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger('digemo.diarizer')
SAMPLE_RATE = 16000
MODEL_ID = 'pyannote/speaker-diarization-community-1'

Segment = Tuple[float, float, str]  # start, end, raw label


class DiarizerUnavailable(RuntimeError):
    """pyannote missing / no token / worker failed. Caller may fall back to the old clustering."""


# --------------------------------------------------------------------------------------
# Running pyannote
# --------------------------------------------------------------------------------------
_pipeline = None


def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizerUnavailable(
            'Chưa cài pyannote.audio. Chạy: pip install pyannote.audio '
            '(hoặc đặt DIARIZE_PYTHON trỏ tới venv riêng).') from exc
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        try:  # token saved by `hf auth login`
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if not token:
        raise DiarizerUnavailable(
            'Thiếu Hugging Face token. Chạy `hf auth login` hoặc đặt biến môi trường HF_TOKEN, '
            'và đã bấm Accept điều khoản trên trang model.')
    try:
        try:
            pipe = Pipeline.from_pretrained(MODEL_ID, token=token)
        except TypeError:  # pyannote.audio 3.x
            pipe = Pipeline.from_pretrained(MODEL_ID, use_auth_token=token)
    except Exception as exc:
        raise DiarizerUnavailable(f'Không tải được {MODEL_ID}: {exc}') from exc
    if pipe is None:
        raise DiarizerUnavailable(f'Không tải được {MODEL_ID} (chưa Accept điều khoản hoặc token sai).')
    if torch.cuda.is_available():
        pipe.to(torch.device('cuda'))
    _pipeline = pipe
    return pipe


def _run_local(wav: np.ndarray, num_speakers: Optional[int]) -> List[Segment]:
    import torch
    pipe = _load_pipeline()
    waveform = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    kwargs = {'num_speakers': int(num_speakers)} if num_speakers else {}
    out = pipe({'waveform': waveform, 'sample_rate': SAMPLE_RATE}, **kwargs)
    # community-1: "exclusive" diarization has no overlaps -> easier to align with Whisper words.
    ann = getattr(out, 'exclusive_speaker_diarization', None) or getattr(out, 'speaker_diarization', out)
    return [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]


def _run_worker(npy_path: Path, num_speakers: Optional[int]) -> List[Segment]:
    py = os.environ['DIARIZE_PYTHON']
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'segs.json'
        cmd = [py, str(Path(__file__).resolve()), '--worker', str(npy_path), str(num_speakers or 0), str(out)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600, check=False)
        if res.returncode != 0 or not out.exists():
            tail = res.stderr.decode('utf-8', errors='replace')[-500:]
            raise DiarizerUnavailable(f'Worker pyannote lỗi: {tail}')
        return [(s, e, l) for s, e, l in json.loads(out.read_text(encoding='utf-8'))]


def diarize(wav_or_npy, num_speakers: Optional[int]) -> List[Segment]:
    """wav_or_npy: float32 mono 16 kHz array, or path to a .npy of it. Returns [(start, end, label)]."""
    if os.environ.get('DIARIZE_PYTHON'):
        if isinstance(wav_or_npy, (str, Path)):
            return _run_worker(Path(wav_or_npy), num_speakers)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'a.npy'
            np.save(p, np.asarray(wav_or_npy, dtype=np.float32))
            return _run_worker(p, num_speakers)
    wav = np.load(wav_or_npy) if isinstance(wav_or_npy, (str, Path)) else wav_or_npy
    return _run_local(np.asarray(wav, dtype=np.float32), num_speakers)


# --------------------------------------------------------------------------------------
# Aligning Whisper words with diarization (pure python, no torch)
# --------------------------------------------------------------------------------------
def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_for_span(start: float, end: float, segs: List[Segment]) -> Optional[str]:
    """Label with the largest overlap; if none overlaps, the closest segment to the span's centre."""
    if not segs:
        return None
    score: Dict[str, float] = {}
    for s, e, l in segs:
        o = _overlap(start, end, s, e)
        if o > 0:
            score[l] = score.get(l, 0.0) + o
    if score:
        return max(score, key=score.get)
    mid = (start + end) / 2
    return min(segs, key=lambda x: 0.0 if x[0] <= mid <= x[1] else min(abs(mid - x[0]), abs(mid - x[1])))[2]


def label_words(words, segs: List[Segment]) -> List[str]:
    """words: [(start, end, text)] -> speaker label per word, with isolated one-word flickers smoothed."""
    labels = [speaker_for_span(w[0], w[1], segs) for w in words]
    for i in range(1, len(labels) - 1):
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]
    return labels


def rows_from_diarization(words, segs: List[Segment], build_utterances) -> List[dict]:
    """Split the word stream at every speaker change, then build sentence-like utterances per run.

    build_utterances is video_pipeline.build_utterances (kept as an argument to avoid a circular import).
    Speakers are renamed 'Speaker 1..N' by order of first appearance.
    """
    words = [w for w in words if w[2]]
    if not words:
        return []
    labels = label_words(words, segs)
    runs, cur, cur_l = [], [], None
    for w, l in zip(words, labels):
        if cur and l != cur_l:
            runs.append((cur_l, cur))
            cur = []
        cur.append(w)
        cur_l = l
    runs.append((cur_l, cur))

    names: Dict[str, str] = {}
    rows = []
    for label, run in runs:
        name = names.setdefault(label, f'Speaker {len(names) + 1}')
        for u in build_utterances(run):
            rows.append({'start': u['start'], 'end': u['end'], 'text': u['text'],
                         'speaker': name, 'dialogue_id': None, 'truth': None})
    rows.sort(key=lambda r: r['start'])
    return rows


# --------------------------------------------------------------------------------------
# Worker entry point (runs inside DIARIZE_PYTHON)
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        _, _, npy, n, dst = sys.argv
        segs = _run_local(np.load(npy), int(n) or None)
        Path(dst).write_text(json.dumps(segs), encoding='utf-8')
    else:
        print('usage: diarizer.py --worker audio.npy NUM_SPEAKERS out.json')
        sys.exit(2)