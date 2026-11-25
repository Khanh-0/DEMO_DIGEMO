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


def _segments(ann) -> List[Segment]:
    return [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]


def _run_local(wav: np.ndarray, num_speakers: Optional[int]) -> Tuple[List[Segment], List[Segment]]:
    """Returns (exclusive, full). Exclusive has no overlaps (one speaker per instant) -> used to assign words.
    Full keeps overlapped speech -> only used to flag utterances where two people talk at once."""
    import torch
    pipe = _load_pipeline()
    waveform = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
    kwargs = {'num_speakers': int(num_speakers)} if num_speakers else {}
    out = pipe({'waveform': waveform, 'sample_rate': SAMPLE_RATE}, **kwargs)
    excl = getattr(out, 'exclusive_speaker_diarization', None)
    full = getattr(out, 'speaker_diarization', None)
    if full is None:      # pyannote 3.x returns the Annotation itself
        full = out
    if excl is None:
        excl = full
    return _segments(excl), _segments(full)


def _run_worker(npy_path: Path, num_speakers: Optional[int]) -> Tuple[List[Segment], List[Segment]]:
    py = os.environ['DIARIZE_PYTHON']
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / 'segs.json'
        cmd = [py, str(Path(__file__).resolve()), '--worker', str(npy_path), str(num_speakers or 0), str(out)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3600, check=False)
        if res.returncode != 0 or not out.exists():
            tail = res.stderr.decode('utf-8', errors='replace')[-500:]
            raise DiarizerUnavailable(f'Worker pyannote lỗi: {tail}')
        data = json.loads(out.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return ([tuple(x) for x in data['exclusive']], [tuple(x) for x in data['full']])
        segs = [tuple(x) for x in data]   # worker cũ chỉ trả một danh sách
        return segs, segs


def diarize_full(wav_or_npy, num_speakers: Optional[int]) -> Tuple[List[Segment], List[Segment]]:
    """wav_or_npy: float32 mono 16 kHz array, or path to a .npy of it. Returns (exclusive, full) segment lists."""
    if os.environ.get('DIARIZE_PYTHON'):
        if isinstance(wav_or_npy, (str, Path)):
            return _run_worker(Path(wav_or_npy), num_speakers)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'a.npy'
            np.save(p, np.asarray(wav_or_npy, dtype=np.float32))
            return _run_worker(p, num_speakers)
    wav = np.load(wav_or_npy) if isinstance(wav_or_npy, (str, Path)) else wav_or_npy
    return _run_local(np.asarray(wav, dtype=np.float32), num_speakers)


def diarize(wav_or_npy, num_speakers: Optional[int]) -> List[Segment]:
    """Backward compatible: exclusive segments only. Returns [(start, end, label)]."""
    return diarize_full(wav_or_npy, num_speakers)[0]


# --------------------------------------------------------------------------------------
# Aligning Whisper words with diarization (pure python, no torch)
# --------------------------------------------------------------------------------------
MAX_ASSIGN_GAP_S = 0.5   # từ nằm xa mọi đoạn nói hơn mức này -> không gán người nói (thường là Whisper bịa trong im lặng)
SMOOTH_CONF = 0.6        # chỉ làm mượt từ lẻ khi nó khớp < 60% thời gian với người nói được gán
MIN_OVERLAP_S = 0.15


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_for_span(start: float, end: float, segs: List[Segment],
                     max_gap: float = MAX_ASSIGN_GAP_S) -> Optional[str]:
    """Label with the largest overlap; if none overlaps, the closest segment only when it is within max_gap
    seconds, otherwise None (unknown) instead of guessing a far-away speaker."""
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

    def dist(x):
        return 0.0 if x[0] <= mid <= x[1] else min(abs(mid - x[0]), abs(mid - x[1]))
    best = min(segs, key=dist)
    return best[2] if dist(best) <= max_gap else None


def label_words(words, segs: List[Segment], smooth_conf: float = SMOOTH_CONF) -> List[Optional[str]]:
    """words: [(start, end, text)] -> speaker label per word (None = unknown).

    A word sandwiched between two words of the same other speaker is smoothed only when it is weakly supported
    by its own label, so a clearly overlapping one-word interjection ("ừ", "no!") keeps its real speaker."""
    labels = [speaker_for_span(w[0], w[1], segs) for w in words]
    conf = []
    for w, l in zip(words, labels):
        if l is None:
            conf.append(0.0)
            continue
        own = sum(_overlap(w[0], w[1], s, e) for s, e, lab in segs if lab == l)
        conf.append(min(1.0, own / max(w[1] - w[0], 1e-3)))
    for i in range(1, len(labels) - 1):
        if labels[i - 1] is not None and labels[i - 1] == labels[i + 1] != labels[i] and conf[i] < smooth_conf:
            labels[i] = labels[i - 1]
    return labels


def overlap_regions(segs: List[Segment], min_dur: float = MIN_OVERLAP_S) -> List[Tuple[float, float]]:
    """Time ranges where two or more different speakers are active (from the non-exclusive diarization)."""
    events = []
    for s, e, l in segs:
        if e > s:
            events.append((s, 1, l))
            events.append((e, -1, l))
    events.sort(key=lambda x: (x[0], x[1]))   # at equal time, ends come before starts: touching != overlapping
    active: Dict[str, int] = {}
    out, start = [], None
    for t, d, l in events:
        was = len(active) >= 2
        active[l] = active.get(l, 0) + d
        if active[l] <= 0:
            del active[l]
        now = len(active) >= 2
        if now and not was:
            start = t
        elif was and not now and start is not None:
            if t - start >= min_dur:
                out.append((start, t))
            start = None
    return out


def overlap_ratio(start: float, end: float, regions: List[Tuple[float, float]]) -> float:
    dur = max(end - start, 1e-6)
    return min(1.0, sum(_overlap(start, end, a, b) for a, b in regions) / dur)


def rows_from_diarization(words, segs: List[Segment], build_utterances,
                          overlap_segs: Optional[List[Segment]] = None,
                          stats: Optional[dict] = None) -> List[dict]:
    """Split the word stream at every speaker change, then build sentence-like utterances per run.

    build_utterances is video_pipeline.build_utterances (kept as an argument to avoid a circular import).
    Speakers are renamed 'Speaker 1..N' by order of first appearance (raw pyannote id kept in 'speaker_raw').
    Words with no speaker close enough are dropped and counted in stats['dropped_words'].
    overlap_segs (non-exclusive diarization) adds an 'overlap' ratio 0..1 to every utterance.
    """
    words = [w for w in words if w[2]]
    if not words:
        return []
    labels = label_words(words, segs)
    kept = [(w, l) for w, l in zip(words, labels) if l is not None]
    dropped = [w[2] for w, l in zip(words, labels) if l is None]
    regions = overlap_regions(overlap_segs) if overlap_segs else []
    runs, cur, cur_l = [], [], None
    for w, l in kept:
        if cur and l != cur_l:
            runs.append((cur_l, cur))
            cur = []
        cur.append(w)
        cur_l = l
    if cur:
        runs.append((cur_l, cur))

    names: Dict[str, str] = {}
    rows = []
    for label, run in runs:
        name = names.setdefault(label, f'Speaker {len(names) + 1}')
        for u in build_utterances(run):
            rows.append({'start': u['start'], 'end': u['end'], 'text': u['text'],
                         'speaker': name, 'speaker_raw': label,
                         'overlap': round(overlap_ratio(u['start'], u['end'], regions), 2),
                         'dialogue_id': None, 'truth': None})
    rows.sort(key=lambda r: r['start'])
    if stats is not None:
        stats.update(dropped_words=len(dropped), dropped_sample=dropped[:8], speaker_names=names,
                     overlap_utterances=sum(1 for r in rows if r['overlap'] >= 0.3),
                     overlap_seconds=round(sum(b - a for a, b in regions), 2))
    return rows

def merge_segments(segs: List[Segment], merge_gap: float = 0.6, min_dur: float = 0.25) -> List[Segment]:
    """Nối các đoạn liền kề cùng người nói (nghỉ <= merge_gap), bỏ đoạn quá ngắn."""
    out: List[list] = []
    for s, e, l in sorted(segs, key=lambda x: x[0]):
        if out and out[-1][2] == l and s - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, l])
    return [(s, e, l) for s, e, l in out if e - s >= min_dur]


def rows_from_segments(wav, segs: List[Segment], transcribe, build_utterances,
                       overlap_segs: Optional[List[Segment]] = None, stats: Optional[dict] = None,
                       on_progress=None, pad: float = 0.1, max_s: float = 14.0,
                       tol: float = 0.05) -> List[dict]:
    """Mỗi đoạn pyannote = một dòng; Whisper chỉ nhận dạng phần audio đã cắt.

    - Thời gian dòng = thời gian pyannote (không lấy từ Whisper).
    - Cắt dư `pad` giây hai đầu cho Whisper khỏi mất âm đầu/cuối, rồi chỉ giữ từ nằm trong [start, end]
      (loại từ của người nói kế bên lọt vào phần đệm).
    - Đoạn dài hơn max_s mới được tách tiếp theo dấu câu/khoảng nghỉ bằng build_utterances.
    transcribe(wav_slice) -> [(start, end, text)] với thời gian tính từ đầu lát cắt.
    """
    segs = merge_segments(segs)
    regions = overlap_regions(overlap_segs) if overlap_segs else []
    dur = len(wav) / SAMPLE_RATE
    names: Dict[str, str] = {}
    rows, empty = [], 0
    for i, (s, e, label) in enumerate(segs):
        if on_progress:
            on_progress(i / max(len(segs), 1))
        a, b = max(0.0, s - pad), min(dur, e + pad)
        piece = np.asarray(wav[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)], dtype=np.float32)
        words = [(ws + a, we + a, t) for ws, we, t in transcribe(piece) if t.strip()]
        words = [w for w in words if s - tol <= (w[0] + w[1]) / 2 <= e + tol]
        if not words:
            empty += 1
            continue
        name = names.setdefault(label, f'Speaker {len(names) + 1}')
        if e - s <= max_s:
            parts = [{'start': s, 'end': e, 'text': ' '.join(w[2] for w in words).strip()}]
        else:
            parts = build_utterances(words)
        for u in parts:
            rows.append({'start': u['start'], 'end': u['end'], 'text': u['text'],
                         'speaker': name, 'speaker_raw': label,
                         'overlap': round(overlap_ratio(u['start'], u['end'], regions), 2),
                         'dialogue_id': None, 'truth': None})
    rows.sort(key=lambda r: r['start'])
    if stats is not None:
        stats.update(empty_segments=empty, speaker_names=names,
                     overlap_utterances=sum(1 for r in rows if r['overlap'] >= 0.3),
                     overlap_seconds=round(sum(b - a for a, b in regions), 2))
    return rows
# --------------------------------------------------------------------------------------
# Worker entry point (runs inside DIARIZE_PYTHON)
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        _, _, npy, n, dst = sys.argv
        excl, full = _run_local(np.load(npy), int(n) or None)
        Path(dst).write_text(json.dumps({'exclusive': excl, 'full': full}), encoding='utf-8')
    else:
        print('usage: diarizer.py --worker audio.npy NUM_SPEAKERS out.json')
        sys.exit(2)