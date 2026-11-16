"""overlap_asr.py - bổ sung dòng transcript cho người CHEN NGANG / NÓI CHỒNG.

Vấn đề: Whisper chạy một lần trên audio trộn, tại vùng chồng thường chỉ ghi giọng to hơn, nên lời của
người còn lại mất. Module này xử lý RIÊNG các vùng chồng:

  1. vùng chồng + danh sách speaker đang nói (từ pyannote 'full' diarization)
  2. tách nguồn vùng đó (separator, ví dụ SpeechBrain SepFormer) -> các stream
  3. gán stream <-> speaker bằng so khớp giọng với audio "sạch" của speaker (embed, ví dụ ECAPA)
  4. Whisper nhận dạng từng stream -> dòng mới có overlap=True, thời gian được phép chồng dòng khác
  5. bỏ dòng trùng với những gì bảng chính đã có

Tất cả thành phần nặng được truyền vào dạng hàm (separator / embed / asr_words) nên test được bằng hàm giả,
và thay model dễ. Kết quả là GỢI Ý: người dùng nghe lại, sửa trong bảng.
"""
import itertools
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger('digemo.overlap_asr')
SR = 16000
Segment = Tuple[float, float, str]
Word = Tuple[float, float, str]


# ----------------------------------------------------------------------------------------
# 1. Vùng chồng kèm danh sách speaker
# ----------------------------------------------------------------------------------------
def overlap_regions_with_speakers(full_segs: List[Segment], min_dur: float = 0.15,
                                  merge_gap: float = 0.3) -> List[Tuple[float, float, List[str]]]:
    events = []
    for s, e, l in full_segs:
        if e > s:
            events += [(s, 1, l), (e, -1, l)]
    events.sort(key=lambda x: (x[0], x[1]))      # kết thúc trước khi bắt đầu ở cùng thời điểm
    active: Dict[str, int] = {}
    regions, start, who = [], None, set()
    for t, d, l in events:
        was = len(active) >= 2
        active[l] = active.get(l, 0) + d
        if active[l] <= 0:
            del active[l]
        now = len(active) >= 2
        if now:
            who |= set(active)
        if now and not was:
            start, who = t, set(active)
        elif was and not now and start is not None:
            if t - start >= min_dur:
                regions.append([start, t, set(who)])
            start = None
    merged: List[list] = []
    for a, b, w in regions:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
            merged[-1][2] |= w
        else:
            merged.append([a, b, set(w)])
    return [(a, b, sorted(w)) for a, b, w in merged]


# ----------------------------------------------------------------------------------------
# 2-3. Tách nguồn và gán stream cho speaker
# ----------------------------------------------------------------------------------------
def reference_audio(wav: np.ndarray, excl_segs: List[Segment], label: str, center: float,
                    max_s: float = 12.0) -> np.ndarray:
    """Audio sạch của speaker (đoạn exclusive, gần vùng chồng nhất) để so khớp giọng."""
    segs = sorted((x for x in excl_segs if x[2] == label), key=lambda x: abs((x[0] + x[1]) / 2 - center))
    parts, total = [], 0.0
    for s, e, _ in segs:
        a, b = int(s * SR), int(e * SR)
        parts.append(np.asarray(wav[a:b], dtype=np.float32))
        total += e - s
        if total >= max_s:
            break
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def assign_streams(streams: Sequence[np.ndarray], speakers: Sequence[str], refs: Dict[str, np.ndarray],
                   embed: Callable[[np.ndarray], np.ndarray]) -> Dict[str, int]:
    """speaker -> chỉ số stream, chọn hoán vị có tổng độ giống giọng cao nhất."""
    spk = [s for s in speakers if len(refs.get(s, [])) >= SR // 2][:len(streams)]
    if not spk:
        return {}
    ref_e = {s: embed(refs[s]) for s in spk}
    st_e = [embed(x) for x in streams]
    best, best_score = {}, -9.0
    for perm in itertools.permutations(range(len(streams)), len(spk)):
        score = sum(_cos(ref_e[s], st_e[i]) for s, i in zip(spk, perm))
        if score > best_score:
            best, best_score = dict(zip(spk, perm)), score
    return best


def make_speechbrain_separator(device: str = 'cuda', source: str = 'speechbrain/sepformer-wsj02mix'):
    """Tạo hàm separator(wav16k) -> [stream16k, ...]. CHƯA KIỂM CHỨNG trên máy bạn: tên model, sample rate
    (wsj02mix là 8 kHz) và định dạng đầu ra cần được thử thật. Bản sepformer-whamr16k nếu có thì dùng 16 kHz."""
    import torch
    import torchaudio
    from speechbrain.inference.separation import SepformerSeparation
    model = SepformerSeparation.from_hparams(source=source, run_opts={'device': device})
    model_sr = int(getattr(model.hparams, 'sample_rate', 8000))

    def separate(x: np.ndarray) -> List[np.ndarray]:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        t = torchaudio.functional.resample(t, SR, model_sr)
        with torch.inference_mode():
            est = model.separate_batch(t.to(device))          # (B, T, n_src)
        out = []
        for k in range(est.shape[-1]):
            s = est[0, :, k].float().cpu().unsqueeze(0)
            s = torchaudio.functional.resample(s, model_sr, SR)[0].numpy()
            s = np.pad(s, (0, max(0, len(x) - len(s))))[:len(x)]
            out.append(s / (np.abs(s).max() + 1e-6) * min(1.0, float(np.abs(x).max()) + 1e-6))
        return out
    return separate


# ----------------------------------------------------------------------------------------
# 4-5. Tạo dòng chồng và loại dòng trùng
# ----------------------------------------------------------------------------------------
def _norm_words(text: str) -> List[str]:
    return [w for w in ''.join(c.lower() if c.isalnum() or c == ' ' else ' ' for c in text).split() if w]


def _is_duplicate(text: str, a: float, b: float, name: str, main_rows: List[dict], thr: float = 0.6) -> bool:
    w = _norm_words(text)
    if not w:
        return True
    for r in main_rows:
        if r['speaker'] != name:
            continue
        ov = min(b, r['end']) - max(a, r['start'])
        if ov <= 0:
            continue
        have = set(_norm_words(r['text']))
        if sum(1 for x in w if x in have) / len(w) >= thr:
            return True
    return False


def add_overlap_rows(wav: np.ndarray, main_rows: List[dict], excl_segs: List[Segment], full_segs: List[Segment],
                     speaker_names: Dict[str, str], asr_words: Callable[[np.ndarray], List[Word]],
                     separator: Callable[[np.ndarray], List[np.ndarray]],
                     embed: Callable[[np.ndarray], np.ndarray], build_utterances: Callable,
                     pad: float = 0.3, min_stream_rms_ratio: float = 0.15,
                     stats: Optional[dict] = None) -> List[dict]:
    """Trả danh sách dòng gồm main_rows + dòng bổ sung (overlap=1.0, overlap_added=True), sắp theo start.

    speaker_names: nhãn pyannote -> tên hiển thị (stats['speaker_names'] của rows_from_diarization)."""
    names = dict(speaker_names)
    rows = [dict(r) for r in main_rows]
    added, skipped = [], []
    for a, b, who in overlap_regions_with_speakers(full_segs):
        if len(who) < 2:
            continue
        lo, hi = max(0.0, a - pad), min(len(wav) / SR, b + pad)
        region = np.asarray(wav[int(lo * SR):int(hi * SR)], dtype=np.float32)
        if len(region) < SR // 4:
            continue
        streams = separator(region)
        rms = [float(np.sqrt(np.mean(s * s))) for s in streams]
        refs = {l: reference_audio(wav, excl_segs, l, (a + b) / 2) for l in who}
        mapping = assign_streams(streams, who, refs, embed)
        for label, k in mapping.items():
            if rms[k] < min_stream_rms_ratio * max(rms):
                skipped.append((a, b, label, 'stream quá yếu'))
                continue
            words = [(s + lo, e + lo, t) for s, e, t in asr_words(streams[k]) if t.strip()]
            words = [w for w in words if w[1] > a - pad and w[0] < b + pad]      # chỉ giữ từ trong vùng chồng
            for u in build_utterances(words):
                if label not in names:
                    names[label] = f'Speaker {len(names) + 1}'
                nm = names[label]
                if _is_duplicate(u['text'], u['start'], u['end'], nm, rows):
                    skipped.append((u['start'], u['end'], label, 'trùng bảng chính'))
                    continue
                added.append({'start': u['start'], 'end': u['end'], 'text': u['text'], 'speaker': nm,
                              'speaker_raw': label, 'overlap': 1.0, 'overlap_added': True,
                              'dialogue_id': None, 'truth': None})
    out = rows + added
    out.sort(key=lambda r: (r['start'], r['end']))
    if stats is not None:
        stats.update(overlap_rows_added=len(added), overlap_skipped=skipped, speaker_names=names)
    return out
