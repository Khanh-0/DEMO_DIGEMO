"""DiGemo video pipeline V1 (no face tracking).

video -> ffmpeg audio -> Whisper words -> utterances -> speaker clustering (N speakers)
      -> RoBERTa / Wav2Vec2 / DenseNet per utterance -> DiGemo (whole dialogue at once).

Can also take a training-style CSV (Dialogue_ID, Speaker, Utterance, StartTime, EndTime, Emotion)
instead of ASR + clustering. Put this file next to app.py and call
`register_video_routes(app, active_engine)` inside create_app().
"""
import atexit
import csv
import io
import logging
import math
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
from diarizer import diarize, rows_from_diarization, DiarizerUnavailable
import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

LOG = logging.getLogger('digemo.video')
HERE = Path(__file__).resolve().parent

MAX_VIDEO_BYTES = 800 * 1024 * 1024
MAX_VIDEO_SECONDS = 10 * 60
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_UTTERANCES = 300
MAX_PER_DIALOGUE_HARD = 100
FRAMES_PER_UTT = 8
MIN_AUDIO_S = 0.4
MAX_UTT_S = 14.0
VIDEO_TTL_SECONDS = 2 * 3600
SAMPLE_RATE = 16000
EXPECTED_DIMS = {'text': 1024, 'audio': 300, 'visual': 342}
ALLOWED_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

CSV_ALIASES = {
    'dialogue': ['dialogue_id', 'dialogueid', 'dialogue', 'conversation_id'],
    'speaker': ['speaker', 'speaker_id', 'spk'],
    'text': ['utterance', 'text', 'transcript', 'sentence'],
    'start': ['starttime', 'start_time', 'start', 'begin'],
    'end': ['endtime', 'end_time', 'end', 'stop'],
    'truth': ['emotion', 'label', 'truth', 'gold'],
}


# ----------------------------------------------------------------------------------------
# Pure helpers (no torch needed; unit-testable)
# ----------------------------------------------------------------------------------------
def parse_time(value) -> float:
    """Accept seconds ('12.5') or MELD style ('0:16:59,894', '01:02.5')."""
    s = str(value).strip().replace(',', '.')
    if not s:
        raise ValueError('thời gian trống')
    if ':' in s:
        parts = s.split(':')
        if len(parts) > 3:
            raise ValueError(f'thời gian không hợp lệ: {value}')
        secs = 0.0
        for p in parts:
            secs = secs * 60 + float(p)
        return secs
    return float(s)


def _norm_header(h: str) -> str:
    return re.sub(r'\s+', '_', (h or '').strip().lower().lstrip('\ufeff'))


def parse_utterance_csv(data: bytes, labels: List[str]) -> List[dict]:
    """Read a training-style CSV. Needs StartTime/EndTime; Speaker/Utterance/Dialogue_ID/Emotion optional."""
    if not data or len(data) > MAX_CSV_BYTES:
        raise ValueError('File CSV trống hoặc lớn hơn 2 MB.')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = data.decode('latin-1')
    first = text.splitlines()[0] if text.strip() else ''
    delim = max([',', ';', '\t'], key=first.count)
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise ValueError('CSV cần có dòng tiêu đề và ít nhất một utterance.')
    header = [_norm_header(h) for h in rows[0]]
    col = {}
    for key, names in CSV_ALIASES.items():
        for n in names:
            if n in header:
                col[key] = header.index(n)
                break
    if 'start' not in col or 'end' not in col:
        raise ValueError('CSV phải có cột StartTime và EndTime (giây hoặc hh:mm:ss,mmm).')
    out = []
    for line_no, r in enumerate(rows[1:], start=2):
        def cell(key):
            i = col.get(key)
            return r[i].strip() if i is not None and i < len(r) else ''
        try:
            start, end = parse_time(cell('start')), parse_time(cell('end'))
        except ValueError as exc:
            raise ValueError(f'CSV dòng {line_no}: {exc}') from exc
        truth = cell('truth').lower()
        out.append({
            'start': start, 'end': end,
            'speaker': cell('speaker') or 'Speaker 1',
            'text': cell('text'),
            'dialogue_id': cell('dialogue') or None,
            'truth': truth if truth in labels else None,
        })
    return out


def build_utterances(words, max_s=MAX_UTT_S, gap_s=0.8, max_words=30) -> List[dict]:
    """words: [(start, end, text)] -> sentence-like utterances, closer to MELD units."""
    utts, cur = [], []

    def flush():
        if not cur:
            return
        text = ' '.join(w[2] for w in cur).strip()
        if re.search(r'\w', text):
            utts.append({'start': float(cur[0][0]), 'end': float(cur[-1][1]), 'text': text})
        cur.clear()

    for w in words:
        if not w[2]:
            continue
        if cur and (w[0] - cur[-1][1] > gap_s or w[1] - cur[0][0] > max_s or len(cur) >= max_words):
            flush()
        cur.append(w)
        if re.search(r'[.!?]["\')\]]*$', w[2]):
            flush()
    flush()
    return utts


def cluster_speakers(embeds: np.ndarray, n: int) -> List[int]:
    """Agglomerative clustering into n groups; labels renumbered by first appearance."""
    count = len(embeds)
    k = max(1, min(int(n), count))
    if k == 1:
        return [0] * count
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(np.asarray(embeds, dtype=np.float64))
    raw = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(x)
    order, labels = {}, []
    for c in raw:
        order.setdefault(int(c), len(order))
        labels.append(order[int(c)])
    return labels


def slice_audio(wav: np.ndarray, start: float, end: float) -> np.ndarray:
    """Cut [start, end]; very short pieces are widened around their centre and zero-padded if needed."""
    dur = len(wav) / SAMPLE_RATE
    if end - start < MIN_AUDIO_S:
        c = (start + end) / 2
        start, end = c - MIN_AUDIO_S / 2, c + MIN_AUDIO_S / 2
    start, end = max(0.0, start), min(dur, end)
    seg = np.asarray(wav[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], dtype=np.float32)
    need = int(MIN_AUDIO_S * SAMPLE_RATE)
    if len(seg) < need:
        seg = np.pad(seg, (0, need - len(seg)))
    return seg


def group_dialogues(rows: List[dict], max_per: int) -> List[List[int]]:
    """Indices grouped by Dialogue_ID (or one group), then split into balanced chunks <= max_per."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get('dialogue_id') or '__all__', []).append(i)
    out = []
    for idxs in groups.values():
        parts = math.ceil(len(idxs) / max_per)
        size = math.ceil(len(idxs) / parts)
        out.extend(idxs[k:k + size] for k in range(0, len(idxs), size))
    return out


def normalize_rows(rows: List[dict], duration: float) -> List[dict]:
    if not rows:
        raise ValueError('Chưa có utterance nào để dự đoán.')
    if len(rows) > MAX_UTTERANCES:
        raise ValueError(f'Tối đa {MAX_UTTERANCES} utterance mỗi lần.')
    clean = []
    for i, r in enumerate(rows):
        start, end = float(r['start']), float(r['end'])
        text = str(r.get('text', '')).strip()
        speaker = str(r.get('speaker', '')).strip()
        if not text:
            raise ValueError(f'Dòng {i + 1}: thiếu transcript.')
        if len(text) > 1500:
            raise ValueError(f'Dòng {i + 1}: transcript quá dài (tối đa 1500 ký tự).')
        if not speaker:
            raise ValueError(f'Dòng {i + 1}: thiếu tên speaker.')
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or start >= duration:
            raise ValueError(f'Dòng {i + 1}: thời gian bắt đầu nằm ngoài video (0–{duration:.1f}s).')
        end = min(end, duration)
        if end <= start:
            raise ValueError(f'Dòng {i + 1}: thời gian kết thúc phải lớn hơn thời gian bắt đầu.')
        clean.append({'src_index': i, 'start': start, 'end': end, 'text': text, 'speaker': speaker,
                      'dialogue_id': (str(r['dialogue_id']).strip() or None) if r.get('dialogue_id') else None,
                      'truth': r.get('truth')})
    clean.sort(key=lambda x: (x['start'], x['src_index']))
    return clean


def resolve_slots(speakers: List[str], requested: Dict[str, int], n_slots: int) -> Dict[str, int]:
    """Each speaker -> model slot; honour the request, fill gaps with unused slots."""
    result = {}
    for s in speakers:
        v = requested.get(s)
        if v is not None:
            if not 0 <= int(v) < n_slots:
                raise ValueError(f'Slot của "{s}" phải trong khoảng 0..{n_slots - 1}.')
            result[s] = int(v)
    used = set(result.values())
    free = [i for i in range(n_slots) if i not in used]
    for s in speakers:
        if s not in result:
            result[s] = free.pop(0) if free else n_slots - 1
    return result


class FaceCropper:
    """Optional, experimental: largest Haar-cascade face with margin, else the full frame."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def crop(self, frame):
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return frame
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        m = int(0.25 * max(w, h))
        H, W = frame.shape[:2]
        return frame[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]


def grab_frames(cap, start: float, end: float, n: int = FRAMES_PER_UTT) -> list:
    import cv2
    times = np.linspace(start, max(start, end - 0.05), n)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    return frames


def probe_video(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Không mở được video. Hãy dùng mp4/avi/mov/mkv/webm hợp lệ.')
        ok, _ = cap.read()
        if not ok:
            raise ValueError('Video không có luồng hình hợp lệ (cần cả hình và tiếng).')
        return {'fps': cap.get(cv2.CAP_PROP_FPS) or 0.0,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def extract_audio(path: Path) -> np.ndarray:
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error', '-i', str(path),
           '-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
           '-t', str(MAX_VIDEO_SECONDS + 5), '-f', 'f32le', 'pipe:1']
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=900, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError('Không tìm thấy ffmpeg. Cài FFmpeg và thêm vào PATH.') from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError('FFmpeg tách âm thanh quá lâu.') from exc
    if res.returncode != 0:
        tail = res.stderr.decode('utf-8', errors='replace')[-300:]
        if 'matches no streams' in tail:
            raise ValueError('Video không có luồng âm thanh.')
        raise ValueError(f'Không tách được âm thanh: {tail}')
    wav = np.frombuffer(res.stdout, dtype='<f4').copy()
    if len(wav) < SAMPLE_RATE // 3 or not np.isfinite(wav).all():
        raise ValueError('Âm thanh quá ngắn hoặc không hợp lệ.')
    if len(wav) / SAMPLE_RATE > MAX_VIDEO_SECONDS:
        raise ValueError(f'Video dài quá {MAX_VIDEO_SECONDS // 60} phút. Hãy cắt ngắn trước khi tải lên.')
    if float(np.sqrt(np.mean(wav * wav))) < 0.00035:
        raise ValueError('Video gần như không có tiếng.')
    return wav


# ----------------------------------------------------------------------------------------
# Adapter around the real DiGemoEngine (torch imported lazily)
# ----------------------------------------------------------------------------------------
class EngineAdapter:
    def __init__(self, engine):
        self.e = engine
        self.labels = engine.labels
        self.ns = engine.ns
        self.lock = engine.lock

    def transcribe_words(self, wav, on_progress: Optional[Callable[[float], None]] = None):
        segments, _ = self.e.asr.transcribe(
            wav, language='en', beam_size=3, vad_filter=True,
            word_timestamps=True, condition_on_previous_text=False)
        total, words = max(len(wav) / SAMPLE_RATE, 1e-6), []
        for seg in segments:
            if seg.words:
                words.extend((float(w.start), float(w.end), w.word.strip()) for w in seg.words)
            elif seg.text.strip():
                words.append((float(seg.start), float(seg.end), seg.text.strip()))
            if on_progress:
                on_progress(min(1.0, float(seg.end) / total))
        return words

    def text(self, s):
        return self.e.encoders.text(s)

    def audio(self, wav):
        return self.e.encoders.audio(wav)

    def visual(self, frames):
        return self.e.encoders.visual(frames)

    def speaker_embedding(self, wav):
        """Mean+std of mid-layer Wav2Vec2 states: rough voice fingerprint for clustering."""
        import torch
        enc = self.e.encoders
        wav = np.asarray(wav[:SAMPLE_RATE * 15], dtype=np.float32)
        inp = enc.audio_pre(wav, sampling_rate=SAMPLE_RATE, return_tensors='pt')
        inp = {k: v.to(enc.device) for k, v in inp.items()}
        with torch.inference_mode():
            hs = enc.audio_enc(**inp, output_hidden_states=True).hidden_states
            x = torch.stack(list(hs[4:9])).mean(0).squeeze(0)
            emb = torch.cat([x.mean(0), x.std(0, unbiased=False)])
        return emb.float().cpu().numpy()

    def classify(self, feats: List[dict], slots: List[int]) -> np.ndarray:
        import torch
        e, n = self.e, len(feats)

        def stacked(col):
            return torch.from_numpy(np.stack([f[col] for f in feats])).unsqueeze(1).to(e.device)

        qmask = torch.zeros((n, 1, e.ns), dtype=torch.float32, device=e.device)
        for i, s in enumerate(slots):
            qmask[i, 0, s] = 1.0
        umask = torch.ones((n, 1), dtype=torch.float32, device=e.device)
        with torch.inference_mode():
            logits = e.model(stacked('text'), stacked('visual'), stacked('audio'), umask, qmask, [n])[0]
        return torch.softmax(logits, -1).float().cpu().numpy()


# ----------------------------------------------------------------------------------------
# Jobs + service
# ----------------------------------------------------------------------------------------
class Job:
    def __init__(self):
        self.id = secrets.token_urlsafe(12)
        self.status = 'running'
        self.progress = 0.0
        self.message = 'Đang bắt đầu…'
        self.result = None
        self.error = None
        self.created = time.monotonic()

    def set(self, progress: float, message: str = None):
        self.progress = float(max(0.0, min(1.0, progress)))
        if message:
            self.message = message

    def public(self):
        return {'id': self.id, 'status': self.status, 'progress': round(self.progress, 3),
                'message': self.message, 'result': self.result, 'error': self.error}


class VideoService:
    def __init__(self, adapter_getter: Callable):
        self.get_adapter = adapter_getter
        self.root = Path(tempfile.mkdtemp(prefix='digemo_video_'))
        self.videos: Dict[str, dict] = {}
        self.jobs: Dict[str, Job] = {}
        self.guard = threading.Lock()
        atexit.register(lambda: shutil.rmtree(self.root, ignore_errors=True))

    # -- storage -----------------------------------------------------------------------
    def new_upload(self, suffix: str):
        with self.guard:
            now = time.monotonic()
            for tok in [t for t, v in self.videos.items() if now - v['created'] > VIDEO_TTL_SECONDS]:
                self._drop(tok)
            token = secrets.token_urlsafe(16)
            vdir = self.root / token
            vdir.mkdir(parents=True)
            self.videos[token] = {'dir': vdir, 'video': vdir / ('video' + suffix), 'created': now,
                                  'cache': {}, 'duration': None}
            return token, self.videos[token]['video']

    def _drop(self, token: str):
        rec = self.videos.pop(token, None)
        if rec:
            shutil.rmtree(rec['dir'], ignore_errors=True)

    def discard(self, token: str):
        with self.guard:
            self._drop(token)

    # -- jobs --------------------------------------------------------------------------
    def start(self, fn, *args) -> Job:
        job = Job()
        with self.guard:
            for jid in [j for j, x in self.jobs.items() if x.status != 'running' and time.monotonic() - x.created > 3600]:
                self.jobs.pop(jid, None)
            self.jobs[job.id] = job

        def run():
            try:
                job.result = fn(job, *args)
                job.status = 'done'
                job.progress = 1.0
            except (ValueError, RuntimeError) as exc:
                job.status, job.error = 'error', str(exc)
            except Exception:
                LOG.exception('Video job failed')
                job.status, job.error = 'error', 'Lỗi xử lý nội bộ. Xem cửa sổ PowerShell để biết chi tiết.'

        threading.Thread(target=run, daemon=True).start()
        return job

    # -- step 1: video (+ optional CSV) -> editable utterance table ---------------------
    def analyze(self, job: Job, token: str, csv_bytes: Optional[bytes], n_speakers: int):
        ad = self.get_adapter()
        rec = self.videos[token]
        job.set(0.02, 'Kiểm tra video…')
        info = probe_video(rec['video'])
        job.set(0.06, 'Tách âm thanh bằng FFmpeg…')
        wav = extract_audio(rec['video'])
        np.save(rec['dir'] / 'audio.npy', wav)
        rec['wav'] = rec['dir'] / 'audio.npy'
        rec['duration'] = len(wav) / SAMPLE_RATE

        if csv_bytes:
            rows = parse_utterance_csv(csv_bytes, ad.labels)
            if len(rows) > MAX_UTTERANCES:
                raise ValueError(f'CSV có hơn {MAX_UTTERANCES} dòng.')
            missing = [r for r in rows if not r['text']]
            for k, r in enumerate(missing):
                job.set(0.1 + 0.7 * k / len(missing), f'Whisper nhận dạng câu thiếu transcript ({k + 1}/{len(missing)})…')
                with ad.lock:
                    words = ad.transcribe_words(slice_audio(wav, r['start'], r['end']))
                r['text'] = ' '.join(w[2] for w in words).strip()
            source = 'csv'
        # else:
        #     job.set(0.1, 'Whisper đang nhận dạng giọng nói…')
        #     with ad.lock:
        #         words = ad.transcribe_words(wav, lambda f: job.set(0.1 + 0.5 * f))
        #     utts = build_utterances(words)
        #     if not utts:
        #         raise ValueError('Không nhận được lời nói nào trong video.')
        #     utts = utts[:MAX_UTTERANCES]
        #     embeds = []
        #     with ad.lock:
        #         for k, u in enumerate(utts):
        #             job.set(0.62 + 0.33 * k / len(utts), f'Phân biệt người nói ({k + 1}/{len(utts)})…')
        #             embeds.append(ad.speaker_embedding(slice_audio(wav, u['start'], u['end'])))
        #     labels = cluster_speakers(np.stack(embeds), n_speakers)
        #     rows = [{'start': u['start'], 'end': u['end'], 'text': u['text'],
        #              'speaker': f'Speaker {l + 1}', 'dialogue_id': None, 'truth': None}
        #             for u, l in zip(utts, labels)]
        else:
            job.set(0.1, 'Whisper đang nhận dạng giọng nói…')
            with ad.lock:
                words = ad.transcribe_words(wav, lambda f: job.set(0.1 + 0.5 * f))
            if not words:
                raise ValueError('Không nhận được lời nói nào trong video.')
            rows = None
            try:
                job.set(0.62, 'pyannote đang phân biệt người nói…')
                segs = diarize(rec['wav'], n_speakers)   # rec['wav'] là audio.npy đã lưu ở trên
                rows = rows_from_diarization(words, segs, build_utterances)[:MAX_UTTERANCES]
                source = 'pyannote'
            except DiarizerUnavailable as exc:
                LOG.warning('pyannote không dùng được (%s), quay về gom cụm Wav2Vec2', exc)
            if not rows:
                utts = build_utterances(words)
                if not utts:
                    raise ValueError('Không nhận được lời nói nào trong video.')
                utts = utts[:MAX_UTTERANCES]
                embeds = []
                with ad.lock:
                    for k, u in enumerate(utts):
                        job.set(0.62 + 0.33 * k / len(utts), f'Phân biệt người nói ({k + 1}/{len(utts)})…')
                        embeds.append(ad.speaker_embedding(slice_audio(wav, u['start'], u['end'])))
                labels = cluster_speakers(np.stack(embeds), n_speakers)
                rows = [{'start': u['start'], 'end': u['end'], 'text': u['text'],
                         'speaker': f'Speaker {l + 1}', 'dialogue_id': None, 'truth': None}
                        for u, l in zip(utts, labels)]
                source = 'auto'

        speakers = list(dict.fromkeys(r['speaker'] for r in rows))
        return {'token': token, 'duration': round(rec['duration'], 2), 'video': info, 'source': source,
                'rows': [{**r, 'start': round(r['start'], 2), 'end': round(r['end'], 2)} for r in rows],
                'speakers': speakers, 'slots': resolve_slots(speakers, {}, ad.ns), 'n_slots': ad.ns}

    # -- step 2: edited table -> encoders -> DiGemo ------------------------------------
    def predict(self, job: Job, token: str, raw_rows: List[dict], requested_slots: Dict[str, int],
                max_per: int, face_crop: bool):
        import cv2
        ad = self.get_adapter()
        rec = self.videos.get(token)
        if rec is None or rec.get('wav') is None:
            raise ValueError('Video đã hết hạn hoặc chưa được phân tích. Hãy tải lại video.')
        max_per = max(1, min(int(max_per), MAX_PER_DIALOGUE_HARD))
        rows = normalize_rows(raw_rows, rec['duration'])
        speakers = list(dict.fromkeys(r['speaker'] for r in rows))
        slots = resolve_slots(speakers, requested_slots, ad.ns)
        wav = np.load(rec['wav'], mmap_mode='r')
        cropper = FaceCropper() if face_crop else None
        cap = cv2.VideoCapture(str(rec['video']))
        feats = []
        try:
            with ad.lock:
                for i, r in enumerate(rows):
                    job.set(0.02 + 0.88 * i / len(rows), f'Trích đặc trưng Text/Audio/Visual ({i + 1}/{len(rows)})…')
                    key = (round(r['start'], 2), round(r['end'], 2), r['text'], bool(face_crop))
                    f = rec['cache'].get(key)
                    if f is None:
                        frames = grab_frames(cap, r['start'], r['end'])
                        if not frames:
                            raise ValueError(f'Dòng {r["src_index"] + 1}: không đọc được hình ảnh tại {r["start"]:.2f}s.')
                        if cropper:
                            frames = [cropper.crop(x) for x in frames]
                        f = {'text': ad.text(r['text']),
                             'audio': ad.audio(slice_audio(wav, r['start'], r['end'])),
                             'visual': ad.visual(frames)}
                        for name, dim in EXPECTED_DIMS.items():
                            if tuple(np.shape(f[name])) != (dim,):
                                raise RuntimeError(f'Embedding {name} sai kích thước: {np.shape(f[name])}')
                        rec['cache'][key] = f
                    feats.append(f)
                job.set(0.92, 'DiGemo đang dự đoán…')
                probs = [None] * len(rows)
                dia_of = [0] * len(rows)
                for d, idxs in enumerate(group_dialogues(rows, max_per)):
                    p = ad.classify([feats[i] for i in idxs], [slots[rows[i]['speaker']] for i in idxs])
                    for k, i in enumerate(idxs):
                        probs[i], dia_of[i] = p[k], d
        finally:
            cap.release()

        results, counts, graded, correct = [], {}, 0, 0
        pos_in_dia: Dict[int, int] = {}
        for i, r in enumerate(rows):
            p = probs[i]
            top = int(np.argmax(p))
            label = ad.labels[top]
            counts[label] = counts.get(label, 0) + 1
            ok = None
            if r.get('truth'):
                ok = (r['truth'] == label)
                graded += 1
                correct += int(ok)
            u = pos_in_dia.get(dia_of[i], 0)
            pos_in_dia[dia_of[i]] = u + 1
            results.append({
                'src_index': r['src_index'], 'dialogue_id': dia_of[i], 'utterance_id': u,
                'speaker': r['speaker'], 'slot': slots[r['speaker']],
                'start': round(r['start'], 2), 'end': round(r['end'], 2), 'text': r['text'],
                'emotion': label, 'confidence': round(float(p[top]) * 100, 2),
                'probabilities': [{'label': l, 'percent': round(float(p[j]) * 100, 2)}
                                  for j, l in enumerate(ad.labels)],
                'truth': r.get('truth'), 'correct': ok})
        return {'results': results, 'counts': counts, 'slots': slots,
                'accuracy': round(100.0 * correct / graded, 2) if graded else None,
                'graded': graded, 'csv': results_csv(results, ad.labels)}


def results_csv(results: List[dict], labels: List[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'StartTime', 'EndTime',
                'Emotion', 'Predicted', 'Confidence'] + [f'prob_{l}' for l in labels])
    for r in results:
        probs = {p['label']: p['percent'] for p in r['probabilities']}
        w.writerow([r['dialogue_id'], r['utterance_id'], r['speaker'], r['text'], r['start'], r['end'],
                    r['truth'] or '', r['emotion'], r['confidence']] + [probs[l] for l in labels])
    return buf.getvalue()


# ----------------------------------------------------------------------------------------
# HTTP layer (models at module level so FastAPI can resolve them)
# ----------------------------------------------------------------------------------------
class RowIn(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    dialogue_id: Optional[str] = None
    truth: Optional[str] = None


class PredictIn(BaseModel):
    token: str
    rows: List[RowIn]
    slots: Dict[str, int] = {}
    max_per_dialogue: int = 30
    face_crop: bool = False


def register_video_routes(app, active_engine, adapter_factory=EngineAdapter):
    def adapter():
        engine = active_engine()
        if getattr(engine, '_video_adapter', None) is None:
            engine._video_adapter = adapter_factory(engine)
        return engine._video_adapter

    svc = VideoService(adapter)

    @app.get('/video')
    def video_page():
        return FileResponse(str(HERE / 'static' / 'video.html'))

    @app.post('/api/video/analyze')
    async def video_analyze(video: UploadFile = File(...), csv_file: Optional[UploadFile] = File(None),
                            num_speakers: int = Form(2)):
        adapter()  # 503 if models are still loading
        if not 1 <= num_speakers <= 8:
            raise HTTPException(status_code=400, detail='Số người nói phải từ 1 đến 8.')
        suffix = Path(video.filename or '').suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail='Chỉ hỗ trợ: ' + ', '.join(sorted(ALLOWED_SUFFIXES)))
        token, path = svc.new_upload(suffix)
        size = 0
        try:
            with open(path, 'wb') as f:
                while True:
                    chunk = await video.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise HTTPException(status_code=413, detail='Video lớn hơn 800 MB.')
                    f.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail='File video trống.')
            csv_bytes = None
            if csv_file is not None and csv_file.filename:
                csv_bytes = await csv_file.read(MAX_CSV_BYTES + 1)
        except HTTPException:
            svc.discard(token)
            raise
        job = svc.start(svc.analyze, token, csv_bytes, num_speakers)
        return {'job_id': job.id, 'token': token}

    @app.post('/api/video/predict')
    def video_predict(req: PredictIn):
        if req.token not in svc.videos:
            raise HTTPException(status_code=400, detail='Video đã hết hạn. Hãy tải lại video.')
        job = svc.start(svc.predict, req.token, [(r.model_dump() if hasattr(r, 'model_dump') else r.dict()) for r in req.rows], req.slots,
                        req.max_per_dialogue, req.face_crop)
        return {'job_id': job.id}

    @app.get('/api/video/jobs/{job_id}')
    def video_job(job_id: str):
        job = svc.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Không tìm thấy tác vụ.')
        return job.public()

    @app.delete('/api/video/{token}')
    def video_delete(token: str):
        svc.discard(token)
        return {'ok': True}

    @app.post('/api/video/{token}/delete')  # for navigator.sendBeacon on page close
    def video_delete_beacon(token: str):
        svc.discard(token)
        return {'ok': True}

    return svc
