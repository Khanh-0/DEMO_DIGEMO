"""DiGemo MELD web UI — reuse the original feature extractors and DiGemo checkpoint.

Run from DiGemo root (PowerShell):
  python ./digemo_web/app.py --checkpoint checkpoints/best_model_MELD_260.pth --features-dir MELD_pro/features
Browser: http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import logging
import secrets
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

LOG = logging.getLogger('digemo.web')
MAX_AUDIO_BYTES = 16 * 1024 * 1024
MAX_FRAME_BYTES = 3 * 1024 * 1024
MAX_SESSIONS = 128
SESSION_TTL_SECONDS = 3 * 3600


@dataclass
class Session:
    history: deque
    last_seen: float


def in_project(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def decode_browser_audio(audio_bytes: bytes, duration_limit_s: int = 15) -> np.ndarray:
    """Read MediaRecorder webm/opus (or mp4) using ffmpeg; output 16 kHz mono f32.

    FFMPEG on PATH is required; no raw recordings are written to disk.
    """
    if not audio_bytes or len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError('Audio trống hoặc lớn hơn giới hạn 16 MB.')
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-nostdin', '-loglevel', 'error', '-i', 'pipe:0',
             '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
             '-t', str(duration_limit_s), '-f', 'f32le', 'pipe:1'],
            input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError('Không tìm thấy ffmpeg. Cài FFmpeg và thêm vào PATH (kiểm tra: ffmpeg -version).') from e
    except subprocess.TimeoutExpired as e:
        raise ValueError('FFmpeg giải mã audio quá lâu.') from e
    if result.returncode != 0:
        tail = result.stderr.decode('utf-8', errors='replace')[-350:]
        raise ValueError(f'Không giải mã được audio trình duyệt: {tail}')
    wav = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(wav) < 16000 // 3 or not np.isfinite(wav).all():
        raise ValueError('Đoạn thu quá ngắn hoặc không hợp lệ (cần ít nhất 0.33 giây).')
    rms = float(np.sqrt(np.mean(wav * wav)))
    if rms < 0.00035:
        raise ValueError('Microphone gần như không có tiếng. Hãy nói to hơn hoặc chọn đúng mic.')
    return wav


def decode_frames(jpegs: list[bytes]) -> list[np.ndarray]:
    import cv2
    frames = []
    for content in jpegs:
        if not content or len(content) > MAX_FRAME_BYTES:
            raise ValueError('Frame webcam không hợp lệ hoặc quá lớn.')
        frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0] < 32 or frame.shape[1] < 32:
            raise ValueError('Không giải mã được ảnh webcam.')
        frames.append(frame)
    if not frames:
        raise ValueError('Không có frame webcam. Hãy cấp quyền camera.')
    return frames


class DiGemoEngine:
    """One instance of each encoder, one DiGemo checkpoint, independent browser sessions."""

    def __init__(self, config):
        from live_erc import EMOTIONS, OnlineEncoders, choose_torch_device
        from predict import load_model_from_checkpoint

        self.labels = EMOTIONS
        self.device = choose_torch_device(config.torch_device)
        self.window = config.window
        self.speaker_index = config.speaker_index
        self.model, cfg, dims, classes, ns, _ = load_model_from_checkpoint(
            str(config.checkpoint), self.device)
        if cfg.dataset != 'MELD' or dims != [1024, 342, 300] or classes != 7 or ns != 9:
            raise ValueError('Web UI MELD yêu cầu checkpoint [1024,342,300], 7 nhãn, 9 speaker slots.')
        if getattr(cfg, 'modals', 'tva') != 'tva':
            raise ValueError('Checkpoint phải sử dụng đầy đủ 3 modality (tva).')
        self.ns = ns
        enc_args = SimpleNamespace(
            features_dir=config.features_dir,
            roberta_checkpoint=config.roberta_checkpoint,
            text_model='roberta-large',
            audio_model='facebook/wav2vec2-base-960h',
            num_frames=8,
        )
        self.encoders = OnlineEncoders(enc_args, self.device)
        from faster_whisper import WhisperModel
        LOG.info('Loading Whisper %s on CPU/int8...', config.whisper_model)
        self.asr = WhisperModel(config.whisper_model, device='cpu', compute_type='int8')
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self.config = config

    def status(self):
        return {
            'ready': True, 'device': self.device, 'checkpoint': self.config.checkpoint.name,
            'asr': self.config.whisper_model, 'dataset': 'MELD',
            'window': self.window, 'emotions': self.labels,
        }

    def prune_sessions(self):
        now = time.monotonic()
        for session_id, session in list(self.sessions.items()):
            if now - session.last_seen > SESSION_TTL_SECONDS:
                del self.sessions[session_id]
        while len(self.sessions) >= MAX_SESSIONS:
            oldest = min(self.sessions, key=lambda k: self.sessions[k].last_seen)
            del self.sessions[oldest]

    def new_session(self):
        with self.lock:
            self.prune_sessions()
            token = secrets.token_urlsafe(24)
            self.sessions[token] = Session(deque(maxlen=self.window), time.monotonic())
            return {'session_id': token, 'window': self.window}

    def reset(self, session_id: str):
        with self.lock:
            s = self.sessions.get(session_id)
            if s is None:
                raise ValueError('Phiên không tồn tại hoặc đã hết hạn. Hãy tải lại trang.')
            s.history.clear()
            s.last_seen = time.monotonic()
            return {'ok': True, 'history_size': 0}

    def predict(self, session_id: str, audio_bytes: bytes, image_bytes: list[bytes],
                manual_transcript: str):
        from live_erc import model_predict
        with self.lock:  # serialize GPU inference + per-session history mutation
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError('Phiên không tồn tại hoặc đã hết hạn. Hãy tải lại trang.')
            wav = decode_browser_audio(audio_bytes)
            frames = decode_frames(image_bytes)
            t0 = time.perf_counter()
            if manual_transcript.strip():
                transcript = manual_transcript.strip()
                transcript_mode = 'manual'
            else:
                segments, _ = self.asr.transcribe(
                    wav, language='en', beam_size=3,
                    vad_filter=False, condition_on_previous_text=False)
                transcript = ' '.join(segment.text.strip() for segment in segments).strip()
                transcript_mode = 'whisper'
            t_asr = time.perf_counter() - t0
            if not transcript:
                raise ValueError('Whisper không nhận được lời nói. Hãy thử lại hoặc nhập transcript thủ công.')
            if len(transcript) > 1500:
                raise ValueError('Transcript quá dài (giới hạn 1500 ký tự).')

            t1 = time.perf_counter()
            t = self.encoders.text(transcript)
            a = self.encoders.audio(wav)
            v = self.encoders.visual(frames)
            if (t.shape, v.shape, a.shape) != ((1024,), (342,), (300,)):
                raise RuntimeError(f'Embedding sai kích thước: T{t.shape}, V{v.shape}, A{a.shape}')
            enc_time = time.perf_counter() - t1
            current = {'text': t, 'visual': v, 'audio': a}
            # Only append after all processing has succeeded. Inference failure leaves history unchanged.
            candidate = deque(session.history, maxlen=self.window)
            candidate.append(current)
            t2 = time.perf_counter()
            pred, prob = model_predict(self.model, candidate, self.speaker_index,
                                       self.ns, self.device)
            digemo_time = time.perf_counter() - t2
            session.history = candidate
            session.last_seen = time.monotonic()
            ranking = np.argsort(prob)[::-1].tolist()
            return {
                'transcript': transcript, 'transcript_mode': transcript_mode,
                'emotion': self.labels[pred], 'confidence': round(float(prob[pred]) * 100, 2),
                'probabilities': [
                    {'label': label, 'percent': round(float(prob[i]) * 100, 2)}
                    for i, label in enumerate(self.labels)
                ],
                'top3': [
                    {'label': self.labels[i], 'percent': round(float(prob[i]) * 100, 2)}
                    for i in ranking[:3]
                ],
                'history_size': len(candidate),
                'latency_ms': {
                    'asr': round(t_asr * 1000), 'encoders': round(enc_time * 1000),
                    'digemo': round(digemo_time * 1000),
                },
                'audio_seconds': round(len(wav) / 16000, 2),
                'frame_count': len(frames),
            }


class SessionRequest(BaseModel):
    session_id: str


def create_app(engine=None) -> FastAPI:
    app = FastAPI(title='DiGemo Emotion Studio', version='1.0.0', docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')

    def active_engine():
        if app.state.engine is None:
            raise HTTPException(status_code=503, detail='AI models are still loading.')
        return app.state.engine

    @app.get('/')
    def index():
        return FileResponse(str(HERE / 'static' / 'index.html'))

    @app.get('/api/status')
    def status():
        return active_engine().status()

    @app.post('/api/session')
    def new_session():
        return active_engine().new_session()

    @app.post('/api/reset')
    def reset(req: SessionRequest):
        try:
            return active_engine().reset(req.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post('/api/predict')
    async def predict(session_id: str = Form(...), transcript: str = Form(''),
                      audio: UploadFile = File(...), frames: list[UploadFile] = File(...)):
        if len(frames) > 32:
            raise HTTPException(status_code=400, detail='Tối đa 32 frame webcam.')
        content = await audio.read(MAX_AUDIO_BYTES + 1)
        pictures = [await f.read(MAX_FRAME_BYTES + 1) for f in frames]
        try:
            return await run_in_threadpool(
                active_engine().predict, session_id, content, pictures, transcript)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            LOG.exception('Unexpected error during DiGemo inference')
            raise HTTPException(status_code=500, detail='Lỗi inference nội bộ. Kiểm tra cửa sổ PowerShell.')

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description='Host browser UI for the existing MELD DiGemo checkpoint')
    parser.add_argument('--checkpoint', default='checkpoints/best_model_MELD_260.pth')
    parser.add_argument('--features-dir', default='MELD_pro/features')
    parser.add_argument('--roberta-checkpoint', default=None)
    parser.add_argument('--torch-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--whisper-model', default='base.en')
    parser.add_argument('--window', type=int, default=10)
    parser.add_argument('--speaker-index', type=int, default=8)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.window <= 100 or not 0 <= args.speaker_index <= 8:
        parser.error('--window must be 1..100 and --speaker-index must be 0..8')
    args.checkpoint = in_project(args.checkpoint)
    args.features_dir = in_project(args.features_dir)
    candidates = [args.features_dir / 'roberta_meld_ft.pt',
                  args.features_dir / 'roberta_large_meld_ft.pt']
    args.roberta_checkpoint = in_project(args.roberta_checkpoint) if args.roberta_checkpoint else next((p for p in candidates if p.exists()), candidates[0])
    required = [args.checkpoint, args.roberta_checkpoint,
                args.features_dir / 'online_audio_transform.joblib',
                args.features_dir / 'online_visual_transform.joblib']
    for path in required:
        if not path.is_file():
            parser.error(f'Missing required model artifact: {path}')
    print('\n[web] Loading trained DiGemo MELD and encoders (once)...', flush=True)
    engine = DiGemoEngine(args)
    print(f'\n[web] Ready! Open http://127.0.0.1:{args.port}\n', flush=True)
    if args.host not in ('127.0.0.1', 'localhost', '::1'):
        print('[SECURITY] LAN exposure requested. Use trusted networks and HTTPS for browser camera/microphone.', flush=True)
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
