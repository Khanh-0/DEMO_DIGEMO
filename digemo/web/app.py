"""DiGemo MELD web UI — reuse the original feature extractors and DiGemo checkpoint.

Run from DiGemo root (PowerShell):
  python ./digemo_web/app.py --checkpoint checkpoints/best_model_MELD_9161.pth --features-dir MELD_pro/features
Browser: http://127.0.0.1:8000  (tự chuyển tới /video)

DiGemo web (video upload) dùng encoder mới Densenet128_FER+.

  python .\digemo_web\app.py --checkpoint checkpoints/digemo_meld.pth `
>>   --encoder-dir Densnet128_FER+ `
>>   --feat-dir Densnet128_FER+/feats `
>>   --text-ckpt Densnet128_FER+/text_ft/best.pt `
>>   --face-ckpt Densnet128_FER+/ckpt_fer/best.pt `
>>   --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]      # D:\DEMO_MELD
HERE = Path(__file__).resolve().parent          # digemo/web
PKG = ROOT / 'digemo'
for p in (ROOT, HERE, PKG / 'model', PKG / 'encoders', PKG / 'pipeline'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from video_pipeline import register_video_routes

LOG = logging.getLogger('digemo.web')
EMOTIONS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']


def in_project(path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def choose_torch_device(name: str) -> str:
    import torch
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; set --torch-device cpu')
    return name


class DiGemoEngine:
    """1 DiGemo checkpoint + encoder mới + Whisper. Các thuộc tính video_pipeline cần:
    labels, ns, model, device, lock, asr, encoders."""

    def __init__(self, config):
        import numpy as np
        from encoders_online import OnlineEncodersV2
        from predict import load_model_from_checkpoint

        self.labels = EMOTIONS
        self.device = choose_torch_device(config.torch_device)
        self.model, cfg, dims, classes, ns, _ = load_model_from_checkpoint(
            str(config.checkpoint), self.device)
        if cfg.dataset != 'MELD' or dims != [1024, 342, 300] or classes != 7 or ns != 9:
            raise ValueError('Cần checkpoint MELD [1024,342,300], 7 nhãn, 9 speaker slots.')
        if getattr(cfg, 'modals', 'tva') != 'tva':
            raise ValueError('Checkpoint phải dùng đủ 3 modality (tva).')
        self.ns = ns

        enc_args = SimpleNamespace(
            encoder_dir=config.encoder_dir, feat_dir=config.feat_dir,
            text_ckpt=config.text_ckpt, face_ckpt=config.face_ckpt,
            audio_model='facebook/wav2vec2-base')
        self.encoders = OnlineEncodersV2(enc_args, self.device)

        from faster_whisper import WhisperModel
        LOG.info('Loading Whisper %s (thử GPU trước)...', config.whisper_model)
        try:
            self.asr = WhisperModel(config.whisper_model, device='cuda', compute_type='float16')
            list(self.asr.transcribe(np.zeros(16000, dtype=np.float32), language='en')[0])
        except Exception as exc:
            LOG.warning('Whisper GPU không dùng được (%s), chuyển sang CPU', exc)
            self.asr = WhisperModel(config.whisper_model, device='cpu', compute_type='int8')
        self.lock = threading.Lock()
        self.config = config

    def status(self):
        return {'ready': True, 'device': self.device, 'checkpoint': Path(self.config.checkpoint).name,
                'asr': self.config.whisper_model, 'dataset': 'MELD', 'emotions': self.labels}


def create_app(engine=None) -> FastAPI:
    app = FastAPI(title='DiGemo Emotion Studio', version='2.0.0', docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.mount('/static', StaticFiles(directory=str(HERE / 'static')), name='static')

    def active_engine():
        if app.state.engine is None:
            raise HTTPException(status_code=503, detail='AI models are still loading.')
        return app.state.engine

    register_video_routes(app, active_engine)

    @app.get('/')
    def index():
        return RedirectResponse('/video')

    @app.get('/api/status')
    def status():
        return active_engine().status()

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description='DiGemo web (encoder Densenet128_FER+)')
    ap.add_argument('--checkpoint',  default='checkpoints/digemo_meld.pth')
    ap.add_argument('--encoder-dir', default='digemo/encoders')
    ap.add_argument('--feat-dir',    default='data/feats')
    ap.add_argument('--text-ckpt',   default='checkpoints/text_ft.pt')
    ap.add_argument('--face-ckpt',   default='checkpoints/face_fer.pt')
    ap.add_argument('--meld-csv',        default='data/csv/test_sent_emo.csv')
    ap.add_argument('--meld-timestamps', default='data/utterance_timestamps.csv')
    ap.add_argument('--torch-device', choices=('auto', 'cuda', 'cpu'), default='auto')
    ap.add_argument('--whisper-model', default='large-v3')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()

    if a.meld_csv:
        os.environ['DIGEMO_MELD_CSV'] = str(in_project(a.meld_csv))
    if a.meld_timestamps:
        os.environ['DIGEMO_MELD_TS'] = str(in_project(a.meld_timestamps))
    for k in ('checkpoint', 'encoder_dir', 'feat_dir', 'text_ckpt', 'face_ckpt'):
        setattr(a, k, in_project(getattr(a, k)))

    required = [a.checkpoint, a.text_ckpt, a.face_ckpt,
                a.feat_dir / 'audio_head.npz', a.feat_dir / 'norm.npz', a.feat_dir / 'speaker_map.json',
                a.encoder_dir / 'encoders.py', a.encoder_dir / 'common.py', a.encoder_dir / 'densenet_fer.py']
    for p in required:
        if not p.is_file():
            ap.error(f'Thiếu file bắt buộc: {p}')

    print('\n[web] Loading DiGemo + encoders (một lần)...', flush=True)
    engine = DiGemoEngine(a)
    print(f'\n[web] Ready! Open http://127.0.0.1:{a.port}\n', flush=True)
    if a.host not in ('127.0.0.1', 'localhost', '::1'):
        print('[SECURITY] Server mở ra mạng và KHÔNG có đăng nhập.', flush=True)
    uvicorn.run(create_app(engine), host=a.host, port=a.port, log_level='info')


if __name__ == '__main__':
    main()