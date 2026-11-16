"""overlap_models.py - nạp model nặng cho overlap_asr.py (lười: chỉ nạp khi thật sự có vùng chồng).

Mặc định chạy trên CPU để không tranh VRAM 4 GB với Whisper + RoBERTa + Wav2Vec2 + DenseNet.
Đặt DIGEMO_OVERLAP_DEVICE=cuda nếu còn dư VRAM.
Cần: pip install speechbrain   (CHƯA kiểm chứng với torch 2.8 / torchaudio 2.8 trên máy bạn)
"""
import logging
import os
import threading
from typing import Callable, List

import numpy as np

LOG = logging.getLogger('digemo.overlap_models')
_lock = threading.Lock()
_cache: dict = {}


def overlap_device() -> str:
    return os.environ.get('DIGEMO_OVERLAP_DEVICE', 'cpu')


def get_embed(device: str = None) -> Callable[[np.ndarray], np.ndarray]:
    """ECAPA-TDNN (VoxCeleb): hàm embed(wav16k) -> vector giọng nói."""
    device = device or overlap_device()
    with _lock:
        if 'embed' not in _cache:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
            clf = EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                 run_opts={'device': device})

            def embed(x: np.ndarray) -> np.ndarray:
                t = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    return clf.encode_batch(t).squeeze().float().cpu().numpy()
            _cache['embed'] = embed
        return _cache['embed']


def get_separator(device: str = None) -> Callable[[np.ndarray], List[np.ndarray]]:
    device = device or overlap_device()
    with _lock:
        if 'sep' not in _cache:
            from overlap_asr import make_speechbrain_separator
            _cache['sep'] = make_speechbrain_separator(device=device)
        return _cache['sep']