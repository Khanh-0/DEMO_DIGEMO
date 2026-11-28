"""Reusable IEMOCAP Text/Audio/Visual encoder wrapper."""
from pathlib import Path
import sys
import numpy as np
import torch

from .config import RuntimeConfig


class IEMOCAPEncoders:
    """Load all three IEMOCAP encoders once and expose fixed-size vectors."""

    def __init__(self, config: RuntimeConfig | None = None):
        self.config = config or RuntimeConfig()
        self.config.validate()
        encoder_path = str(self.config.encoder_dir.resolve())
        if encoder_path not in sys.path:
            sys.path.insert(0, encoder_path)
        from encoders import FaceEncoder, OpenSmileAudioEncoder, TextEncoder
        from transformers import AutoTokenizer

        self.device = self.config.resolved_device()
        self.text_model = TextEncoder.load(str(self.config.text_checkpoint), self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model.base)
        stats = np.load(self.config.iemocap_norm)
        if not self.config.is_iemocap:
            raise ValueError("IEMOCAPEncoders is only for dataset='iemocap'")
        self.audio_model = OpenSmileAudioEncoder(stats["a_mean"], stats["a_std"])
        self.face_model = FaceEncoder(str(self.config.face_checkpoint), self.device)
        self.v_mean = stats["v_mean"].astype(np.float32)
        self.v_std = np.maximum(stats["v_std"].astype(np.float32), 1e-6)

    @torch.inference_mode()
    def text(self, transcript: str) -> np.ndarray:
        text = "" if transcript is None else str(transcript)
        tokens = self.tokenizer([text], padding=True, truncation=True, max_length=128,
                                return_tensors="pt").to(self.device)
        out = self.text_model.features(tokens["input_ids"], tokens["attention_mask"])[0]
        value = out.float().cpu().numpy().astype(np.float32)
        if value.shape != (1024,):
            raise RuntimeError(f"Unexpected text feature shape: {value.shape}")
        return value

    def audio(self, wav: np.ndarray) -> np.ndarray:
        raw = self.audio_model.extract_signal(np.asarray(wav, dtype=np.float32), 16000)
        value = self.audio_model.normalize(raw).astype(np.float32)
        if value.shape != (1582,):
            raise RuntimeError(f"Unexpected audio feature shape: {value.shape}")
        return value

    @torch.inference_mode()
    def visual(self, face_crops) -> np.ndarray:
        if not face_crops:
            return np.zeros(342, dtype=np.float32)
        raw = self.face_model.embed_pil(list(face_crops)).mean(0)
        value = ((raw - self.v_mean) / self.v_std).astype(np.float32)
        if value.shape != (342,):
            raise RuntimeError(f"Unexpected visual feature shape: {value.shape}")
        return value
