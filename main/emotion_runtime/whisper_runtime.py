"""Lazy faster-whisper wrapper for block/final transcription."""
import numpy as np


class WhisperRuntime:
    def __init__(self, model_name="small", device="auto"):
        from faster_whisper import WhisperModel
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav, sample_rate=16000):
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if not wav.size:
            return ""
        segments, _ = self.model.transcribe(
            wav, language="en", beam_size=5, best_of=5,
            vad_filter=True,
            vad_parameters={"threshold": 0.3, "min_silence_duration_ms": 300, "speech_pad_ms": 200},
            condition_on_previous_text=False, temperature=0.0,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
