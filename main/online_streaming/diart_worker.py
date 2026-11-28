"""Diart streaming diarization worker for audio-timestamped utterances."""
from __future__ import annotations
from pathlib import Path
import os
import queue
import subprocess
import threading
# Diart's official pyannote checkpoints are trusted artifacts and were saved
# before PyTorch 2.6 changed torch.load's default weights_only behavior.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import torch
from omegaconf import ListConfig

torch.serialization.add_safe_globals([ListConfig])
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import FileAudioSource
from diart.inference import StreamingInference


class DiartWorker:
    def __init__(self, wav_path, on_segment, on_error, device=None):
        self.wav_path = Path(wav_path); self.on_segment = on_segment; self.on_error = on_error
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def _run(self):
        try:
            config = SpeakerDiarizationConfig(step=.5, latency=.5, device=self.device)
            pipeline = SpeakerDiarization(config)
            source = FileAudioSource(self.wav_path, sample_rate=16000, block_duration=.5)
            inference = StreamingInference(pipeline, source, do_plot=False, show_progress=False)
            pending = None

            def emit(segment):
                if segment["end"] > segment["start"]:
                    self.on_segment(segment)

            def collect(value):
                nonlocal pending
                annotation, _ = value
                for segment, _, speaker in annotation.itertracks(yield_label=True):
                    current = {"speaker": str(speaker), "start": float(segment.start), "end": float(segment.end)}
                    if pending is None:
                        pending = current
                    elif current["speaker"] == pending["speaker"] and current["start"] <= pending["end"] + .35:
                        pending["end"] = max(pending["end"], current["end"])
                    else:
                        emit(pending); pending = current
            inference.attach_hooks(collect)
            inference()
            if pending is not None: emit(pending)
        except Exception as exc:
            self.on_error(str(exc))

    def join(self, timeout=None):
        self.thread.join(timeout)
