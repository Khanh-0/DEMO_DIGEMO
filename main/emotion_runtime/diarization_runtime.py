"""Pyannote diarization adapter reused by the realtime demo."""
from pathlib import Path
import sys
import numpy as np


class DiarizationRuntime:
    def __init__(self, project_root):
        root = Path(project_root)
        pipeline_dir = root / "digemo" / "pipeline"
        if str(pipeline_dir) not in sys.path:
            sys.path.insert(0, str(pipeline_dir))
        from diarizer import diarize_full
        self._diarize_full = diarize_full

    def diarize(self, wav: np.ndarray, num_speakers=None):
        exclusive, full = self._diarize_full(
            np.asarray(wav, dtype=np.float32), num_speakers
        )
        return {
            "exclusive": [
                {"start": float(s), "end": float(e), "speaker": str(label)}
                for s, e, label in exclusive
            ],
            "full": [
                {"start": float(s), "end": float(e), "speaker": str(label)}
                for s, e, label in full
            ],
        }
