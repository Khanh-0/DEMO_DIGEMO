"""Per-track rolling buffers for audio, face crops and emotion state."""
from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrackState:
    track_id: int
    max_audio_samples: int
    max_crops: int = 24
    audio: deque = field(default_factory=deque)
    crops: deque = field(default_factory=lambda: deque(maxlen=24))
    text_history: deque = field(default_factory=lambda: deque(maxlen=8))
    start_frame: int | None = None
    last_frame: int | None = None
    speaking: bool = False
    asd_score: float = 0.0
    emotion: str | None = None
    confidence: float = 0.0

    def __post_init__(self):
        self.audio = deque(maxlen=self.max_audio_samples)
        self.crops = deque(maxlen=self.max_crops)

    def append_audio(self, samples):
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.audio.extend(values.tolist())

    def append_crop(self, crop, frame_index):
        self.crops.append(crop)
        if self.start_frame is None:
            self.start_frame = frame_index
        self.last_frame = frame_index

    def audio_array(self):
        return np.asarray(self.audio, dtype=np.float32)

    def clear_segment(self):
        self.audio.clear()
        self.crops.clear()
        self.text_history.clear()
        self.start_frame = None
        self.last_frame = None

    def result(self):
        return {
            "track_id": self.track_id,
            "speaking": bool(self.speaking),
            "asd_score": float(self.asd_score),
            "emotion": self.emotion,
            "confidence": float(self.confidence),
            "text": " ".join(self.text_history),
        }


class TrackBufferStore:
    def __init__(self, fps=25, max_seconds=8.0):
        self.fps = fps
        self.max_seconds = max_seconds
        self.max_audio_samples = int(16000 * max_seconds)
        self.tracks = {}

    def get(self, track_id):
        track_id = int(track_id)
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id, self.max_audio_samples)
        return self.tracks[track_id]

    def update_status(self, track_id, speaking, asd_score, frame_index):
        track = self.get(track_id)
        track.speaking = bool(speaking)
        track.asd_score = float(asd_score)
        track.last_frame = frame_index
        return track

    def results(self):
        return [self.tracks[k].result() for k in sorted(self.tracks)]
