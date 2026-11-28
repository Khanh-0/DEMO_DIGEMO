"""Bounded identity/track cache for online streaming."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrackState:
    track_id: int
    box: list
    last_seen_frame: int
    missing_frames: int = 0
    rows: deque = field(default_factory=lambda: deque(maxlen=37))
    recent_crops: deque = field(default_factory=lambda: deque(maxlen=8))
    recent_embeddings: deque = field(default_factory=lambda: deque(maxlen=3))
    embedding_centroid: np.ndarray | None = None
    emotion: str | None = None
    confidence: float = 0.0
    text: str = ""
    speaker_id: str = ""
    box_history: deque = field(default_factory=lambda: deque(maxlen=37))

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, **values):
        for key, value in values.items(): setattr(self, key, value)

    def update_embedding(self, value):
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        self.recent_embeddings.append(vector)
        self.embedding_centroid = vector if self.embedding_centroid is None else .8*self.embedding_centroid + .2*vector

    def trim_after_snapshot(self):
        while len(self.recent_crops) > 2:
            self.recent_crops.popleft()


@dataclass
class ReIDEntry:
    speaker_id: int
    embedding_centroid: np.ndarray
    expires_frame: int
