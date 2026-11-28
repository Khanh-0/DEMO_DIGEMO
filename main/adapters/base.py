"""Base adapter for loading dialogue data."""
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Utterance:
    """Represents one utterance."""
    dialogue_id: str
    utterance_id: str
    speaker: str
    start_time: float
    end_time: float
    text: str
    text_feature: list  # (1024,)
    audio_feature: list  # (audio_dim,)
    visual_feature: list  # (342,)
    emotion_label: int  # 0-5 or 0-6
    emotion_name: str


@dataclass
class Dialogue:
    """Represents one dialogue with its utterances."""
    dialogue_id: str
    utterances: List[Utterance]


class BaseAdapter(ABC):
    """Abstract base for data adapters."""
    
    @abstractmethod
    def load_dialogues(self, dialogue_ids: List[str] = None) -> List[Dialogue]:
        """Load dialogues. If dialogue_ids is None, load all test dialogues."""
        pass
    
    @abstractmethod
    def get_test_dialogue_ids(self) -> List[str]:
        """Return list of all test dialogue IDs."""
        pass
    
    @abstractmethod
    def get_emotions(self) -> Dict[int, str]:
        """Return emotion ID -> name mapping."""
        pass
