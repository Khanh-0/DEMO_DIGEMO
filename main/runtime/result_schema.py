"""Standardized output schema for all predictors."""
from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class PredictionResult:
    """Unified prediction result format."""
    model: str  # "CRG-3", "DiGemo", "DMCER"
    dataset: str  # "IEMOCAP", "MELD"
    dialogue_id: str
    utterance_id: str
    speaker: str  # "F", "M", or speaker name
    start_time: float  # seconds
    end_time: float  # seconds
    text: str
    emotion: str  # emotion label
    emotion_id: int  # emotion class index
    confidence: float  # 0.0-1.0
    probabilities: List[float]  # logits/probabilities for all classes
    
    def to_dict(self):
        return asdict(self)
    
    def to_csv_row(self):
        """Format as CSV-compatible row."""
        return {
            "model": self.model,
            "dataset": self.dataset,
            "dialogue_id": self.dialogue_id,
            "utterance_id": self.utterance_id,
            "speaker": self.speaker,
            "start_time": f"{self.start_time:.4f}",
            "end_time": f"{self.end_time:.4f}",
            "text": self.text,
            "emotion": self.emotion,
            "confidence": f"{self.confidence:.4f}",
        }
