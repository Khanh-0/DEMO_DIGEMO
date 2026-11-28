"""Base predictor class for all models."""
from abc import ABC, abstractmethod
from typing import List
from adapters import Utterance
from runtime.result_schema import PredictionResult


class BasePredictor(ABC):
    """Abstract base for all predictors."""
    
    @abstractmethod
    def predict(self, utterances: List[Utterance]) -> List[PredictionResult]:
        """
        Predict emotions for a list of utterances.
        
        Args:
            utterances: List of Utterance objects from the same dialogue
        
        Returns:
            List of PredictionResult objects in same order
        """
        pass
