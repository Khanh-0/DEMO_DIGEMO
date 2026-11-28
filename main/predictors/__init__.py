"""Model predictors for offline inference."""
from .base import BasePredictor
from .crg3_predictor import CRG3Predictor
from .digemo_predictor import DiGemoPredictor
from .dmcer_predictor import DMCERPredictor


def get_predictor(model: str, dataset: str = "IEMOCAP") -> BasePredictor:
    """Factory for getting the right predictor."""
    if model == "CRG-3":
        return CRG3Predictor(dataset)
    elif model == "DiGemo":
        return DiGemoPredictor(dataset)
    elif model == "DMCER":
        return DMCERPredictor(dataset)
    else:
        raise ValueError(f"Unknown model: {model}")


__all__ = [
    "BasePredictor", "CRG3Predictor", "DiGemoPredictor", "DMCERPredictor",
    "get_predictor"
]
