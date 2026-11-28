"""Data adapters for loading PKL/CSV features."""
from .base import BaseAdapter, Utterance, Dialogue
from .iemocap_pkl import IEMOCAPPKLAdapter
from .meld_pkl import MELDPKLAdapter

def get_adapter(dataset: str) -> BaseAdapter:
    """Factory for getting the right adapter."""
    if dataset.lower() == "iemocap":
        return IEMOCAPPKLAdapter()
    elif dataset.lower() == "meld":
        return MELDPKLAdapter()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

__all__ = [
    "BaseAdapter", "Utterance", "Dialogue",
    "IEMOCAPPKLAdapter", "MELDPKLAdapter", "get_adapter"
]
