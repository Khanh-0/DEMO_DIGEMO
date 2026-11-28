"""Offline uploaded-video test mode: PKL/CSV/checkpoint inference only."""
MODE = "offline"
MODELS = ("CRG-3", "DiGemo", "DMCER")
from .routes import bp

__all__ = ["MODE", "MODELS", "bp"]
