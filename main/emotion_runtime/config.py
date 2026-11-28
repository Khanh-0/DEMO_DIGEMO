"""Configuration for the reusable IEMOCAP realtime runtime."""
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True)
class RuntimeConfig:
    """Paths and timing settings; model files stay outside folder ``a``."""

    project_root: Path = field(default_factory=lambda: Path(
        os.environ.get("DEMO_MELD_ROOT", r"E:\Do_an\DEMO\demo_meld")
    ))
    dataset: str = os.environ.get("EMOTION_DATASET", "iemocap").lower()
    device: str = os.environ.get("EMOTION_DEVICE", "auto")
    fps: int = 25
    sample_rate: int = 16000
    asd_hop_frames: int = 12
    emotion_interval_seconds: float = 1.0
    max_track_seconds: float = 8.0
    whisper_model: str = os.environ.get("WHISPER_MODEL", "large-v3")

    labels: tuple[str, ...] = ("hap", "sad", "neu", "ang", "exc", "fru")

    @property
    def is_iemocap(self) -> bool:
        return self.dataset == "iemocap"

    @property
    def is_meld(self) -> bool:
        return self.dataset == "meld"

    @property
    def crg3_checkpoint(self) -> Path:
        if self.is_meld:
            return self.project_root / "checkpoints" / "meld" / "digemo_meld.pth"
        return self.project_root / "CRG3" / "outputs" / "iemocap" / "best_test.pt"

    @property
    def text_dim(self) -> int:
        return 1024

    @property
    def audio_dim(self) -> int:
        return 300 if self.is_meld else 1582

    @property
    def visual_dim(self) -> int:
        return 342

    @property
    def emotion_labels(self) -> tuple[str, ...]:
        return ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger") if self.is_meld else self.labels

    @property
    def encoder_root(self) -> Path:
        """Self-contained encoder package and assets inside folder ``a``."""
        return Path(__file__).resolve().parents[1] / "encoders"

    @property
    def encoder_dir(self) -> Path:
        return self.encoder_root / "code"

    @property
    def text_checkpoint(self) -> Path:
        if self.is_meld:
            return self.encoder_root / "checkpoints" / "meld" / "text_ft.pt"
        return self.encoder_root / "checkpoints" / "iemocap" / "text_ft.pt"

    @property
    def feature_dir(self) -> Path:
        return self.project_root / "checkpoints" / ("meld" if self.is_meld else "iemocap")

    @property
    def iemocap_norm(self) -> Path:
        return self.encoder_root / "checkpoints" / "iemocap" / "norm.npz"

    @property
    def face_checkpoint(self) -> Path:
        return self.encoder_root / "checkpoints" / "shared" / "face_fer.pt"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        if self.dataset not in ("iemocap", "meld"):
            raise ValueError("dataset must be 'iemocap' or 'meld'")
        if self.is_iemocap:
            paths = {
                "CRG-3 checkpoint": self.crg3_checkpoint,
                "Text checkpoint": self.text_checkpoint,
                "IEMOCAP norm": self.iemocap_norm,
                "Face checkpoint": self.face_checkpoint,
                "encoder directory": self.encoder_dir,
            }
        else:
            paths = {
                "MELD encoder directory": self.encoder_dir,
                "MELD text checkpoint": self.encoder_root / "checkpoints" / "meld" / "text_ft.pt",
                "MELD audio head": self.encoder_root / "checkpoints" / "meld" / "audio_head.npz",
                "MELD norm": self.encoder_root / "checkpoints" / "meld" / "norm.npz",
                "Face checkpoint": self.face_checkpoint,
            }
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))
        if self.fps <= 0 or self.sample_rate <= 0 or self.asd_hop_frames <= 0:
            raise ValueError("fps, sample_rate and asd_hop_frames must be positive")
