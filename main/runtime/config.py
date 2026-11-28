
from pathlib import Path
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    """All paths and settings managed in one place."""
    
    # Project root
    PROJECT_ROOT: Path = Path(r"E:\Do_an\DEMO\demo_meld")
    
    # === PATHS ===
    # PKL features
    IEMOCAP_PKL: Path = PROJECT_ROOT / "data" / "feats" / "iemocap_multi_features.pkl"
    MELD_PKL: Path = PROJECT_ROOT / "data" / "feats" / "meld_multi_features.pkl"
    
    # Timestamps
    IEMOCAP_CSV: Path = PROJECT_ROOT / "checkpoints" / "iemocap" / "splits" / "iemocap_6class_split.csv"
    MELD_TIMESTAMPS: Path = PROJECT_ROOT / "data" / "utterance_timestamps.csv"
    
    # Model checkpoints
    CRG3_IEMOCAP_CKPT: Path = PROJECT_ROOT / "CRG3" / "checkpoints" / "iemocap.pt"
    CRG3_MELD_CKPT: Path = PROJECT_ROOT / "CRG3" / "checkpoints" / "meld.pt"
    
    DIGEMO_IEMOCAP_CKPT: Path = PROJECT_ROOT / "checkpoints" / "digemo_iemocap.pth"
    DIGEMO_MELD_CKPT: Path = PROJECT_ROOT / "checkpoints" / "meld" / "digemo_meld.pth"
    
    DMCER_IEMOCAP_CKPT: Path = PROJECT_ROOT / "Dynamic-Multimodal-Causal-Graph-Framework-ERC" / "code" / "IEMOCAP" / "models" / "final_best.pt"
    DMCER_MELD_CKPT: Path = PROJECT_ROOT / "Dynamic-Multimodal-Causal-Graph-Framework-ERC" / "code" / "MELD" / "models" / "final_best.pt"
    
    # DMCER test data
    DMCER_IEMOCAP_DATA: Path = PROJECT_ROOT / "Dynamic-Multimodal-Causal-Graph-Framework-ERC" / "code" / "IEMOCAP" / "data_test"
    DMCER_MELD_DATA: Path = PROJECT_ROOT / "Dynamic-Multimodal-Causal-Graph-Framework-ERC" / "code" / "MELD" / "data_test"
    
    # === FEATURE DIMENSIONS ===
    TEXT_DIM: int = 1024
    AUDIO_DIM_IEMOCAP: int = 1582
    AUDIO_DIM_MELD: int = 300
    VISUAL_DIM: int = 342
    
    # === EMOTION LABELS ===
    IEMOCAP_EMOTIONS: tuple = ("hap", "sad", "neu", "ang", "exc", "fru")  # PKL/checkpoint order, 6 classes
    MELD_EMOTIONS: tuple = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")  # 7 classes
    
    # === DEVICE ===
    DEVICE: str = os.environ.get("EMOTION_DEVICE", "cpu")
    
    # === OTHER ===
    DEBUG: bool = os.environ.get("DEMO_DEBUG", "0").lower() in ("1", "true")
    
    def validate(self) -> None:
        """Check all required paths exist."""
        required = {
            "IEMOCAP_PKL": self.IEMOCAP_PKL,
            "MELD_PKL": self.MELD_PKL,
            "IEMOCAP_CSV": self.IEMOCAP_CSV,
            "MELD_TIMESTAMPS": self.MELD_TIMESTAMPS,
            "CRG3_IEMOCAP_CKPT": self.CRG3_IEMOCAP_CKPT,
            "CRG3_MELD_CKPT": self.CRG3_MELD_CKPT,
            "DIGEMO_IEMOCAP_CKPT": self.DIGEMO_IEMOCAP_CKPT,
            "DIGEMO_MELD_CKPT": self.DIGEMO_MELD_CKPT,
            "DMCER_IEMOCAP_CKPT": self.DMCER_IEMOCAP_CKPT,
            "DMCER_MELD_CKPT": self.DMCER_MELD_CKPT,
            "DMCER_IEMOCAP_DATA": self.DMCER_IEMOCAP_DATA,
            "DMCER_MELD_DATA": self.DMCER_MELD_DATA,
        }
        missing = {k: v for k, v in required.items() if not v.exists()}
        if missing:
            msg = "Missing required assets:\n"
            for k, v in missing.items():
                msg += f"  {k}: {v}\n"
            raise FileNotFoundError(msg)


# Default singleton
config = Config()
