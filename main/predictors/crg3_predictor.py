"""CRG-3 predictor for IEMOCAP."""
import sys
from typing import List
import torch
import numpy as np

from .base import BasePredictor
from adapters import Utterance
from runtime.result_schema import PredictionResult
from runtime.config import config


class CRG3Predictor(BasePredictor):
    """CRG-3 offline inference using PKL features."""
    
    # CRG-3 is trained directly on the IEMOCAP PKL label order.
    EMOTIONS = {
        0: "hap", 1: "sad", 2: "neu", 3: "ang", 4: "exc", 5: "fru"
    }
    
    def __init__(self, dataset: str = "IEMOCAP"):
        self.dataset = dataset.upper()
        if self.dataset not in ("IEMOCAP", "MELD"):
            raise ValueError(f"Unsupported CRG-3 dataset: {dataset}")
        self.device = torch.device(config.DEVICE)
        self.emotions = {0: "hap", 1: "sad", 2: "neu", 3: "ang", 4: "exc", 5: "fru"} if self.dataset == "IEMOCAP" else {0: "neutral", 1: "surprise", 2: "fear", 3: "sadness", 4: "joy", 5: "disgust", 6: "anger"}
        self._load_model()
    
    def _load_model(self):
        """Load CRG-3 checkpoint."""
        # Add CRG3 to path
        if str(config.PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(config.PROJECT_ROOT))
        
        from CRG3.config import model_config_from_dict
        from CRG3.model import build_model
        
        # Load checkpoint
        ckpt_path = config.CRG3_MELD_CKPT if self.dataset == "MELD" else config.CRG3_IEMOCAP_CKPT
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if str(checkpoint.get("dataset", "")).upper() != self.dataset:
            raise ValueError(f"CRG-3 checkpoint dataset mismatch: expected {self.dataset}, got {checkpoint.get('dataset')}")
        
        # Rebuild model
        cfg = model_config_from_dict(checkpoint["model_config"])
        dims = checkpoint["feature_dims"]
        num_classes = checkpoint.get("num_classes", 6)
        
        self.model = build_model(cfg, dims, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.cfg = cfg
    
    def predict(self, utterances: List[Utterance]) -> List[PredictionResult]:
        """Predict emotions using causal history."""
        if not utterances:
            return []
        
        # CRG-3 uses causal history (past utterances only)
        results = []
        history = []  # Accumulate for causal modeling
        
        for utt in utterances:
            # Add current utterance to history
            history.append({
                "text": np.asarray(utt.text_feature, dtype=np.float32),
                "audio": np.asarray(utt.audio_feature, dtype=np.float32),
                "visual": np.asarray(utt.visual_feature, dtype=np.float32),
                "speaker": 0 if utt.speaker == "F" else 1,  # CRG-3 uses 0/1 for speakers
            })
            
            # Predict with current history (causal)
            emotion_id, confidence, probs = self._forward(history)
            emotion_name = self.emotions.get(emotion_id, "unknown")
            
            result = PredictionResult(
                model="CRG-3",
                dataset=self.dataset,
                dialogue_id=utt.dialogue_id,
                utterance_id=utt.utterance_id,
                speaker=utt.speaker,
                start_time=utt.start_time,
                end_time=utt.end_time,
                text=utt.text,
                emotion=emotion_name,
                emotion_id=emotion_id,
                confidence=confidence,
                probabilities=probs
            )
            results.append(result)
        
        return results
    
    @torch.inference_mode()
    def _forward(self, history):
        """Forward pass with causal history."""
        from CRG3.data_loader import Collator
        
        n = len(history)
        batch = {
            "text": torch.as_tensor(np.stack([x["text"] for x in history]), dtype=torch.float32)[None].to(self.device),
            "audio": torch.as_tensor(np.stack([x["audio"] for x in history]), dtype=torch.float32)[None].to(self.device),
            "visual": torch.as_tensor(np.stack([x["visual"] for x in history]), dtype=torch.float32)[None].to(self.device),
            "mask": torch.ones(1, n, dtype=torch.bool, device=self.device),
            "speaker": torch.as_tensor([[x["speaker"] for x in history]], dtype=torch.long, device=self.device),
        }
        
        # Build causal neighbor tables
        ids = [x["speaker"] for x in history]
        diff, same = [], []
        for i in range(n):
            past = list(range(i))
            d = [j for j in past if ids[j] != ids[i]]
            s = [j for j in past if ids[j] == ids[i]]
            diff.append(d)
            same.append(s)
        
        def pack(rows):
            width = max(1, max((len(x) for x in rows), default=0))
            idx = torch.full((1, n, width), -1, dtype=torch.long, device=self.device)
            for i, row in enumerate(rows):
                if row:
                    idx[0, i, :len(row)] = torch.tensor(row, device=self.device)
            return {"idx": idx, "mask": idx.ge(0)}
        
        batch["diff"] = pack(diff)
        batch["same"] = pack(same)
        
        # Forward
        logits = self.model(batch)
        probs = logits[0, -1].float().softmax(-1).cpu().numpy()
        emotion_id = int(probs.argmax())
        confidence = float(probs[emotion_id])
        
        return emotion_id, confidence, probs.tolist()
