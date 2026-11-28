"""DiGemo predictor for IEMOCAP and MELD."""
import sys
import importlib.util
from typing import List, Optional
import torch
import numpy as np

from .base import BasePredictor
from adapters import Utterance
from runtime.result_schema import PredictionResult
from runtime.config import config


class DiGemoPredictor(BasePredictor):
    """DiGemo offline inference using PKL features."""
    
    # DiGemo IEMOCAP checkpoint order: hap, sad, neu, ang, exc, fru.
    IEMOCAP_EMOTIONS = {0: "hap", 1: "sad", 2: "neu", 3: "ang", 4: "exc", 5: "fru"}
    MELD_EMOTIONS = {0: "neutral", 1: "surprise", 2: "fear", 3: "sadness", 4: "joy", 5: "disgust", 6: "anger"}
    
    def __init__(self, dataset: str = "IEMOCAP"):
        self.dataset = dataset
        self.emotions = self.IEMOCAP_EMOTIONS if dataset == "IEMOCAP" else self.MELD_EMOTIONS
        self.device = torch.device(config.DEVICE)
        self._load_model()
    
    def _load_model(self):
        """Load DiGemo checkpoint."""
        # Load the isolated test bundle under unique module names. Importing
        # `model` directly is unsafe because Light-ASD also registers `model`.
        bundle = config.PROJECT_ROOT / "DiGemo_1"
        if str(bundle) not in sys.path:
            sys.path.insert(0, str(bundle))
        model_spec = importlib.util.spec_from_file_location("offline_digemo_model", bundle / "model.py")
        predict_spec = importlib.util.spec_from_file_location("offline_digemo_predict", bundle / "predict.py")
        if model_spec is None or model_spec.loader is None or predict_spec is None or predict_spec.loader is None:
            raise ImportError(f"Không load được DiGemo bundle: {bundle}")
        model_module = importlib.util.module_from_spec(model_spec)
        sys.modules["offline_digemo_model"] = model_module
        # predict.py imports `model`; bind it only while executing the isolated module.
        previous_model = sys.modules.get("model")
        sys.modules["model"] = model_module
        try:
            model_spec.loader.exec_module(model_module)
            predict_module = importlib.util.module_from_spec(predict_spec)
            sys.modules["offline_digemo_predict"] = predict_module
            predict_spec.loader.exec_module(predict_module)
        finally:
            if previous_model is None:
                sys.modules.pop("model", None)
            else:
                sys.modules["model"] = previous_model
        DiGemo = model_module.DiGemo
        _safe_torch_load = predict_module._safe_torch_load
        _strip_module_prefix = predict_module._strip_module_prefix
        
        if self.dataset == "IEMOCAP":
            ckpt_path = config.DIGEMO_IEMOCAP_CKPT
            embedding_dims = [1024, 342, 1582]  # text, visual, audio
            n_speakers = 2
            n_classes = 6
        else:  # MELD
            ckpt_path = config.DIGEMO_MELD_CKPT
            embedding_dims = [1024, 342, 300]
            n_speakers = 9
            n_classes = 7
        
        # Load checkpoint
        checkpoint = _safe_torch_load(str(ckpt_path), map_location=self.device)
        args = checkpoint["args"]
        args.no_cuda = (self.device.type == "cpu")
        
        # Build model
        self.model = DiGemo(args, embedding_dims, n_classes).to(self.device)
        state_dict = _strip_module_prefix(checkpoint["model_state_dict"])
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.n_speakers = n_speakers
        self.n_classes = n_classes
    
    def predict(self, utterances: List[Utterance]) -> List[PredictionResult]:
        """Predict emotions using DiGemo."""
        if not utterances:
            return []
        
        # Prepare batch
        seq_len = len(utterances)
        batch_size = 1
        
        # Stack features
        text_feat = torch.tensor(
            np.stack([utt.text_feature for utt in utterances]),
            dtype=torch.float32
        ).unsqueeze(1)  # (seq, 1, 1024)
        
        visual_feat = torch.tensor(
            np.stack([utt.visual_feature for utt in utterances]),
            dtype=torch.float32
        ).unsqueeze(1)  # (seq, 1, 342)
        
        audio_feat = torch.tensor(
            np.stack([utt.audio_feature for utt in utterances]),
            dtype=torch.float32
        ).unsqueeze(1)  # (seq, 1, audio_dim)
        
        # Masks
        umask = torch.ones(seq_len, batch_size, device=self.device)
        qmask = torch.zeros(seq_len, batch_size, self.n_speakers, device=self.device)
        
        # Set speaker one-hot (simplified: assign based on utterance order and speaker)
        for i, utt in enumerate(utterances):
            speaker_idx = 0 if utt.speaker == "F" else 1 if utt.speaker == "M" else int(utt.speaker) % self.n_speakers
            qmask[i, 0, speaker_idx] = 1.0
        
        dia_lengths = [seq_len]
        
        # Forward
        with torch.inference_mode():
            text_feat = text_feat.to(self.device)
            visual_feat = visual_feat.to(self.device)
            audio_feat = audio_feat.to(self.device)
            
            fused_logit, _, _, _, _ = self.model(
                text_feat, visual_feat, audio_feat, umask, qmask, dia_lengths
            )
            
            probs = torch.softmax(fused_logit, dim=-1).cpu().numpy()
        
        # Map predictions to utterances
        results = []
        for i, utt in enumerate(utterances):
            pred_probs = probs[i]
            emotion_id = int(np.argmax(pred_probs))
            confidence = float(pred_probs[emotion_id])
            emotion_name = self.emotions.get(emotion_id, "unknown")
            
            result = PredictionResult(
                model="DiGemo",
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
                probabilities=pred_probs.tolist()
            )
            results.append(result)
        
        return results
