"""DMCER predictor for IEMOCAP and MELD."""
import sys
import json
import importlib.util
from pathlib import Path
from typing import List
import torch
import numpy as np

from .base import BasePredictor
from adapters import Utterance
from runtime.result_schema import PredictionResult
from runtime.config import config


class DMCERPredictor(BasePredictor):
    """DMCER offline inference."""
    
    IEMOCAP_EMOTIONS = {0: "neu", 1: "fru", 2: "ang", 3: "sad", 4: "hap", 5: "exc"}
    MELD_EMOTIONS = {0: "neutral", 1: "surprise", 2: "fear", 3: "sadness", 4: "joy", 5: "disgust", 6: "anger"}
    
    def __init__(self, dataset: str = "IEMOCAP"):
        self.dataset = dataset
        self.emotions = self.IEMOCAP_EMOTIONS if dataset == "IEMOCAP" else self.MELD_EMOTIONS
        self.device = torch.device(config.DEVICE)
        self._load_model()
    
    def _load_model(self):
        """Load DMCER model."""
        if self.dataset == "IEMOCAP":
            dmcer_root = config.DMCER_IEMOCAP_DATA.parent
            ckpt_path = config.DMCER_IEMOCAP_CKPT
            data_root = config.DMCER_IEMOCAP_DATA
        else:  # MELD
            dmcer_root = config.DMCER_MELD_DATA.parent
            ckpt_path = config.DMCER_MELD_CKPT
            data_root = config.DMCER_MELD_DATA
        
        # Add DMCER to path
        if str(dmcer_root) not in sys.path:
            sys.path.insert(0, str(dmcer_root))
        
        model_file = dmcer_root / "model.py"
        module_name = f"dmcer_{self.dataset.lower()}_model"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Không load được DMCER model source: {model_file}")
        dmcer_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = dmcer_module
        spec.loader.exec_module(dmcer_module)
        DM_CER, DMConfig = dmcer_module.DM_CER, dmcer_module.DMConfig
        
        # Load config override
        config_override_path = data_root / "dmcer_config_override.json"
        cfg = DMConfig()
        if config_override_path.exists():
            with open(config_override_path, "r") as f:
                override = json.load(f)
                for k, v in override.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        
        # Build and load model
        self.model = DM_CER(cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.cfg = cfg
    
    def predict_test_dialogue(self, dialogue_id: str) -> List[PredictionResult]:
        """Predict using DMCER's own preprocessed data_test JSONL."""
        path = config.DMCER_IEMOCAP_DATA / "iemocap_test.jsonl" if self.dataset == "IEMOCAP" else config.DMCER_MELD_DATA / "test.jsonl"
        samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        sample = next((x for x in samples if str(x.get("dialog_id")) == str(dialogue_id)), None)
        if sample is None:
            raise ValueError(f"DMCER không có dialogue {dialogue_id} trong {path}")
        turns = sample["turns"]
        text = torch.tensor([t["text_emb"] for t in turns], dtype=torch.float32)[None].to(self.device)
        audio = torch.tensor([t["audio"] for t in turns], dtype=torch.float32)[None].to(self.device)
        vision = torch.tensor([t["vision"] for t in turns], dtype=torch.float32)[None].to(self.device)
        speakers = torch.tensor([[int(t["speaker_idx"]) for t in turns]], dtype=torch.long).to(self.device)
        length = torch.tensor([len(turns)], dtype=torch.long).to(self.device)
        labels = torch.full((1, len(turns)), -100, dtype=torch.long).to(self.device)
        batch = {"text": text, "audio": audio, "vision": vision, "speakers": speakers, "length": length, "labels": labels}
        with torch.inference_mode():
            logits = self.model(batch, missing_modality_p=0.0)["logits"][0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results=[]
        for i,t in enumerate(turns):
            eid=int(np.argmax(probs[i])); result=PredictionResult(model="DMCER",dataset=self.dataset,dialogue_id=str(dialogue_id),utterance_id=str(t.get("utterance_id",i)),speaker=str(t.get("speaker_idx",0)),start_time=float(t.get("start",0)),end_time=float(t.get("end",0)),text=str(t.get("text", "")),emotion=self.emotions.get(eid,"unknown"),emotion_id=eid,confidence=float(probs[i,eid]),probabilities=probs[i].tolist()); results.append(result)
        return results

    def predict(self, utterances: List[Utterance]) -> List[PredictionResult]:
        """Predict using DMCER-compatible features; reject incompatible PKL data."""
        if not utterances:
            return []
        
        seq_len = len(utterances)
        
        # DMCER checkpoints use their own preprocessed test features. Do not
        # silently feed DiGemo/CRG-3 PKL dimensions into a DMCER checkpoint.
        expected = (int(getattr(self.cfg, "d_text", 0)), int(getattr(self.cfg, "d_audio", 0)), int(getattr(self.cfg, "d_vision", 0)))
        actual = (len(utterances[0].text_feature), len(utterances[0].audio_feature), len(utterances[0].visual_feature))
        if actual != expected:
            raise ValueError(
                f"DMCER {self.dataset} cần feature dimensions text/audio/vision={expected}, "
                f"nhưng PKL adapter cung cấp {actual}. Dùng DMCER data_test/inference adapter riêng; "
                "không dùng feature PKL của DiGemo/CRG-3."
            )

        text = torch.tensor(
            np.stack([utt.text_feature for utt in utterances]),
            dtype=torch.float32
        )[None]  # (1, seq_len, text_dim)
        
        audio = torch.tensor(
            np.stack([utt.audio_feature for utt in utterances]),
            dtype=torch.float32
        )[None]  # (1, seq_len, audio_dim)
        
        vision = torch.tensor(
            np.stack([utt.visual_feature for utt in utterances]),
            dtype=torch.float32
        )[None]  # (1, seq_len, vision_dim)
        
        # Speaker indices
        speakers = torch.tensor(
            [[0 if utt.speaker in ("F", "0") else 1 for utt in utterances]],
            dtype=torch.long
        )  # (1, seq_len)
        
        # Length
        length = torch.tensor([seq_len], dtype=torch.long)
        
        # Dummy labels for inference
        labels = torch.full((1, seq_len), -100, dtype=torch.long)
        
        batch = {
            "text": text.to(self.device),
            "audio": audio.to(self.device),
            "vision": vision.to(self.device),
            "speakers": speakers.to(self.device),
            "length": length,
            "labels": labels.to(self.device),
        }
        
        # Forward
        with torch.inference_mode():
            output = self.model(batch, missing_modality_p=0.0)
            logits = output["logits"]  # (1, seq_len, num_classes)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (1, seq_len, num_classes)
        
        # Map predictions
        results = []
        for i, utt in enumerate(utterances):
            pred_probs = probs[0, i]
            emotion_id = int(np.argmax(pred_probs))
            confidence = float(pred_probs[emotion_id])
            emotion_name = self.emotions.get(emotion_id, "unknown")
            
            result = PredictionResult(
                model="DMCER",
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
