"""CRG-3 IEMOCAP checkpoint adapter."""
from pathlib import Path
import sys
import torch
import numpy as np

from .config import RuntimeConfig


class CRG3Runtime:
    def __init__(self, config=None):
        self.config = config or RuntimeConfig()
        self.config.validate()
        root = self.config.project_root
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from CRG3.config import model_config_from_dict
        from CRG3.model import build_model
        checkpoint = torch.load(self.config.crg3_checkpoint, map_location="cpu", weights_only=False)
        cfg = model_config_from_dict(checkpoint["model_config"])
        dims = checkpoint["feature_dims"]
        self.device = self.config.resolved_device()
        self.model = build_model(cfg, dims, checkpoint["num_classes"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.cfg = cfg
        self.history = []

    def predict(self, text, audio, visual, speaker=0):
        """Predict one current utterance using a minimal causal dialogue history."""
        import torch
        from CRG3.data_loader import Collator
        # Keep a lightweight history for later integration; current model call uses one utterance.
        self.history.append({"text": np.asarray(text), "audio": np.asarray(audio),
                             "visual": np.asarray(visual), "speaker": speaker})
        n = len(self.history)
        batch = {
            "text": torch.as_tensor(np.stack([x["text"] for x in self.history]), dtype=torch.float32)[None].to(self.device),
            "audio": torch.as_tensor(np.stack([x["audio"] for x in self.history]), dtype=torch.float32)[None].to(self.device),
            "visual": torch.as_tensor(np.stack([x["visual"] for x in self.history]), dtype=torch.float32)[None].to(self.device),
            "mask": torch.ones(1, n, dtype=torch.bool, device=self.device),
            "speaker": torch.as_tensor([[x["speaker"] for x in self.history]], dtype=torch.long, device=self.device),
        }
        # CRG-3 supports speaker adjacency when speakers are supplied; empty neighbor tables
        # are accepted only by legacy paths, so materialize causal neighbors for this history.
        diff, same = [], []
        ids = [x["speaker"] for x in self.history]
        for i in range(n):
            past = list(range(i))
            d = [j for j in past if ids[j] != ids[i]]
            s = [j for j in past if ids[j] == ids[i]]
            diff.append(d); same.append(s)
        def pack(rows):
            width=max(1,max((len(x) for x in rows),default=0)); idx=torch.full((1,n,width),-1,dtype=torch.long,device=self.device)
            for i,row in enumerate(rows):
                if row: idx[0,i,:len(row)]=torch.tensor(row,device=self.device)
            return {"idx":idx,"mask":idx.ge(0)}
        batch["diff"] = pack(diff); batch["same"] = pack(same)
        with torch.inference_mode():
            logits = self.model(batch)
        probs = logits[0, -1].float().softmax(-1).cpu().numpy()
        label_id = int(probs.argmax())
        return {"emotion_id": label_id, "emotion": self.config.labels[label_id],
                "confidence": float(probs[label_id]), "probabilities": probs.tolist()}
