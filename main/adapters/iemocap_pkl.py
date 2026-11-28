"""Adapter for IEMOCAP PKL-based features with CSV timestamps."""
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .base import Utterance, Dialogue, BaseAdapter
from runtime.config import config


class IEMOCAPPKLAdapter(BaseAdapter):
    """Load IEMOCAP from PKL + CSV."""
    
    # PKL/CSV label_id order from iemocap_6class_split.csv:
    # 0 hap, 1 sad, 2 neu, 3 ang, 4 exc, 5 fru.
    EMOTIONS = {
        0: "hap", 1: "sad", 2: "neu", 3: "ang", 4: "exc", 5: "fru"
    }
    
    def __init__(self):
        self.pkl_data = None
        self.csv_data = None
        self._load_pkl()
        self._load_csv()
    
    def _load_pkl(self):
        """Load IEMOCAP PKL."""
        with open(config.IEMOCAP_PKL, "rb") as f:
            self.pkl_data = pickle.load(f, encoding="latin1")
        # PKL structure (tuple 12):
        # [0-2,9]: dicts dialogue_id -> [utterance_ids, speakers, labels, text_strings]
        # [3-6]: text features (4 variants, 1024 dims)
        # [7]: audio (1582 dims)
        # [8]: visual (342 dims)
        # [10]: train dialogue list
        # [11]: test dialogue list
    
    def _load_csv(self):
        """Load IEMOCAP CSV for timestamps and metadata."""
        df = pd.read_csv(config.IEMOCAP_CSV)
        # Filter for test split and 6-class emotions only
        df = df[df["split"] == "test"]
        df = df[df["is_6class"] == True]
        self.csv_data = df
    
    def get_test_dialogue_ids(self) -> List[str]:
        """Return test dialogue IDs."""
        return self.pkl_data[11]
    
    def get_emotions(self) -> Dict[int, str]:
        """Return emotion mapping."""
        return self.EMOTIONS.copy()
    
    def load_dialogues(self, dialogue_ids: List[str] = None) -> List[Dialogue]:
        """Load dialogues from PKL + CSV."""
        if dialogue_ids is None:
            dialogue_ids = self.get_test_dialogue_ids()
        
        dialogues = []
        for dia_id in dialogue_ids:
            if dia_id not in self.pkl_data[0]:
                continue
            
            dia = self._load_dialogue(dia_id)
            if dia and dia.utterances:
                dialogues.append(dia)
        
        return dialogues
    
    def _load_dialogue(self, dialogue_id: str) -> Optional[Dialogue]:
        """Load one dialogue."""
        pkl = self.pkl_data
        
        # Get utterance metadata from PKL
        utt_ids = pkl[0].get(dialogue_id, [])
        speakers = pkl[1].get(dialogue_id, [])
        labels = pkl[2].get(dialogue_id, [])
        text_strings = pkl[9].get(dialogue_id, [])
        
        # Features from PKL
        text_feats = pkl[3].get(dialogue_id, [])  # or [4], [5], [6] - use first variant
        audio_feats = pkl[7].get(dialogue_id, [])
        visual_feats = pkl[8].get(dialogue_id, [])
        
        if not utt_ids:
            return None
        
        utterances = []
        for i, (utt_id, speaker, label, text) in enumerate(zip(
            utt_ids, speakers, labels, text_strings
        )):
            # Match with CSV for timestamps
            csv_match = self.csv_data[
                (self.csv_data["dialogue_id"] == dialogue_id) &
                (self.csv_data["utterance_id"] == utt_id)
            ]
            
            if csv_match.empty:
                # Fallback: use dummy times
                start_time = i * 1.0
                end_time = (i + 1) * 1.0
            else:
                row = csv_match.iloc[0]
                start_time = float(row["start_time"])
                end_time = float(row["end_time"])
            
            # Get features (safely)
            try:
                text_feat = text_feats[i].tolist() if isinstance(text_feats, np.ndarray) else text_feats[i]
            except (IndexError, TypeError):
                text_feat = [0.0] * 1024
            
            try:
                audio_feat = audio_feats[i].tolist() if isinstance(audio_feats, np.ndarray) else audio_feats[i]
            except (IndexError, TypeError):
                audio_feat = [0.0] * 1582
            
            try:
                visual_feat = visual_feats[i].tolist() if isinstance(visual_feats, np.ndarray) else visual_feats[i]
            except (IndexError, TypeError):
                visual_feat = [0.0] * 342
            
            utt = Utterance(
                dialogue_id=dialogue_id,
                utterance_id=utt_id,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=text,
                text_feature=text_feat,
                audio_feature=audio_feat,
                visual_feature=visual_feat,
                emotion_label=int(label),
                emotion_name=self.EMOTIONS.get(int(label), "unknown")
            )
            utterances.append(utt)
        
        return Dialogue(dialogue_id=dialogue_id, utterances=utterances)
