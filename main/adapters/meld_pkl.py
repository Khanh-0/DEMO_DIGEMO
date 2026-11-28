"""Adapter for MELD PKL-based features with timestamps."""
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .base import Utterance, Dialogue, BaseAdapter
from runtime.config import config


class MELDPKLAdapter(BaseAdapter):
    """Load MELD from PKL + timestamps CSV."""
    
    EMOTIONS = {
        0: "neutral", 1: "surprise", 2: "fear", 3: "sadness",
        4: "joy", 5: "disgust", 6: "anger"
    }
    
    def __init__(self):
        self.pkl_data = None
        self.timestamps_df = None
        self._load_pkl()
        self._load_timestamps()
    
    def _load_pkl(self):
        """Load MELD PKL."""
        with open(config.MELD_PKL, "rb") as f:
            self.pkl_data = pickle.load(f, encoding="latin1")
        # PKL structure (list 14):
        # [0-2]: lists/dicts with utterance metadata per dialogue
        # [1]: one-hot speaker encoding (9 speakers)
        # [2]: emotion labels
        # [4-7]: text features (1024 dims)
        # [8]: audio features (300 dims)
        # [9]: visual features (342 dims)
        # [10]: text strings
        # [11]: train dialogue indices
        # [12]: test dialogue indices
    
    def _load_timestamps(self):
        """Load MELD timestamps CSV."""
        self.timestamps_df = pd.read_csv(config.MELD_TIMESTAMPS)
    
    TEST_PKL_OFFSET = 1153

    def get_test_dialogue_ids(self) -> List[str]:
        """Return original MELD test Dialogue_ID values used by the CSV files."""
        test_indices = sorted(self.pkl_data[12])
        return [str(int(idx) - self.TEST_PKL_OFFSET) for idx in test_indices]

    def _pkl_dialogue_index(self, dialogue_id: str) -> int:
        """Map original CSV Dialogue_ID (0..279) to test PKL index (1153..1432)."""
        csv_id = int(dialogue_id)
        direct = csv_id
        shifted = csv_id + self.TEST_PKL_OFFSET
        if shifted in self.pkl_data[0]:
            return shifted
        if direct in self.pkl_data[0] and direct in self.pkl_data[12]:
            return direct
        raise KeyError(f"MELD dialogue {dialogue_id} không có trong test PKL")
    
    def get_emotions(self) -> Dict[int, str]:
        """Return emotion mapping."""
        return self.EMOTIONS.copy()
    
    def load_dialogues(self, dialogue_ids: List[str] = None) -> List[Dialogue]:
        """Load dialogues from PKL + timestamps."""
        if dialogue_ids is None:
            dialogue_ids = self.get_test_dialogue_ids()
        
        dialogues = []
        for dia_id in dialogue_ids:
            dia = self._load_dialogue(dia_id)
            if dia and dia.utterances:
                dialogues.append(dia)
        
        return dialogues
    
    def _load_dialogue(self, dialogue_id: str) -> Optional[Dialogue]:
        """Load one dialogue."""
        pkl = self.pkl_data
        
        # Convert dialogue_id to index
        try:
            csv_dialogue_id = str(int(dialogue_id))
            dia_idx = self._pkl_dialogue_index(csv_dialogue_id)
        except (ValueError, TypeError, KeyError):
            return None
        
        # Get utterance metadata from the shifted test PKL index.
        try:
            utt_ids = pkl[0][dia_idx]  # list of utterance IDs
            speakers_onehot = pkl[1][dia_idx]  # (num_utt, 9)
            labels = pkl[2][dia_idx]  # list of emotion labels
            text_strings = pkl[10][dia_idx]  # list of text strings
        except (IndexError, KeyError, TypeError):
            return None
        
        # Features
        try:
            text_feats = pkl[4][dia_idx]  # (num_utt, 1024)
            audio_feats = pkl[8][dia_idx]  # (num_utt, 300)
            visual_feats = pkl[9][dia_idx]  # (num_utt, 342)
        except (IndexError, TypeError):
            return None
        
        if not utt_ids:
            return None
        
        utterances = []
        for i, (utt_id, speaker_oh, label, text) in enumerate(zip(
            utt_ids, speakers_onehot, labels, text_strings
        )):
            # Decode speaker from one-hot
            speaker_idx = int(np.argmax(speaker_oh)) if isinstance(speaker_oh, (list, np.ndarray)) else 0
            speaker = str(speaker_idx)
            
            # Match with timestamps CSV
            ts_match = self.timestamps_df[
                (self.timestamps_df["dialogue_id"] == dia_idx) &
                (self.timestamps_df["utterance_id"] == utt_id)
            ]
            
            if ts_match.empty:
                # Fallback: use dummy times
                start_time = i * 1.0
                end_time = (i + 1) * 1.0
            else:
                row = ts_match.iloc[0]
                start_time = float(row["start"])
                end_time = float(row["end"])
            
            # Get features (safely)
            try:
                text_feat = text_feats[i].tolist() if isinstance(text_feats, np.ndarray) else text_feats[i]
            except (IndexError, TypeError):
                text_feat = [0.0] * 1024
            
            try:
                audio_feat = audio_feats[i].tolist() if isinstance(audio_feats, np.ndarray) else audio_feats[i]
            except (IndexError, TypeError):
                audio_feat = [0.0] * 300
            
            try:
                visual_feat = visual_feats[i].tolist() if isinstance(visual_feats, np.ndarray) else visual_feats[i]
            except (IndexError, TypeError):
                visual_feat = [0.0] * 342
            
            utt = Utterance(
                dialogue_id=csv_dialogue_id,
                utterance_id=str(utt_id),
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
        
        return Dialogue(dialogue_id=csv_dialogue_id, utterances=utterances)
