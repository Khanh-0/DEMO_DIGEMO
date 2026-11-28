"""Offline CSV metadata and Check File services.

This module only reads CSV/PKL metadata. It never imports or initializes
Whisper, Pyannote, YuNet, Light-ASD, or any online encoder.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from runtime.config import config


class OfflineDataError(ValueError):
    """Raised when an uploaded test video cannot be matched to test data."""


def _key(value: Any) -> str:
    """Normalize IDs without changing their displayed value."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _seconds(value: Any) -> float:
    """Parse CSV seconds or HH:MM:SS,mmm timestamps."""
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    hh, mm, ss = text.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss.replace(",", "."))


def infer_iemocap_dialogue_id(filename: str) -> str:
    """Extract Ses##[FM]_... dialogue ID from an unchanged filename."""
    stem = Path(filename).stem
    match = re.search(r"(Ses\d{2}[FM]_[^./\\]+)", stem, re.IGNORECASE)
    if not match:
        raise OfflineDataError(
            f"Không suy ra được IEMOCAP dialogue_id từ tên file: {filename}"
        )
    return match.group(1)


def _meld_rows(dialogue_id: str) -> list[dict[str, Any]]:
    labels = pd.read_csv(config.PROJECT_ROOT / "data" / "csv" / "test_sent_emo.csv")
    timestamps = pd.read_csv(config.MELD_TIMESTAMPS)
    did = _key(dialogue_id)
    labels = labels[labels["Dialogue_ID"].map(_key) == did]
    timestamps = timestamps[timestamps["dialogue_id"].map(_key) == did]
    ts_by_utt = {_key(r["utterance_id"]): r for _, r in timestamps.iterrows()}
    rows = []
    for _, row in labels.sort_values("Utterance_ID").iterrows():
        uid = _key(row["Utterance_ID"])
        ts = ts_by_utt.get(uid)
        if ts is None:
            raise OfflineDataError(f"MELD thiếu timestamp: dialogue={did}, utterance={uid}")
        rows.append({
            "dialogue_id": did,
            "utterance_id": uid,
            "speaker": str(row["Speaker"]),
            "start": float(ts["start"]),
            "end": float(ts["end"]),
            "text": str(row["Utterance"]),
            "emotion": str(row["Emotion"]).strip().lower(),
            "sentiment": str(row["Sentiment"]).strip().lower(),
            "original_video": str(ts["original_video"]),
        })
    if not rows:
        raise OfflineDataError(f"Không tìm thấy MELD Dialogue_ID={did} trong test_sent_emo.csv")
    return rows


def _iemocap_rows(dialogue_id: str) -> list[dict[str, Any]]:
    df = pd.read_csv(config.IEMOCAP_CSV)
    mask = (
        (df["dialogue_id"].astype(str) == dialogue_id)
        & (df["split"].astype(str).str.lower() == "test")
        & (df["is_6class"].astype(bool))
    )
    rows = []
    for _, row in df[mask].sort_values("start_time").iterrows():
        rows.append({
            "dialogue_id": dialogue_id,
            "utterance_id": str(row["utterance_id"]),
            "speaker": str(row["speaker"]),
            "start": float(row["start_time"]),
            "end": float(row["end_time"]),
            "text": str(row["text"]),
            "emotion": str(row["emotion"]).strip().lower(),
            "label_id": int(row["label_id"]),
            "video_path": str(row["video_path"]),
        })
    if not rows:
        all_dialogue = df[df["dialogue_id"].astype(str) == dialogue_id]
        if not all_dialogue.empty:
            split_values = ", ".join(sorted(all_dialogue["split"].astype(str).unique()))
            raise OfflineDataError(
                f"IEMOCAP dialogue {dialogue_id} tồn tại nhưng thuộc split={split_values}, "
                "không thuộc test. Tập test hiện dùng Session 5, ví dụ: Ses05F_impro01.avi"
            )
        raise OfflineDataError(
            f"Không tìm thấy IEMOCAP dialogue {dialogue_id} trong iemocap_6class_split.csv"
        )
    return rows


def check_file(dataset: str, filename: str, dialogue_id: str | None = None) -> dict[str, Any]:
    """Return modal-ready metadata for an uploaded offline test video."""
    dataset = dataset.upper().strip()
    if dataset == "MELD":
        if not dialogue_id:
            raise OfflineDataError("MELD cần Dialogue_ID để Check File")
        rows = _meld_rows(dialogue_id)
    elif dataset == "IEMOCAP":
        dialogue_id = dialogue_id or infer_iemocap_dialogue_id(filename)
        rows = _iemocap_rows(dialogue_id)
    else:
        raise OfflineDataError(f"Dataset không hỗ trợ: {dataset}")
    return {
        "dataset": dataset,
        "dialogue_id": rows[0]["dialogue_id"],
        "filename": filename,
        "utterance_count": len(rows),
        "timestamp_match": True,
        "rows": rows,
    }
