"""Offline model runner facade.

It imports only PKL/CSV adapters and offline predictors. Online runtime is not
loaded from this module.
"""
from __future__ import annotations

from adapters import get_adapter
from predictors import get_predictor
from runtime.registry import ModelRegistry


def predict_dialogue(dataset: str, model: str, dialogue_id: str) -> list[dict]:
    dataset = dataset.upper().strip()
    if not ModelRegistry.is_valid(dataset, model):
        raise ValueError(f"Invalid offline combination: {dataset} + {model}")
    adapter = get_adapter(dataset)
    dialogues = adapter.load_dialogues([str(dialogue_id)])
    if not dialogues:
        raise ValueError(f"Không tìm thấy dialogue {dialogue_id} trong {dataset} PKL/test data")
    predictor = get_predictor(model, dataset)
    if model == "DMCER":
        return [result.to_dict() for result in predictor.predict_test_dialogue(str(dialogue_id))]
    return [result.to_dict() for result in predictor.predict(dialogues[0].utterances)]
