"""Registry of valid model/dataset combinations."""
from typing import List, Tuple, Dict, Optional


class ModelRegistry:
    """Manage valid model/dataset combinations."""
    
    # Format: (dataset, model) -> data_source
    VALID_COMBINATIONS = {
        ("IEMOCAP", "CRG-3"): "IEMOCAP PKL",
        ("IEMOCAP", "DiGemo"): "IEMOCAP PKL",
        ("IEMOCAP", "DMCER"): "DMCER IEMOCAP test data",
        ("MELD", "CRG-3"): "MELD PKL",
        ("MELD", "DiGemo"): "MELD PKL",
        ("MELD", "DMCER"): "DMCER MELD test data",
    }
    
    @classmethod
    def is_valid(cls, dataset: str, model: str) -> bool:
        """Check if dataset+model combination is valid."""
        return (dataset, model) in cls.VALID_COMBINATIONS
    
    @classmethod
    def get_data_source(cls, dataset: str, model: str) -> Optional[str]:
        """Get data source for a valid combination."""
        return cls.VALID_COMBINATIONS.get((dataset, model))
    
    @classmethod
    def get_models_for_dataset(cls, dataset: str) -> List[str]:
        """Return all valid models for a dataset."""
        return [model for (ds, model) in cls.VALID_COMBINATIONS.keys() if ds == dataset]
    
    @classmethod
    def get_datasets(cls) -> List[str]:
        """Return all supported datasets."""
        return sorted(set(ds for (ds, _) in cls.VALID_COMBINATIONS.keys()))
    
    @classmethod
    def get_all_combinations(cls) -> List[Tuple[str, str]]:
        """Return all valid (dataset, model) pairs."""
        return list(cls.VALID_COMBINATIONS.keys())
