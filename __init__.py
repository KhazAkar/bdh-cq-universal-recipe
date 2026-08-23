"""BDH-CQ Universal Fine-Tuning Recipe Library"""
from .config import get_full_config, get_phase_config
from .implementations import (
    MemoryBuffer, 
    LatentReasoningStep, 
    BDHCQModel,
    ReasoningAdapter,
    UniversalReasoningWrapper,
    ModelCapabilities,
    RecurrentReasoningModule
)
from .training import BDHCQTrainer, CurriculumCallback, ARCTaskDataset, arc_collate_fn
from .utils import load_arc_dataset, create_dataloaders
__version__ = "1.0.0"
__all__ = [
    "get_full_config", "get_phase_config", 
    "MemoryBuffer", "LatentReasoningStep", "BDHCQModel", 
    "ReasoningAdapter", "UniversalReasoningWrapper", "ModelCapabilities",
    "RecurrentReasoningModule",
    "BDHCQTrainer", "CurriculumCallback", "ARCTaskDataset", 
    "arc_collate_fn", "load_arc_dataset", "create_dataloaders"
]
