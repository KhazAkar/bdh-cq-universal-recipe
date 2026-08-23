"""Configuration for BDH-CQ Universal Recipe"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

@dataclass
class PaperMetadata:
    id: str = "arXiv:2608.09888"
    title: str = "BDH-CQ: In-Context Learning with Recurrent Latent Reasoning"
    authors: List[str] = field(default_factory=lambda: ["Bjorn Engdahl", "Adrian Kosowski", "Jan Chorowski", "Zuzanna Stamirowska", "Przemyslaw Uznanski", "Junlin Jiang", "Rohan Phadke", "Remigiusz Kinas", "Richard Zhong"])
    date: str = "2026-08-10"
    url: str = "https://arxiv.org/abs/2608.09888"

@dataclass
class KeyAchievement:
    pass_at_2: float = 0.295
    cost_per_task_usd: float = 0.0007
    cost_vs_gpt56_luna: float = 57.0
    model_size: str = "150M"
    benchmark: str = "ARC-AGI-1"

@dataclass
class FullConfig:
    metadata: PaperMetadata = field(default_factory=PaperMetadata)
    key_achievement: KeyAchievement = field(default_factory=KeyAchievement)
    export_date: str = "2026-08-23"
    version: str = "1.0"

_full_config: Optional[FullConfig] = None

def get_full_config() -> FullConfig:
    global _full_config
    if _full_config is None:
        _full_config = FullConfig()
    return _full_config

def get_phase_config(phase: int) -> Dict[str, Any]:
    config = get_full_config()
    return {"metadata": {"paper": config.metadata.__dict__, "achievement": config.key_achievement.__dict__}, "version": config.version}
