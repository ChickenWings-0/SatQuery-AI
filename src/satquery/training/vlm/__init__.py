"""Phase 7 VLM domain adaptation: the QLoRA profile, its guards and its trainer.

Importing this package pulls in no torch, no transformers and no bitsandbytes.
Everything that needs them — model loading, the PEFT wiring, the trainer — is
imported inside the function that uses it, so the API, the deterministic layer
and the whole test suite keep running on a machine with none of them installed.
"""

from satquery.training.vlm.qlora import (
    DEFAULT_PROFILE_PATH,
    LLM_TARGET_MODULES,
    VISION_TARGET_MODULES,
    DataSpec,
    LoraSpec,
    QLoraProfile,
    QuantizationSpec,
    TrainSpec,
    VramEstimate,
    estimate_footprint,
    load_profile,
    resolve_target_modules,
    sample_to_chat,
    select_vision_blocks,
)

__all__ = [
    "DEFAULT_PROFILE_PATH",
    "LLM_TARGET_MODULES",
    "VISION_TARGET_MODULES",
    "DataSpec",
    "LoraSpec",
    "QLoraProfile",
    "QuantizationSpec",
    "TrainSpec",
    "VramEstimate",
    "estimate_footprint",
    "load_profile",
    "resolve_target_modules",
    "sample_to_chat",
    "select_vision_blocks",
]
