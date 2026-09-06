"""Model serving: the VLM that synthesises answers out of deterministic evidence.

Nothing in this package decides *what* to measure. By the time a backend here is
called, every number that may appear in the answer has already been measured and
is sitting in a :class:`~satquery.evidence.fact_sheet.FactSheet`. The model's
entire job is to turn those measurements and the rendered views into a sentence
a human wants to read (Master.md §8 Phase 4).
"""

from satquery.models.loader import (
    VRAM_BUDGET_BYTES,
    BackendKind,
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
    ModelLoadError,
    PromptImage,
    VlmBackend,
    VramGuardError,
    available_backend,
    get_backend,
    release_backend,
    resolve_torch_device,
    vram_snapshot,
)

__all__ = [
    "VRAM_BUDGET_BYTES",
    "BackendKind",
    "BackendUnavailableError",
    "GenerationRequest",
    "GenerationResult",
    "ModelLoadError",
    "PromptImage",
    "VlmBackend",
    "VramGuardError",
    "available_backend",
    "get_backend",
    "release_backend",
    "resolve_torch_device",
    "vram_snapshot",
]
