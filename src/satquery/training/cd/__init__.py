"""Bi-temporal change detection: the model, its data, and its checkpoints.

The lineage is TinyCD/ChangeFormer: one **shared** encoder applied to both
epochs, per-scale absolute differencing, and a light FPN decoder back to full
resolution. Siamese weight sharing is the load-bearing choice — two independent
encoders learn to describe each epoch, whereas one encoder applied twice is
forced to put the two epochs in the same feature space, which is the only thing
that makes their difference mean anything.

The encoder is initialised from SSL4EO-S12 (Master.md §8 Phase 5). Change
datasets are small — LEVIR-CD is 637 tiles — and a randomly initialised encoder
trained on them memorises rather than generalises, which shows up immediately as
cross-resolution collapse between the 0.5 m and 10 m runs.

Imports here are deliberately shallow: ``model`` and ``checkpoint`` need only
torch, so the serving path can load a checkpoint without Lightning or torchgeo
installed. ``module`` and ``data`` are training-only and pull those in.
"""

from satquery.training.cd.checkpoint import (
    CheckpointBundle,
    CheckpointError,
    Normalisation,
    default_checkpoint_path,
    find_checkpoint,
    load_bundle,
    save_bundle,
)
from satquery.training.cd.model import (
    ChangeDecoder,
    ModelSummary,
    SiameseChangeDetector,
    SiameseConfig,
    build_detector,
    summarise,
)

__all__ = [
    "ChangeDecoder",
    "CheckpointBundle",
    "CheckpointError",
    "ModelSummary",
    "Normalisation",
    "SiameseChangeDetector",
    "SiameseConfig",
    "build_detector",
    "default_checkpoint_path",
    "find_checkpoint",
    "load_bundle",
    "save_bundle",
    "summarise",
]
