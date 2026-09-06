"""The Lightning task: loss, metrics, optimisation and threshold calibration.

Three decisions carry the training run, and all three are about class imbalance.
Under 5 % of pixels change in LEVIR-CD and under 2 % in OSCD, so:

* **The loss is BCE + soft Dice.** Plain BCE is minimised acceptably well by
  predicting "no change" everywhere, which scores 95 % pixel accuracy and an F1
  of zero. Dice is computed over the positive class and is therefore not
  satisfied by that solution. BCE stabilises the early epochs when Dice's
  gradient is nearly flat; Dice supplies the signal once anything is predicted.
* **The reported metric is F1 over the changed class**, never pixel accuracy.
  Master.md's ``F1 >= 0.88`` gate is on this number.
* **The threshold is calibrated after training, not fixed at 0.5.** ``0.5`` is
  where the loss is minimised; the operating point that maximises F1 on
  imbalanced data is elsewhere, usually lower. :meth:`ChangeDetectionTask.
  on_validation_epoch_end` sweeps it and the best value travels in the
  checkpoint bundle.

ROCm notes: nothing here is CUDA-specific. ``bf16-mixed`` is the precision to
train at on gfx1100 — fp16 overflows in the deep stages of a ResNet trunk on this
architecture, and bf16's exponent range is the whole reason it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import Tensor

from satquery.training.cd.model import SiameseChangeDetector, SiameseConfig

DICE_EPS: Final[float] = 1.0
"""Laplace smoothing on the Dice quotient. Also what keeps a tile with no changed
pixels at all from producing 0/0 — it scores 1.0, which is correct: predicting no
change on a tile with no change is right."""

THRESHOLD_SWEEP: Final[tuple[float, ...]] = tuple(round(0.05 * i, 2) for i in range(1, 20))
"""0.05 to 0.95. Fixed and coarse on purpose: a finer sweep over one validation
split fits the split rather than the operating point."""


def dice_loss(logits: Tensor, target: Tensor, eps: float = DICE_EPS) -> Tensor:
    """Soft Dice over the changed class, averaged across the batch."""
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    total = probabilities.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (total + eps)).mean()


@dataclass
class ConfusionCounts:
    """Accumulated pixel counts for the changed class, at one threshold."""

    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def update(self, predicted: Tensor, target: Tensor) -> None:
        """Add one batch."""
        self.tp += float((predicted * target).sum())
        self.fp += float((predicted * (1.0 - target)).sum())
        self.fn += float(((1.0 - predicted) * target).sum())

    @property
    def precision(self) -> float:
        """Changed-class precision, 0 when nothing was predicted."""
        denominator = self.tp + self.fp
        return float(self.tp / denominator) if denominator else 0.0

    @property
    def recall(self) -> float:
        """Changed-class recall, 0 when there was nothing to find."""
        denominator = self.tp + self.fn
        return float(self.tp / denominator) if denominator else 0.0

    @property
    def f1(self) -> float:
        """Changed-class F1 — the number Master.md's Phase 5 gate is on."""
        p, r = self.precision, self.recall
        return float(2.0 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def iou(self) -> float:
        """Changed-class intersection over union."""
        denominator = self.tp + self.fp + self.fn
        return float(self.tp / denominator) if denominator else 0.0


def _lightning() -> Any:
    """Import Lightning, saying plainly what to install when it is absent.

    Raises:
        ImportError: Lightning is not installed.
    """
    try:
        import lightning.pytorch as pl
    except ImportError:
        try:
            import pytorch_lightning as pl
        except ImportError as error:
            raise ImportError(
                "PyTorch Lightning is required to train. Install it with "
                "`uv sync --extra cd`. Serving a trained checkpoint does not "
                "need it."
            ) from error
    return pl


def build_task(
    config: SiameseConfig | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    encoder_lr_scale: float = 0.1,
    bce_weight: float = 0.5,
    max_epochs: int = 60,
    positive_weight: float | None = None,
) -> Any:
    """Construct the Lightning task.

    Built inside a function because the class has to subclass
    ``LightningModule``, and importing Lightning at module scope would make the
    serving path depend on it. Everything the task does that is worth testing —
    the loss, the confusion counts, the sweep — is a module-level function above
    and is tested without Lightning installed.

    Args:
        config: Architecture; defaults to the ResNet-18 RGB detector.
        learning_rate: Peak LR for the decoder.
        weight_decay: AdamW decay, applied to weights but not to norms or biases.
        encoder_lr_scale: The pretrained encoder is fine-tuned an order of
            magnitude slower than the randomly initialised decoder. Training both
            at the same rate destroys the SSL4EO features in the first epoch,
            which is the single most common way this kind of run fails silently.
        bce_weight: Mix between BCE and Dice.
        max_epochs: Horizon for the cosine schedule.
        positive_weight: ``pos_weight`` for BCE. None derives it per batch from
            the observed imbalance, which adapts between LEVIR-CD and OSCD
            without a second hyper-parameter.

    Returns:
        A ``LightningModule`` instance.

    Raises:
        ImportError: Lightning is not installed.
    """
    pl = _lightning()

    class ChangeDetectionTask(pl.LightningModule):  # type: ignore[misc, name-defined]
        """Trains :class:`SiameseChangeDetector` and calibrates its threshold."""

        def __init__(self) -> None:
            """Build the detector and the metric accumulators."""
            super().__init__()
            self.save_hyperparameters(
                {
                    "config": (config or SiameseConfig()).to_dict(),
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "encoder_lr_scale": encoder_lr_scale,
                    "bce_weight": bce_weight,
                    "max_epochs": max_epochs,
                    "positive_weight": positive_weight,
                }
            )
            self.model = SiameseChangeDetector(config or SiameseConfig())
            self.sweep: dict[float, ConfusionCounts] = {}
            self.best_threshold = 0.5
            self.best_f1 = 0.0

        # -- forward and loss ------------------------------------------

        def forward(self, pre: Tensor, post: Tensor) -> Tensor:
            """Per-pixel change logits."""
            logits: Tensor = self.model(pre, post)
            return logits

        def _loss(self, logits: Tensor, target: Tensor) -> Tensor:
            """BCE (imbalance-weighted) mixed with soft Dice."""
            if positive_weight is not None:
                weight = torch.tensor(positive_weight, device=logits.device)
            else:
                positives = target.sum()
                total = target.numel()
                # Clamped: a batch with no changed pixels would otherwise put an
                # unbounded weight on a class that is not there.
                weight = torch.clamp(
                    (total - positives) / torch.clamp(positives, min=1.0), max=100.0
                )
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)
            return bce_weight * bce + (1.0 - bce_weight) * dice_loss(logits, target)

        @staticmethod
        def _unpack(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
            """Split a datamodule batch into pre, post and a float target."""
            pre, post = batch["image1"], batch["image2"]
            target = batch["mask"].float()
            if target.ndim == 3:
                target = target.unsqueeze(1)
            return pre, post, target

        # -- loops -----------------------------------------------------

        def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
            """One optimisation step."""
            pre, post, target = self._unpack(batch)
            loss = self._loss(self(pre, post), target)
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            return loss

        def on_validation_epoch_start(self) -> None:
            """Reset the threshold sweep."""
            self.sweep = {threshold: ConfusionCounts() for threshold in THRESHOLD_SWEEP}

        def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
            """Score one validation batch at every candidate threshold."""
            pre, post, target = self._unpack(batch)
            logits = self(pre, post)
            loss = self._loss(logits, target)
            probabilities = torch.sigmoid(logits)
            for threshold, counts in self.sweep.items():
                counts.update((probabilities >= threshold).float(), target)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss

        def on_validation_epoch_end(self) -> None:
            """Pick the operating point, and log the metric the gate is on."""
            if not self.sweep:
                return
            threshold, counts = max(self.sweep.items(), key=lambda item: (item[1].f1, -item[0]))
            self.best_threshold, self.best_f1 = threshold, counts.f1
            self.log_dict(
                {
                    "val/f1": counts.f1,
                    "val/iou": counts.iou,
                    "val/precision": counts.precision,
                    "val/recall": counts.recall,
                    "val/threshold": threshold,
                    # At the fixed 0.5 as well, so the training log shows what
                    # calibration is actually buying.
                    "val/f1_at_0.5": self.sweep[0.5].f1,
                },
                prog_bar=True,
                sync_dist=True,
            )

        def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
            """Score the held-out split at the calibrated threshold."""
            return self.validation_step(batch, batch_idx)

        # -- optimisation ----------------------------------------------

        def configure_optimizers(self) -> dict[str, Any]:
            """AdamW with a slower encoder, cosine schedule, no decay on norms."""
            decay, no_decay, encoder = [], [], []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith("encoder."):
                    encoder.append(parameter)
                elif parameter.ndim <= 1:
                    # Biases and normalisation scales. Decaying them pulls the
                    # network towards a degenerate identity and buys nothing.
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)

            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder, "lr": learning_rate * encoder_lr_scale,
                     "weight_decay": weight_decay},
                    {"params": decay, "lr": learning_rate, "weight_decay": weight_decay},
                    {"params": no_decay, "lr": learning_rate, "weight_decay": 0.0},
                ]
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=learning_rate * 0.01
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

    return ChangeDetectionTask()


def calibrate_threshold(
    probabilities: Any, targets: Any, sweep: tuple[float, ...] = THRESHOLD_SWEEP
) -> tuple[float, ConfusionCounts]:
    """Pick the threshold maximising changed-class F1.

    Kept as a plain function so the sweep can be re-run over a saved probability
    field — recalibrating for a new sensor without retraining is exactly what the
    Cartosat transfer will need.

    Args:
        probabilities: Predicted change probabilities.
        targets: Binary ground truth of the same shape.
        sweep: Candidate thresholds.

    Returns:
        ``(threshold, counts)`` for the best operating point. Ties break towards
        the lower threshold, which favours recall — a missed change is worse than
        a flagged one an analyst can dismiss.
    """
    best: tuple[float, ConfusionCounts] = (0.5, ConfusionCounts())
    for threshold in sweep:
        counts = ConfusionCounts()
        counts.update((probabilities >= threshold).float(), targets.float())
        if counts.f1 > best[1].f1:
            best = (threshold, counts)
    return best


__all__ = [
    "DICE_EPS",
    "THRESHOLD_SWEEP",
    "ConfusionCounts",
    "build_task",
    "calibrate_threshold",
    "dice_loss",
]
