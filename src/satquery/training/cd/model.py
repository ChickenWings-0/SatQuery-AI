"""The Siamese change detector: shared encoder, per-scale differencing, FPN decoder.

Architecture, and why each part is the way it is:

* **One encoder, applied twice.** Weight sharing forces both epochs into the same
  feature space. Two independent encoders would each learn to describe their own
  epoch, and the difference between two unrelated descriptions is noise.
* **Absolute difference, not concatenation.** ``|f(pre) - f(post)|`` is symmetric
  in the two epochs by construction, so the model cannot learn "the second image
  is the new one" as a shortcut, and swapping the inputs cannot change the
  answer. Concatenation is strictly more expressive and, on 637 training tiles,
  reliably spends that expressiveness on memorising acquisition order.
* **All four scales.** LEVIR-CD change is buildings — tens of pixels at 0.5 m —
  while OSCD change is urban expansion at 10 m, hundreds of metres across. A
  decoder reading only the deepest stage resolves the second and misses the
  first; the FPN path is what lets one architecture serve both resolutions, which
  is the cross-resolution evidence Phase 5 is asked to produce.
* **One logit channel.** Change is binary here. Per-class change is a Phase 6
  concern and arrives through the segmenter, not through this head — see
  ``change_statistics``.

Only ``torch`` is imported. Lightning and torchgeo live in
:mod:`satquery.training.cd.module` and :mod:`satquery.training.cd.data`, so a
serving checkout can instantiate this model and load weights into it without
either installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from satquery.training.cd.encoder import DEFAULT_BANDS, EncoderReport, ResNetEncoder

DECODER_CHANNELS: Final[int] = 64
"""Width of every FPN level. Uniform, so the top-down path is a plain addition
and the decoder's cost does not grow with the backbone's."""


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """3x3 convolution, normalisation, activation.

    GroupNorm rather than BatchNorm: change detection trains at batch sizes of 4
    to 8 on a 24 GB card at 512 px, and BatchNorm statistics estimated over four
    samples are noise that shows up as unstable validation F1 between epochs.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.ReLU(inplace=True),
    )


def _as_int(value: object, default: int) -> int:
    """Read an integer out of a checkpoint record, defaulting on anything else."""
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _as_float(value: object, default: float) -> float:
    """Read a float out of a checkpoint record, defaulting on anything else."""
    return (
        float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default
    )


@dataclass
class SiameseConfig:
    """Everything that defines one detector architecture.

    Stored in the checkpoint verbatim, because rebuilding the model with a
    different config and loading the weights anyway produces a network that runs,
    scores nothing like the training run, and reports no error at all.
    """

    backbone: str = "resnet18"
    bands: tuple[str, ...] = DEFAULT_BANDS
    decoder_channels: int = DECODER_CHANNELS
    dropout: float = 0.1
    pretrained: bool = True

    @property
    def in_channels(self) -> int:
        """Input channels per epoch."""
        return len(self.bands)

    def to_dict(self) -> dict[str, object]:
        """A JSON-safe record for the checkpoint."""
        return {
            "backbone": self.backbone,
            "bands": list(self.bands),
            "decoder_channels": self.decoder_channels,
            "dropout": self.dropout,
            "pretrained": self.pretrained,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> SiameseConfig:
        """Rebuild a config from its checkpoint record.

        Every field is coerced defensively. The record is whatever JSON the
        bundle carried, so a malformed entry falls back to the declared default
        rather than raising here — the state dict load that follows is the check
        that actually matters, and it will fail loudly if the shape is wrong.
        """
        bands = raw.get("bands") or DEFAULT_BANDS
        return cls(
            backbone=str(raw.get("backbone", "resnet18")),
            bands=tuple(str(b) for b in bands) if isinstance(bands, Iterable) else DEFAULT_BANDS,
            decoder_channels=_as_int(raw.get("decoder_channels"), DECODER_CHANNELS),
            dropout=_as_float(raw.get("dropout"), 0.1),
            pretrained=bool(raw.get("pretrained", True)),
        )


class ChangeDecoder(nn.Module):
    """Top-down FPN over the per-scale differences, back to full resolution."""

    def __init__(
        self, encoder_channels: tuple[int, ...], channels: int = DECODER_CHANNELS,
        dropout: float = 0.1,
    ) -> None:
        """Project every scale to a common width and fuse coarse into fine."""
        super().__init__()
        self.laterals = nn.ModuleList(
            nn.Conv2d(width, channels, kernel_size=1) for width in encoder_channels
        )
        self.smooth = nn.ModuleList(
            _conv_block(channels, channels) for _ in encoder_channels[:-1]
        )
        self.head = nn.Sequential(
            _conv_block(channels, channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, differences: list[Tensor], size: tuple[int, int]) -> Tensor:
        """Fuse the per-scale differences and upsample to *size*.

        Args:
            differences: ``|f(pre) - f(post)|`` at strides 4, 8, 16, 32.
            size: ``(height, width)`` of the input, so the logits come back on
                the grid the caller supplied rather than a power-of-two of it.

        Returns:
            A ``(B, 1, H, W)`` logit map.
        """
        projected = [
            lateral(feature)
            for lateral, feature in zip(self.laterals, differences, strict=True)
        ]
        merged = projected[-1]
        for index in range(len(projected) - 2, -1, -1):
            upsampled = F.interpolate(
                merged, size=projected[index].shape[-2:], mode="bilinear", align_corners=False
            )
            merged = self.smooth[index](projected[index] + upsampled)
        logits = self.head(merged)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


class SiameseChangeDetector(nn.Module):
    """Shared-encoder change detector producing one logit per input pixel."""

    def __init__(self, config: SiameseConfig | None = None) -> None:
        """Build the encoder and decoder described by *config*."""
        super().__init__()
        self.config = config or SiameseConfig()
        self.encoder = ResNetEncoder(
            backbone=self.config.backbone,
            bands=self.config.bands,
            pretrained=self.config.pretrained,
        )
        self.decoder = ChangeDecoder(
            encoder_channels=self.encoder.channels,
            channels=self.config.decoder_channels,
            dropout=self.config.dropout,
        )

    @property
    def encoder_report(self) -> EncoderReport:
        """How the encoder was initialised, for the checkpoint's provenance."""
        return self.encoder.report

    def forward(self, pre: Tensor, post: Tensor) -> Tensor:
        """Return per-pixel change logits for a co-registered pair.

        Args:
            pre: ``(B, C, H, W)`` pre-change epoch.
            post: ``(B, C, H, W)`` post-change epoch, same grid.

        Returns:
            ``(B, 1, H, W)`` logits. Positive means changed.

        Raises:
            ValueError: The two epochs are not on the same grid.
        """
        if pre.shape != post.shape:
            raise ValueError(
                f"the two epochs must share a grid, got {tuple(pre.shape)} and "
                f"{tuple(post.shape)}"
            )
        # Batching both epochs through one forward pass rather than calling the
        # encoder twice halves the kernel-launch overhead and, more importantly,
        # guarantees identical weights are applied — there is no code path in
        # which the two epochs could see different parameters.
        features = self.encoder(torch.cat([pre, post], dim=0))
        batch = pre.shape[0]
        differences = [torch.abs(f[:batch] - f[batch:]) for f in features]
        logits: Tensor = self.decoder(differences, size=pre.shape[-2:])
        return logits

    @torch.no_grad()
    def predict_probabilities(self, pre: Tensor, post: Tensor) -> Tensor:
        """Return per-pixel change probabilities in ``[0, 1]``."""
        return torch.sigmoid(self.forward(pre, post))


@dataclass
class ModelSummary:
    """Parameter counts, for the training log and the checkpoint."""

    total: int
    trainable: int
    encoder: int
    decoder: int
    extras: dict[str, object] = field(default_factory=dict)


def summarise(model: SiameseChangeDetector) -> ModelSummary:
    """Count parameters by part."""
    def _count(module: nn.Module, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in module.parameters() if p.requires_grad or not trainable_only
        )

    return ModelSummary(
        total=_count(model),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        encoder=_count(model.encoder),
        decoder=_count(model.decoder),
        extras={"encoder_source": model.encoder_report.source},
    )


def build_detector(
    backbone: str = "resnet18",
    bands: tuple[str, ...] = DEFAULT_BANDS,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> SiameseChangeDetector:
    """Construct a detector from the parameters the training script exposes."""
    return SiameseChangeDetector(
        SiameseConfig(backbone=backbone, bands=bands, pretrained=pretrained, dropout=dropout)
    )


__all__ = [
    "DECODER_CHANNELS",
    "ChangeDecoder",
    "ModelSummary",
    "SiameseChangeDetector",
    "SiameseConfig",
    "build_detector",
    "summarise",
]
