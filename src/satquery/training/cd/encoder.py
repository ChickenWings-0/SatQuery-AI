"""The shared encoder, initialised from SSL4EO-S12 where the weights are present.

Change datasets are small. LEVIR-CD is 637 tiles and OSCD is 24 scene pairs, and
an encoder trained from scratch on either memorises the training split — which
does not show up as a bad validation F1, because the validation split shares the
same sensor, season and radiometry. It shows up in exactly the place we care
about: transferring the 0.5 m model to 10 m imagery, and transferring either to
Cartosat. Self-supervised pretraining on SSL4EO-S12's 1.2 M Sentinel-1/2 patches
is the cheapest available defence, so it is the default.

**Band adaptation is the interesting part.** SSL4EO-S12 weights are trained on
13-band Sentinel-2. LEVIR-CD is 3-band RGB VHR aerial. The stem convolution
therefore has to be re-shaped from 13 input channels to 3, and *how* decides
whether the pretraining survives:

* Averaging all 13 filters into each new channel — the usual quick fix — destroys
  the spectral selectivity the encoder learned. Every input channel then sees the
  same filter bank.
* Selecting the S2 bands that correspond to the ones we actually have (B4/B3/B2
  for RGB, B8 for NIR) keeps each filter matched to the wavelength it was trained
  on, and rescales for the changed channel count so activation magnitudes carry
  over.

The second is what this module does, and it is why :data:`SENTINEL2_BAND_INDEX`
exists rather than a comment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import Tensor, nn

log = logging.getLogger(__name__)

SENTINEL2_BAND_INDEX: Final[dict[str, int]] = {
    "coastal": 0,
    "blue": 1,
    "green": 2,
    "red": 3,
    "rededge1": 4,
    "rededge2": 5,
    "rededge3": 6,
    "nir": 7,
    "nir08": 8,
    "watervapour": 9,
    "cirrus": 10,
    "swir1": 11,
    "swir2": 12,
}
"""Position of each logical band in SSL4EO-S12's 13-channel Sentinel-2 stack.
This is the mapping the stem surgery indexes into; getting it wrong silently
pairs the red filter with the blue channel and looks like a bad training run."""

DEFAULT_BANDS: Final[tuple[str, ...]] = ("red", "green", "blue")
"""What LEVIR-CD provides, and the minimum every optical pair in the registry
declares as ``required_bands``."""

SSL4EO_WEIGHTS: Final[dict[str, tuple[str, ...]]] = {
    "resnet18": ("ResNet18_Weights.SENTINEL2_ALL_MOCO", "RESNET18_SENTINEL2_ALL_MOCO"),
    "resnet50": ("ResNet50_Weights.SENTINEL2_ALL_MOCO", "RESNET50_SENTINEL2_ALL_MOCO"),
}
"""torchgeo weight enum members carrying SSL4EO-S12 pretraining, by backbone.

Two spellings per backbone, newest first. torchgeo 0.8 renamed these to the
qualified ``<Enum>.<MEMBER>`` form; the bare ``RESNET18_...`` name is what 0.7
and earlier answered to. Both are tried because getting this wrong does not
fail — :func:`_torchgeo_weights` falls back to random initialisation, and a run
that quietly starts from noise instead of SSL4EO-S12 looks exactly like a run
that did not, until the F1 gate is missed."""

FEATURE_CHANNELS: Final[dict[str, tuple[int, ...]]] = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
}
"""Channel width of each ResNet stage output, at strides 4, 8, 16 and 32."""


class EncoderError(RuntimeError):
    """The encoder could not be constructed as configured."""


@dataclass(frozen=True)
class EncoderReport:
    """How the encoder was actually initialised.

    Recorded into the checkpoint and, from there, into the trace: an ablation row
    that cannot say whether its encoder was pretrained is not a result.
    """

    backbone: str
    in_channels: int
    bands: tuple[str, ...]
    pretrained: bool
    source: str
    """``"ssl4eo-s12"``, ``"imagenet"`` or ``"random"``."""

    note: str = ""


def _torchgeo_weights(backbone: str) -> Any | None:
    """Fetch the SSL4EO-S12 weight enum for *backbone*, if torchgeo is installed.

    Returns None rather than raising: a machine without torchgeo can still train
    (worse) and can still run inference from a checkpoint (identically), and
    turning a missing optional dependency into a crash would break both.
    """
    members = SSL4EO_WEIGHTS.get(backbone)
    if not members:
        return None
    try:
        from torchgeo.models import api as torchgeo_api  # noqa: F401
        from torchgeo.models import get_weight
    except ImportError:
        log.warning(
            "torchgeo is not installed, so SSL4EO-S12 weights cannot be loaded. "
            "Install it with `uv sync --extra cd`; training will otherwise start "
            "from ImageNet or random initialisation."
        )
        return None
    for member in members:
        try:
            return get_weight(member)
        except (ValueError, KeyError):  # torchgeo version skew; try the next spelling
            continue
    log.warning(
        "torchgeo %s exposes none of %s, so the encoder starts from random "
        "initialisation rather than SSL4EO-S12.",
        getattr(__import__("torchgeo"), "__version__", "?"),
        list(members),
    )
    return None


def adapt_stem(
    weight: Tensor, source_bands: int, bands: tuple[str, ...]
) -> Tensor:
    """Re-shape a pretrained stem convolution onto the bands we actually have.

    Args:
        weight: The pretrained ``(out, source_bands, k, k)`` stem kernel.
        source_bands: How many channels the pretraining used.
        bands: Logical band names of our input, in channel order.

    Returns:
        An ``(out, len(bands), k, k)`` kernel.

    Raises:
        EncoderError: The kernel does not have the expected channel count.
    """
    if weight.shape[1] != source_bands:
        raise EncoderError(
            f"stem kernel has {weight.shape[1]} input channels, expected {source_bands}"
        )

    columns: list[Tensor] = []
    for band in bands:
        index = SENTINEL2_BAND_INDEX.get(band)
        if index is not None and index < source_bands:
            columns.append(weight[:, index])
        else:
            # A band the pretraining never saw — a panchromatic channel, say —
            # gets the mean filter. That is the honest default for a channel with
            # no wavelength counterpart, and it applies to that channel alone
            # rather than to all of them.
            columns.append(weight.mean(dim=1))
    adapted = torch.stack(columns, dim=1)

    # Summing over fewer channels shrinks the pre-activation scale, and a stem
    # whose outputs are a third of what the following BatchNorm's running
    # statistics expect wastes the first epochs re-learning them.
    return adapted * (source_bands / max(1, len(bands)))


class ResNetEncoder(nn.Module):
    """A ResNet trunk exposing its four stage outputs.

    torchvision rather than timm: it is already a dependency, the ResNet layout
    is stable across versions, and the stage outputs are reachable without a
    feature-extraction wrapper whose keys change between releases.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        bands: tuple[str, ...] = DEFAULT_BANDS,
        pretrained: bool = True,
    ) -> None:
        """Build the trunk and initialise it as well as this machine allows.

        Raises:
            EncoderError: *backbone* is not a supported ResNet.
        """
        super().__init__()
        if backbone not in FEATURE_CHANNELS:
            raise EncoderError(
                f"unsupported backbone {backbone!r}; choose from {sorted(FEATURE_CHANNELS)}"
            )
        self.backbone_name = backbone
        self.bands = tuple(bands)
        self.channels = FEATURE_CHANNELS[backbone]

        trunk, report = self._build(backbone, pretrained)
        self.report = report
        self.stem = nn.Sequential(trunk.conv1, trunk.bn1, trunk.relu)
        self.pool = trunk.maxpool
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

    def _build(self, backbone: str, pretrained: bool) -> tuple[Any, EncoderReport]:
        """Construct the torchvision trunk and load whatever weights are available."""
        from torchvision.models import get_model

        in_channels = len(self.bands)
        trunk = get_model(backbone, weights=None)
        source = "random"
        note = ""

        weights = _torchgeo_weights(backbone) if pretrained else None
        if weights is not None:
            try:
                state = weights.get_state_dict(progress=False)
                source, note = self._load_ssl4eo(trunk, state)
            except Exception as error:  # noqa: BLE001 - offline or cache miss
                note = f"SSL4EO-S12 weights could not be fetched ({error}); starting random"
                log.warning(note)
        elif pretrained:
            note = "torchgeo unavailable; encoder starts from random initialisation"

        # Re-shape the stem last, so it applies whether the weights came from
        # SSL4EO or from nothing at all.
        stem = trunk.conv1
        trunk.conv1 = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=stem.bias is not None,
        )
        if source == "ssl4eo-s12":
            with torch.no_grad():
                trunk.conv1.weight.copy_(
                    adapt_stem(self._pretrained_stem, 13, self.bands)
                )

        return trunk, EncoderReport(
            backbone=backbone,
            in_channels=in_channels,
            bands=self.bands,
            pretrained=source != "random",
            source=source,
            note=note,
        )

    def _load_ssl4eo(self, trunk: Any, state: dict[str, Tensor]) -> tuple[str, str]:
        """Load an SSL4EO state dict, keeping the 13-band stem aside for adaptation."""
        self._pretrained_stem = state["conv1.weight"].clone()
        # The stem is withheld deliberately. SSL4EO's is 13-channel and the trunk's
        # is 3-channel, and ``strict=False`` forgives a *missing* key but not a
        # shape mismatch — it raises, which the caller turns into a silent fall
        # back to random initialisation. It is re-applied, band-adapted, by
        # :func:`adapt_stem` once the trunk has been reshaped.
        payload = {key: value for key, value in state.items() if key != "conv1.weight"}
        missing, unexpected = trunk.load_state_dict(payload, strict=False)
        skipped = [
            key for key in missing if not key.startswith("fc.") and key != "conv1.weight"
        ]
        note = (
            f"SSL4EO-S12: {len(skipped)} tensors were not in the checkpoint"
            if skipped
            else "SSL4EO-S12 weights loaded"
        )
        if unexpected:
            note += f"; {len(unexpected)} unused"
        return "ssl4eo-s12", note

    def forward(self, x: Tensor) -> list[Tensor]:
        """Return the four stage feature maps, at strides 4, 8, 16 and 32."""
        x = self.pool(self.stem(x))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


__all__ = [
    "DEFAULT_BANDS",
    "FEATURE_CHANNELS",
    "SENTINEL2_BAND_INDEX",
    "SSL4EO_WEIGHTS",
    "EncoderError",
    "EncoderReport",
    "ResNetEncoder",
    "adapt_stem",
]
