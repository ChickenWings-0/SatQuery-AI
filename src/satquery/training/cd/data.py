"""torchgeo datamodules for LEVIR-CD and OSCD, adapted to one batch contract.

Two datasets, deliberately, and they are not redundant:

* **LEVIR-CD** — 637 pairs of 1024x1024 Google Earth tiles at ~0.5 m, building
  change. This is the VHR regime, and the closest public proxy for the Cartosat-2S
  imagery the hidden evaluation set uses.
* **OSCD** — 24 Sentinel-2 scene pairs at 10 m, urban expansion. This is the
  multispectral regime and the one the mandated BigEarthNet sensor actually
  matches.

Training the same architecture on both and reporting the gap is Master.md §8
Phase 5's cross-resolution evidence. That only means something if the two runs
differ in the data and *nothing else*, which is why both go through one
:class:`ChangeDataModule` facade rather than two bespoke pipelines.

**The facade earns its place at the batch contract.** torchgeo's change datasets
have not agreed on a batch layout across releases: some yield ``image1`` and
``image2``, others a single stacked ``image`` of shape ``(B, 2, C, H, W)``, and
the mask arrives with or without a channel axis. :func:`normalise_batch` pins all
of that down to ``{"image1", "image2", "mask"}`` once, so the Lightning task never
carries a version check.

Nothing here downloads anything unless ``download=True`` is passed explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor

DATASET_GSD_M: Final[dict[str, float]] = {"levircd": 0.5, "oscd": 10.0}
"""Ground sample distance per dataset. Recorded into the checkpoint, which is
what lets the two ablation rows be compared as a resolution study rather than as
two unrelated numbers."""

DATASET_BANDS: Final[dict[str, tuple[str, ...]]] = {
    "levircd": ("red", "green", "blue"),
    # OSCD carries all 13 Sentinel-2 bands. RGB keeps one architecture across
    # both datasets, which is the point of the comparison; `--bands all` in the
    # training script widens the stem for a separate run.
    "oscd": ("red", "green", "blue"),
}

DATASET_NAMES: Final[tuple[str, ...]] = ("levircd", "oscd")

DEFAULT_PATCH_PX: Final[int] = 256
DEFAULT_BATCH: Final[int] = 8
"""Eight 256 px pairs of a ResNet-18 Siamese fit in well under 24 GB at bf16.
The batch is small because the datasets are, not because the card is."""


class DataError(RuntimeError):
    """A datamodule could not be constructed."""


def _torchgeo_datamodules() -> Any:
    """Import torchgeo's datamodules, or explain what to install.

    Raises:
        DataError: torchgeo is not installed.
    """
    try:
        from torchgeo import datamodules
    except ImportError as error:
        raise DataError(
            "torchgeo is required for the change-detection datamodules. Install "
            "it with `uv sync --extra cd`. Serving a trained checkpoint does not "
            "need it."
        ) from error
    return datamodules


def normalise_batch(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Pin one torchgeo change batch to ``image1`` / ``image2`` / ``mask``.

    Handles both layouts torchgeo has shipped — separate epochs, and a single
    stacked ``(B, 2, C, H, W)`` tensor — and gives the mask a channel axis so the
    loss never has to broadcast.

    Raises:
        DataError: The batch carries neither layout.
    """
    if "image1" in batch and "image2" in batch:
        pre, post = batch["image1"], batch["image2"]
    elif "image" in batch:
        stacked = batch["image"]
        if stacked.ndim < 4 or stacked.shape[-4] != 2:
            raise DataError(
                f"a stacked change batch must be (..., 2, C, H, W), got "
                f"{tuple(stacked.shape)}"
            )
        pre, post = stacked[..., 0, :, :, :], stacked[..., 1, :, :, :]
    else:
        raise DataError(
            f"batch has neither image1/image2 nor a stacked image: {sorted(batch)}"
        )

    mask = batch.get("mask")
    if mask is None:
        raise DataError("a change batch must carry a mask")
    mask = mask.float()
    if mask.ndim == pre.ndim - 1:
        mask = mask.unsqueeze(-3)
    return {"image1": pre.float(), "image2": post.float(), "mask": mask}


@dataclass(frozen=True)
class DatasetProfile:
    """What one dataset contributes to a checkpoint's provenance."""

    name: str
    gsd_m: float
    bands: tuple[str, ...]

    @classmethod
    def for_dataset(cls, name: str) -> DatasetProfile:
        """Look up a profile by dataset name.

        Raises:
            DataError: The dataset is not one this module supports.
        """
        key = name.strip().lower()
        if key not in DATASET_GSD_M:
            raise DataError(f"unknown dataset {name!r}; choose from {list(DATASET_NAMES)}")
        return cls(name=key, gsd_m=DATASET_GSD_M[key], bands=DATASET_BANDS[key])


class ChangeDataModule:
    """A thin, version-tolerant facade over one torchgeo change datamodule.

    Composition rather than subclassing: subclassing ``LEVIRCDDataModule`` would
    bind us to whichever constructor signature the installed torchgeo release
    happens to have, and those have changed. Wrapping the dataloaders costs one
    indirection and survives the churn.
    """

    def __init__(
        self,
        dataset: str = "levircd",
        root: Path | str = "data/raw",
        batch_size: int = DEFAULT_BATCH,
        patch_size: int = DEFAULT_PATCH_PX,
        num_workers: int = 4,
        download: bool = False,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Configure the datamodule without constructing or downloading anything.

        Args:
            dataset: ``"levircd"`` or ``"oscd"``.
            root: Where the archives live, or will be downloaded to.
            batch_size: Pairs per step.
            patch_size: Random crop size in pixels.
            num_workers: Dataloader workers.
            download: Fetch the archive if it is absent. **Off by default** — a
                dataset download is an explicit act, never a side effect of
                importing or of a test constructing this class.
            val_split_pct: OSCD ships no validation split, so one is carved out.
            **kwargs: Passed to the underlying torchgeo datamodule.

        Raises:
            DataError: The dataset name is not supported.
        """
        self.profile = DatasetProfile.for_dataset(dataset)
        self.root = Path(root)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.download = download
        self.val_split_pct = val_split_pct
        self.kwargs = kwargs
        self._inner: Any | None = None

    @property
    def name(self) -> str:
        """The dataset this module serves."""
        return self.profile.name

    def build(self) -> Any:
        """Construct the underlying torchgeo datamodule, once.

        Raises:
            DataError: torchgeo is missing, or refused the configuration.
        """
        if self._inner is not None:
            return self._inner

        datamodules = _torchgeo_datamodules()
        common: dict[str, Any] = {
            "batch_size": self.batch_size,
            "patch_size": self.patch_size,
            "num_workers": self.num_workers,
            "root": str(self.root),
            "download": self.download,
            **self.kwargs,
        }
        if self.name == "levircd":
            factory = getattr(
                datamodules, "LEVIRCDDataModule", None
            ) or datamodules.LEVIRCDPlusDataModule
        else:
            factory = datamodules.OSCDDataModule
            common.setdefault("val_split_pct", self.val_split_pct)
            # OSCD's 13-band product would otherwise widen the stem and make the
            # two ablation rows differ in more than resolution.
            common.setdefault("bands", "rgb")

        try:
            self._inner = factory(**common)
        except TypeError as error:
            raise DataError(
                f"the installed torchgeo's {factory.__name__} rejected "
                f"{sorted(common)}: {error}"
            ) from error
        return self._inner

    def setup(self, stage: str | None = None) -> None:
        """Prepare and split the dataset."""
        inner = self.build()
        if hasattr(inner, "prepare_data"):
            inner.prepare_data()
        inner.setup(stage or "fit")

    def _loader(self, split: str) -> Any:
        """Wrap one torchgeo dataloader so its batches match our contract."""
        inner = self.build()
        loader = getattr(inner, f"{split}_dataloader")()
        return _AdaptedLoader(loader)

    def train_dataloader(self) -> Any:
        """Training batches, normalised."""
        return self._loader("train")

    def val_dataloader(self) -> Any:
        """Validation batches, normalised."""
        return self._loader("val")

    def test_dataloader(self) -> Any:
        """Test batches, normalised."""
        return self._loader("test")


class _AdaptedLoader:
    """Applies :func:`normalise_batch` to every batch of a wrapped dataloader."""

    def __init__(self, loader: Any) -> None:
        """Wrap *loader*."""
        self.loader = loader

    def __iter__(self) -> Any:
        """Yield normalised batches."""
        for batch in self.loader:
            yield normalise_batch(batch)

    def __len__(self) -> int:
        """Number of batches, when the wrapped loader knows."""
        return len(self.loader)

    def __getattr__(self, name: str) -> Any:
        """Expose the wrapped loader's other attributes (``dataset``, ``sampler``)."""
        return getattr(self.loader, name)


def channel_statistics(
    loader: Any, max_batches: int = 64
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Estimate per-channel mean and standard deviation over both epochs.

    Both epochs are pooled into one estimate deliberately: standardising them
    separately would remove exactly the radiometric difference the model is
    supposed to detect.

    Args:
        loader: Any iterable of normalised batches.
        max_batches: Cap, so this is seconds rather than a full pass.

    Returns:
        ``(mean, std)`` per channel, ready for
        :class:`~satquery.training.cd.checkpoint.Normalisation`.
    """
    totals: Tensor | None = None
    squares: Tensor | None = None
    count = 0.0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        pixels = torch.cat([batch["image1"], batch["image2"]], dim=0)
        flat = pixels.transpose(0, 1).reshape(pixels.shape[1], -1).double()
        totals = flat.sum(dim=1) if totals is None else totals + flat.sum(dim=1)
        squares = (flat**2).sum(dim=1) if squares is None else squares + (flat**2).sum(dim=1)
        count += float(flat.shape[1])

    if totals is None or squares is None or count == 0:
        raise DataError("the loader yielded no batches to estimate statistics from")

    mean = totals / count
    variance = torch.clamp(squares / count - mean**2, min=1e-12)
    return (
        tuple(round(float(v), 6) for v in mean),
        tuple(round(float(v), 6) for v in variance.sqrt()),
    )


__all__ = [
    "DATASET_BANDS",
    "DATASET_GSD_M",
    "DATASET_NAMES",
    "DEFAULT_BATCH",
    "DEFAULT_PATCH_PX",
    "ChangeDataModule",
    "DataError",
    "DatasetProfile",
    "channel_statistics",
    "normalise_batch",
]
