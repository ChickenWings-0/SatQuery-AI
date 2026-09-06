"""Device management, the VRAM guard, and the one generation interface.

Three things live here, and nothing else does:

* **The interface both backends implement.** :class:`VlmBackend` is the whole
  contract between the tools and the model. ``hf_backend`` and
  ``llamacpp_client`` are interchangeable because neither the tools nor the
  prompt library can tell them apart — which is what makes the offline GGUF demo
  path a configuration change rather than a code path (Master.md §8 Phase 4).
* **The VRAM guard.** The box has 24 GB shared between an 8B VLM in bf16 and,
  from Phase 5, the CV models. The budget is 22 GB and it is enforced *before*
  the weights are read, because a HIP out-of-memory during ``from_pretrained``
  takes the whole process down rather than degrading one step.
* **Lazy load and unload.** A backend is constructed at import time and holds no
  weights until something calls :meth:`VlmBackend.load`. The idle sweep unloads
  it again, so a service that answers one VQA question an hour is not sitting on
  16 GB the CV tools want.

Availability is deliberately probed *offline*: :func:`available_backend` never
touches the network and never reads a weight file. A machine without the model
downloaded reports no backend, the registry marks ``vlm_*`` unavailable, and the
plan falls back to the templated answer exactly as it did in Phase 3. Making the
probe a network call would turn a flaky connection into a 40-second request.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from satquery.core.config import Settings, get_settings

Rgb = npt.NDArray[np.uint8]

VRAM_BUDGET_BYTES: Final[int] = 22 * 1024**3
"""22 GB of the 24 GB card. The remaining 2 GB is the compositor, the fragmentation
headroom HIP's caching allocator needs, and the CV models from Phase 5 on."""

BYTES_PER_PARAM: Final[dict[str, int]] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}
"""Weight footprint per parameter, by dtype. bf16 is the only one that fits."""

ACTIVATION_OVERHEAD: Final[float] = 1.20
"""Weights are not the whole cost: the vision tower's activations, the KV cache
and the allocator's slack add roughly a fifth again on this model at 448 px views."""

DEFAULT_PARAM_COUNT: Final[int] = 8_000_000_000
"""Qwen3-VL-8B-Instruct. Used only to size the guard before the weights exist."""

MODEL_ID: Final[str] = "Qwen/Qwen3-VL-8B-Instruct"

_HF_CACHE_PREFIX: Final[str] = "models--"


class BackendKind(StrEnum):
    """Which serving path answers a ``vlm_*`` step."""

    HF = "hf"
    """``transformers`` on ROCm, bf16. The production and training-parity path."""

    LLAMACPP = "llamacpp"
    """A GGUF quantisation behind llama.cpp's server. The offline demo path."""

    NONE = "none"
    """No backend is reachable; ``vlm_*`` tools are unavailable and answers are
    templated. This is the honest state of a machine with no weights on it."""


class ModelLoadError(RuntimeError):
    """The weights could not be brought up on this machine."""


class BackendUnavailableError(ModelLoadError):
    """No serving backend is configured or reachable.

    Raised at generation time rather than at import time, so a missing model is a
    ``DEGRADED`` step and a templated answer rather than a failed startup.
    """


class VramGuardError(ModelLoadError):
    """Loading these weights would exceed the VRAM budget."""

    def __init__(self, needed_bytes: int, budget_bytes: int, in_use_bytes: int) -> None:
        """Record the arithmetic that refused the load."""
        super().__init__(
            f"loading needs ~{needed_bytes / 1024**3:.1f} GB on top of "
            f"{in_use_bytes / 1024**3:.1f} GB already resident, over the "
            f"{budget_bytes / 1024**3:.1f} GB budget"
        )
        self.needed_bytes = needed_bytes
        self.budget_bytes = budget_bytes
        self.in_use_bytes = in_use_bytes


# --------------------------------------------------------------------- interface


@dataclass(frozen=True)
class PromptImage:
    """One rendered view as the model receives it.

    The label is not decoration. It is the string
    :mod:`satquery.render.view_labels` produced and the string the adaptation
    corpus was built with, and the model's binding of "NDVI heatmap" to a
    physical quantity rests on the two being byte-identical.
    """

    label: str
    rgb: Rgb

    def __post_init__(self) -> None:
        """Reject anything that is not an (H, W, 3) uint8 image."""
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) RGB view, got {self.rgb.shape}")


@dataclass(frozen=True)
class GenerationRequest:
    """One generation call, identical in shape for both backends."""

    system: str
    user: str
    images: tuple[PromptImage, ...] = ()
    max_new_tokens: int = 384
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    stop: tuple[str, ...] = ()

    @property
    def greedy(self) -> bool:
        """True when the request asks for deterministic decoding.

        Every policy-table entry sets ``temperature: 0.0``, because
        AGENT_POLICY_DAG §8 can only promise a reproducible ``answer.text`` if
        decoding is greedy.
        """
        return self.temperature <= 0.0


@dataclass(frozen=True)
class GenerationResult:
    """What one generation produced, and what it cost."""

    text: str
    backend: BackendKind
    model_id: str
    device: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    truncated: bool = False
    """True when generation stopped on the token budget rather than on an EOS.
    A truncated answer may have lost the citation of its last number, so the
    tools record it rather than presenting the fragment as a finished answer."""


@runtime_checkable
class VlmBackend(Protocol):
    """The one interface a serving path implements."""

    kind: BackendKind
    model_id: str

    @property
    def is_loaded(self) -> bool:
        """True when the weights are resident and generation is immediate."""
        ...

    def load(self) -> None:
        """Bring the weights up. Idempotent, and safe to call concurrently."""
        ...

    def unload(self) -> None:
        """Release the weights and the device memory behind them."""
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer one request, loading the weights first if they are not up."""
        ...


# ------------------------------------------------------------------ device / VRAM


def resolve_torch_device(declared: str) -> str:
    """Map a contract device string onto the string ``torch`` understands.

    ROCm builds of PyTorch expose HIP devices through the ``cuda`` API, so the
    registry's ``rocm:0`` is ``cuda:0`` to torch. Keeping the translation in one
    function stops that confusion from being restated in every backend.
    """
    normalised = declared.strip().lower()
    if normalised in {"cpu", ""}:
        return "cpu"
    if normalised in {"auto", "gpu"}:
        return "cuda:0" if torch_cuda_available() else "cpu"
    if normalised.startswith("rocm"):
        _, _, index = normalised.partition(":")
        return f"cuda:{index or '0'}"
    return normalised


def _torch() -> Any | None:
    """Import ``torch`` if it is installed, without making it a hard dependency."""
    try:
        import torch
    except ImportError:
        return None
    return torch


def torch_cuda_available() -> bool:
    """True when a HIP/CUDA device is actually usable from this process."""
    torch = _torch()
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except (RuntimeError, AssertionError):  # pragma: no cover - driver-dependent
        return False


@dataclass(frozen=True)
class VramSnapshot:
    """Device memory as the driver reports it, in bytes."""

    device: str
    total: int = 0
    free: int = 0
    allocated: int = 0

    @property
    def used(self) -> int:
        """Bytes in use on the device, including other processes."""
        return max(0, self.total - self.free)

    @property
    def available(self) -> bool:
        """True when a real device answered."""
        return self.total > 0


def vram_snapshot(device: str = "cuda:0") -> VramSnapshot:
    """Read device memory, returning an empty snapshot when there is no device."""
    torch = _torch()
    if torch is None or device == "cpu" or not torch_cuda_available():
        return VramSnapshot(device=device)
    try:
        free, total = torch.cuda.mem_get_info(device)
        allocated = int(torch.cuda.memory_allocated(device))
    except (RuntimeError, AssertionError, ValueError):  # pragma: no cover - driver
        return VramSnapshot(device=device)
    return VramSnapshot(device=device, total=int(total), free=int(free), allocated=allocated)


def estimate_weight_bytes(
    param_count: int = DEFAULT_PARAM_COUNT, dtype: str = "bfloat16"
) -> int:
    """Bytes an inference load of *param_count* parameters is expected to occupy."""
    per_param = BYTES_PER_PARAM.get(dtype, 2)
    return int(param_count * per_param * ACTIVATION_OVERHEAD)


def guard_vram(
    needed_bytes: int,
    device: str = "cuda:0",
    budget_bytes: int = VRAM_BUDGET_BYTES,
) -> VramSnapshot:
    """Refuse a load that would take the device over budget.

    Checked against what is *already resident on the card*, not against this
    process's allocations: a browser or a second analysis holding 6 GB is just as
    fatal to a 16 GB load as our own tensors would be.

    Raises:
        VramGuardError: The load would exceed *budget_bytes*.
    """
    snapshot = vram_snapshot(device)
    if not snapshot.available:
        # No device to run out of. A CPU load is slow, not unsafe.
        return snapshot
    if snapshot.used + needed_bytes > budget_bytes:
        raise VramGuardError(needed_bytes, budget_bytes, snapshot.used)
    return snapshot


def apply_memory_fraction(device: str, budget_bytes: int = VRAM_BUDGET_BYTES) -> None:
    """Cap this process's share of the device at the budget.

    Belt and braces over :func:`guard_vram`: the guard sizes the load up front,
    while this makes an unexpectedly large allocation *during* generation raise a
    Python exception the executor can record, instead of the allocator quietly
    consuming every last byte and the kernel killing the process.
    """
    torch = _torch()
    if torch is None or device == "cpu" or not torch_cuda_available():
        return
    snapshot = vram_snapshot(device)
    if not snapshot.available:  # pragma: no cover - driver-dependent
        return
    fraction = min(1.0, budget_bytes / snapshot.total)
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    except (RuntimeError, ValueError, TypeError):  # pragma: no cover - torch version
        return


def empty_device_cache(device: str) -> None:
    """Return cached blocks to the driver after an unload."""
    torch = _torch()
    if torch is None or device == "cpu" or not torch_cuda_available():
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:  # pragma: no cover - driver-dependent
        return


# ------------------------------------------------------------------- backend base


@dataclass
class BackendConfig:
    """Everything a backend needs, resolved from settings once.

    Held as a plain record rather than read from :func:`get_settings` inside the
    backends, so a test can construct a backend against a temporary directory
    without touching process-wide state.
    """

    kind: BackendKind | None = None
    """None means "decide for me": :func:`available_backend` picks whichever
    path this machine can actually serve. An explicit :attr:`BackendKind.NONE`
    disables the VLM, which is what a benchmark of the deterministic layer alone
    wants."""

    model_id: str = MODEL_ID
    model_path: Path | None = None
    gguf_path: Path | None = None
    server_url: str | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    budget_bytes: int = VRAM_BUDGET_BYTES
    idle_unload_s: float = 900.0
    request_timeout_s: float = 180.0
    max_new_tokens: int = 384
    attn_implementation: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class LazyBackend:
    """Shared lazy-load, idle-unload and thread-safety for a serving path.

    Both concrete backends inherit this rather than reimplementing the locking.
    Generation is serialised by :attr:`_lock`: the executor already limits GPU
    steps to one at a time, but nothing stops two *requests* reaching the same
    process, and ``transformers`` generation over one model instance is not
    reentrant.
    """

    kind: BackendKind = BackendKind.NONE

    def __init__(self, config: BackendConfig) -> None:
        """Wire the backend to its configuration without loading anything."""
        self.config = config
        self.model_id = config.model_id
        self._lock = threading.RLock()
        self._loaded = False
        self._last_used = 0.0

    # -- lifecycle ---------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True when the weights are resident."""
        return self._loaded

    @property
    def idle_seconds(self) -> float:
        """Seconds since the last generation, or 0 while nothing has run."""
        return 0.0 if not self._last_used else time.monotonic() - self._last_used

    def load(self) -> None:
        """Bring the weights up exactly once, however many callers ask."""
        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True
            self._last_used = time.monotonic()

    def unload(self) -> None:
        """Drop the weights if they are up."""
        with self._lock:
            if not self._loaded:
                return
            self._unload()
            self._loaded = False
            self._last_used = 0.0

    def unload_if_idle(self, idle_s: float | None = None) -> bool:
        """Unload when nothing has generated for *idle_s* seconds.

        Args:
            idle_s: The threshold, honoured literally — ``0.0`` unloads now. When
                omitted, the configured one is used, where ``0`` carries the
                opposite convention and means "keep the weights resident".

        Returns:
            True when this call actually released the weights.
        """
        if idle_s is None:
            threshold = self.config.idle_unload_s
            if threshold <= 0:
                return False
        else:
            threshold = idle_s
        if not self._loaded or self.idle_seconds < threshold:
            return False
        self.unload()
        return True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Answer one request under the model lock, loading first if needed."""
        with self._lock:
            self.load()
            started = time.perf_counter()
            result = self._generate(request)
            self._last_used = time.monotonic()
        duration_ms = int((time.perf_counter() - started) * 1000)
        return (
            result
            if result.duration_ms
            else GenerationResult(
                text=result.text,
                backend=result.backend,
                model_id=result.model_id,
                device=result.device,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                duration_ms=duration_ms,
                truncated=result.truncated,
            )
        )

    # -- subclass hooks ----------------------------------------------------

    def _load(self) -> None:
        """Bring the weights up. Called once, under the lock."""
        raise NotImplementedError

    def _unload(self) -> None:
        """Release the weights. Called once, under the lock."""
        raise NotImplementedError

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one generation. Called under the lock, with the weights up."""
        raise NotImplementedError


# ------------------------------------------------------------- availability probe


def _hf_cache_roots() -> list[Path]:
    """Every directory a Hugging Face snapshot might have been downloaded into."""
    roots: list[Path] = []
    for variable in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    home = os.environ.get("HF_HOME")
    roots.append(Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def snapshot_dir(model_id: str) -> Path | None:
    """Find a downloaded snapshot of *model_id*, without contacting the hub.

    Returns:
        The directory holding the weights, or None when nothing is cached.
    """
    folder = _HF_CACHE_PREFIX + model_id.replace("/", "--")
    for root in _hf_cache_roots():
        snapshots = root / folder / "snapshots"
        if not snapshots.is_dir():
            continue
        for candidate in sorted(snapshots.iterdir()):
            if candidate.is_dir() and any(candidate.glob("config.json")):
                return candidate
    return None


def _weights_present(config: BackendConfig) -> bool:
    """True when the transformers path has something local to load."""
    if config.model_path is not None:
        return (config.model_path / "config.json").is_file()
    return snapshot_dir(config.model_id) is not None


def _transformers_importable() -> bool:
    """True when both halves of the HF path are installed.

    ``importlib.util.find_spec`` rather than an import: this runs at registry
    load time, and importing ``torch`` costs seconds and a chunk of RSS on a
    machine that is never going to use it.
    """
    from importlib.util import find_spec

    try:
        return find_spec("torch") is not None and find_spec("transformers") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def available_backend(config: BackendConfig | None = None) -> BackendKind:
    """Decide which serving path this machine can actually offer, offline.

    ``auto`` prefers ``transformers`` because that is the path the Phase 7 adapter
    is trained and served on; GGUF is the fallback precisely because it is the one
    that still works with the network unplugged and the card busy.
    """
    resolved = config or backend_config()
    if resolved.kind is BackendKind.NONE:
        return BackendKind.NONE
    if resolved.kind is BackendKind.LLAMACPP:
        return BackendKind.LLAMACPP if _llamacpp_configured(resolved) else BackendKind.NONE
    if resolved.kind is BackendKind.HF:
        return (
            BackendKind.HF
            if _transformers_importable() and _weights_present(resolved)
            else BackendKind.NONE
        )

    if _transformers_importable() and _weights_present(resolved):
        return BackendKind.HF
    if _llamacpp_configured(resolved):
        return BackendKind.LLAMACPP
    return BackendKind.NONE


def _llamacpp_configured(config: BackendConfig) -> bool:
    """True when a llama.cpp server address or a local GGUF file is configured."""
    if config.server_url:
        return True
    return config.gguf_path is not None and config.gguf_path.is_file()


# ------------------------------------------------------------------- construction


def backend_config(settings: Settings | None = None) -> BackendConfig:
    """Resolve the backend configuration from application settings."""
    resolved = settings or get_settings()
    return BackendConfig(
        kind=_requested_kind((resolved.vlm_backend or "auto").strip().lower(), resolved),
        model_id=resolved.vlm_model_id,
        model_path=resolved.vlm_model_path,
        gguf_path=resolved.vlm_gguf_path,
        server_url=resolved.vlm_server_url or None,
        device=resolve_torch_device(resolved.vlm_device),
        dtype=resolved.vlm_dtype,
        budget_bytes=int(resolved.vlm_vram_budget_gb * 1024**3),
        idle_unload_s=resolved.vlm_idle_unload_s,
        request_timeout_s=resolved.vlm_request_timeout_s,
        max_new_tokens=resolved.vlm_max_new_tokens,
        attn_implementation=resolved.vlm_attn_implementation or None,
    )


def _requested_kind(requested: str, settings: Settings) -> BackendKind | None:
    """Turn the ``SATQUERY_VLM_BACKEND`` setting into a kind, or None for auto."""
    if settings.vlm_disabled or requested == "none":
        return BackendKind.NONE
    if requested == "hf":
        return BackendKind.HF
    if requested == "llamacpp":
        return BackendKind.LLAMACPP
    return None


_BACKEND: VlmBackend | None = None
_BACKEND_LOCK: Final[threading.Lock] = threading.Lock()


def get_backend(config: BackendConfig | None = None) -> VlmBackend:
    """Return the process-wide backend, constructing it on first use.

    Constructing is cheap — no weights are read until the first
    :meth:`VlmBackend.generate`.

    Raises:
        BackendUnavailableError: No serving path is reachable on this machine.
    """
    global _BACKEND
    resolved = config or backend_config()
    with _BACKEND_LOCK:
        if _BACKEND is not None and config is None:
            return _BACKEND
        kind = available_backend(resolved)
        if kind is BackendKind.HF:
            from satquery.models.hf_backend import HuggingFaceBackend

            backend: VlmBackend = HuggingFaceBackend(resolved)
        elif kind is BackendKind.LLAMACPP:
            from satquery.models.llamacpp_client import LlamaCppBackend

            backend = LlamaCppBackend(resolved)
        else:
            raise BackendUnavailableError(
                "no VLM backend is available: neither a local "
                f"{resolved.model_id} snapshot nor a llama.cpp server is configured"
            )
        if config is None:
            _BACKEND = backend
        return backend


def release_backend() -> None:
    """Unload and forget the process-wide backend.

    Called by the idle sweep, by ``/v1/admin`` style shutdown paths, and by any
    test that constructed a real backend.
    """
    global _BACKEND
    with _BACKEND_LOCK:
        if _BACKEND is not None:
            _BACKEND.unload()
        _BACKEND = None


def set_backend(backend: VlmBackend | None) -> None:
    """Install a backend explicitly. The seam the unit tests inject a fake at."""
    global _BACKEND
    with _BACKEND_LOCK:
        _BACKEND = backend


__all__ = [
    "ACTIVATION_OVERHEAD",
    "BYTES_PER_PARAM",
    "DEFAULT_PARAM_COUNT",
    "MODEL_ID",
    "VRAM_BUDGET_BYTES",
    "BackendConfig",
    "BackendKind",
    "BackendUnavailableError",
    "GenerationRequest",
    "GenerationResult",
    "LazyBackend",
    "ModelLoadError",
    "PromptImage",
    "VlmBackend",
    "VramGuardError",
    "VramSnapshot",
    "apply_memory_fraction",
    "available_backend",
    "backend_config",
    "empty_device_cache",
    "estimate_weight_bytes",
    "get_backend",
    "guard_vram",
    "release_backend",
    "resolve_torch_device",
    "set_backend",
    "snapshot_dir",
    "torch_cuda_available",
    "vram_snapshot",
]
