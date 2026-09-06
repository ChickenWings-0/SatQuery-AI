"""Qwen3-VL-8B-Instruct on ROCm via ``transformers``, in bfloat16.

This is the production path and the one the Phase 7 QLoRA adapter is trained
against, so it is also the path that defines what "the model" means for every
number in the ablation table.

``torch`` and ``transformers`` are imported inside :meth:`_load`, never at module
scope. The registry imports every tool module at startup to build the
capabilities panel, and paying two seconds and several hundred megabytes of RSS
to import torch on a machine that has no weights — the CI machine, most
obviously — would be a cost with no benefit.

bfloat16 is not a tuning choice. 8B parameters in fp16 or bf16 is ~16 GB before
activations, fp32 does not fit on a 24 GB card at all, and bf16's exponent range
avoids the overflow fp16 hits in the vision tower's attention on high-dynamic-range
index views.
"""

from __future__ import annotations

import time
from typing import Any, Final

import numpy as np

from satquery.models.loader import (
    BackendConfig,
    BackendKind,
    GenerationRequest,
    GenerationResult,
    LazyBackend,
    ModelLoadError,
    PromptImage,
    apply_memory_fraction,
    empty_device_cache,
    estimate_weight_bytes,
    guard_vram,
)

_MODEL_CLASSES: Final[tuple[str, ...]] = (
    "Qwen3VLForConditionalGeneration",
    "AutoModelForImageTextToText",
)
"""Tried in order. The dedicated class ships with the Qwen3-VL release; the
generic one is the same weights through the auto class on a transformers build
that predates it. Falling back keeps the backend working across the version skew
between a pinned CI environment and the ROCm image."""


def _to_pil(image: PromptImage) -> Any:
    """Convert a rendered view to the PIL image ``transformers`` expects."""
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(image.rgb, dtype=np.uint8), mode="RGB")


def build_messages(request: GenerationRequest) -> list[dict[str, Any]]:
    """Lay the request out as Qwen3-VL chat messages.

    Each view is preceded by its own label as a text part, so the label and the
    pixels are adjacent in the token stream. Naming the images only in the system
    prompt would leave the model to align six unlabelled tensors against a list —
    which it does unreliably, and which silently breaks the moment a view is
    unavailable and the later slots renumber (DATA_ADAPTATION_PLAN §2.4).
    """
    content: list[dict[str, Any]] = []
    for image in request.images:
        content.append({"type": "text", "text": image.label})
        content.append({"type": "image", "image": _to_pil(image)})
    content.append({"type": "text", "text": request.user})
    return [
        {"role": "system", "content": [{"type": "text", "text": request.system}]},
        {"role": "user", "content": content},
    ]


class HuggingFaceBackend(LazyBackend):
    """``transformers`` serving of Qwen3-VL, with the VRAM guard in front of it."""

    kind = BackendKind.HF

    def __init__(self, config: BackendConfig) -> None:
        """Wire the backend without importing torch or reading any weights."""
        super().__init__(config)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def source(self) -> str:
        """What ``from_pretrained`` is pointed at: a local directory or a repo id."""
        return str(self.config.model_path) if self.config.model_path else self.config.model_id

    def _load(self) -> None:
        """Read the weights onto the device, refusing to start if they will not fit.

        Raises:
            ModelLoadError: transformers is missing, or the weights would not fit.
        """
        try:
            import torch
            import transformers
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelLoadError(f"the transformers backend is not installed: {error}") from error

        device = self.config.device
        guard_vram(
            estimate_weight_bytes(dtype=self.config.dtype),
            device=device,
            budget_bytes=self.config.budget_bytes,
        )
        apply_memory_fraction(device, self.config.budget_bytes)

        dtype = getattr(torch, self.config.dtype, torch.bfloat16)
        kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": True}
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        model = self._from_pretrained(transformers, kwargs)
        model.to(device)
        model.eval()

        self._torch = torch
        self._model = model
        # Bound through an explicitly Any local: transformers is an optional
        # extra, so its symbols are typed on a serving box and Any on one that
        # never installed it. Calling through Any is silent under both, where an
        # inline ignore would be needed on the first and unused on the second.
        auto_processor: Any = transformers.AutoProcessor
        self._processor = auto_processor.from_pretrained(self.source, local_files_only=True)

        # The estimate above sized the load; this checks what it actually cost,
        # because a guard that is never reconciled against reality is decoration.
        try:
            guard_vram(0, device=device, budget_bytes=self.config.budget_bytes)
        except ModelLoadError:
            self._unload()
            raise

    def _from_pretrained(self, transformers: Any, kwargs: dict[str, Any]) -> Any:
        """Instantiate the model through whichever class this build provides.

        Raises:
            ModelLoadError: No known class could load these weights.
        """
        errors: list[str] = []
        for class_name in _MODEL_CLASSES:
            factory = getattr(transformers, class_name, None)
            if factory is None:
                continue
            try:
                return factory.from_pretrained(self.source, **kwargs)
            except Exception as error:  # noqa: BLE001 - try the next class
                errors.append(f"{class_name}: {type(error).__name__}: {error}")
        raise ModelLoadError(
            f"could not load {self.source} through any of {', '.join(_MODEL_CLASSES)}"
            + (f" ({'; '.join(errors)})" if errors else "")
        )

    def _unload(self) -> None:
        """Drop the model and give the memory back to the driver."""
        self._model = None
        self._processor = None
        self._torch = None
        empty_device_cache(self.config.device)

    # -- generation --------------------------------------------------------

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Run one greedy (or sampled) generation and decode the new tokens only.

        Raises:
            ModelLoadError: The weights are not resident.
        """
        model, processor, torch = self._model, self._processor, self._torch
        if model is None or processor is None or torch is None:  # pragma: no cover - guarded
            raise ModelLoadError("generate() was called before the weights were loaded")

        if request.greedy:
            # A fixed seed still matters: the processor's image resize and any
            # tie-breaking in the sampler must not vary between two runs of the
            # same trace (AGENT_POLICY_DAG §8).
            torch.manual_seed(request.seed)

        messages = build_messages(request)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.config.device)

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        started = time.perf_counter()
        with torch.inference_mode():
            produced = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=not request.greedy,
                temperature=None if request.greedy else request.temperature,
                top_p=None if request.greedy else request.top_p,
            )
        duration_ms = int((time.perf_counter() - started) * 1000)

        new_tokens = produced[0][prompt_tokens:]
        completion_tokens = int(new_tokens.shape[-1])
        text = processor.decode(new_tokens, skip_special_tokens=True).strip()

        return GenerationResult(
            text=text,
            backend=self.kind,
            model_id=self.model_id,
            device=self.config.device,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=duration_ms,
            truncated=completion_tokens >= request.max_new_tokens,
        )


__all__ = ["HuggingFaceBackend", "build_messages"]
