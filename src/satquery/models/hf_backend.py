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

**The weights are never quantised here, and that is deliberate.** The Phase 7
adapter was trained under NF4, so serving under NF4 would match the training
path — but bitsandbytes 4-bit *generation* is broken on this hardware (gfx1100,
ROCm 6.4, torch 2.9.1): the base weights alone, adapter detached, answer "Name
three primary colours" with a run of close-parens, while the same weights in
bf16 answer correctly. 4-bit training is unaffected, which is why a 19-hour run
completed without anyone noticing. bf16 costs ~16.4 GB of weights and a measured
~18.7 GiB peak with six views attached, inside the 22 GB budget, so there is
room to simply not quantise. Revisit only against a bitsandbytes build that
demonstrably generates.

Set ``SATQUERY_VLM_ADAPTER_PATH`` to serve the fine-tuned model; unset, this
path serves stock Qwen3-VL, which answers in a different format and cites
nothing.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
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
    apply_stop,
    empty_device_cache,
    estimate_weight_bytes,
    guard_vram,
)
from satquery.models.prompts.layout import LAYOUT_VERSION, chat_prompt, layout_fingerprint

LAYOUT_FILE: Final[str] = "layout.json"
"""Written beside the adapter by ``scripts/train_vlm.py``: the layout version and
the processor fingerprint the adapter was trained under. Read back at load."""

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


_CHAT_CONTROL_TOKENS: Final[tuple[str, ...]] = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)
"""Chat and vision scaffolding removed from a decoded answer.

Named explicitly rather than handled by ``skip_special_tokens``, which cannot
tell these from the box sentinels a grounding answer is *made of*. Listing them
means a token absent from this tuple survives into the text, which is the safe
direction: an unexpected marker is visible in the trace, where a silently
deleted box is not.
"""


def _strip_chat_tokens(text: str) -> str:
    """Remove chat and vision scaffolding, leaving the box sentinels intact."""
    for token in _CHAT_CONTROL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


_EOS_TOKENS: Final[tuple[str, ...]] = ("<|im_end|>", "<|endoftext|>")
"""Where a Qwen3-VL turn ends, as token ids rather than text.

Resolved against the tokenizer that ships beside the adapter and handed to
``generate()`` explicitly, so stopping never depends on whichever
``generation_config.json`` the base repository happens to carry. The text-level
stop set in :data:`~satquery.models.loader.CHAT_STOP_STRINGS` stays as the
second line of defence.
"""


def eos_token_ids(processor: Any) -> tuple[int, ...]:
    """The end-of-turn token ids the processor's tokenizer knows, in stop order."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return ()
    ids: list[int] = []
    for token in _EOS_TOKENS:
        try:
            resolved = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # noqa: BLE001 - a tokenizer without the token is not an error
            continue
        if isinstance(resolved, int) and resolved >= 0 and resolved not in ids:
            ids.append(resolved)
    return tuple(ids)


def check_finite_weights(model: Any, source: str) -> int:
    """Refuse to serve weights that carry a NaN or an infinity.

    A single non-finite value anywhere in the network poisons every logit that
    passes through it, and greedy decoding over NaN logits is ``argmax`` over
    NaN — token 0, which Qwen spells ``!``, repeated to the token budget. That
    looks like a model that will not stop talking, and a 500-sample benchmark
    was scored to 0.0 % before anyone looked at the text. The cause was one
    corrupted 4 KB block in a cached safetensors shard, which nothing in the
    load path checks: ``safetensors`` verifies the header, not the payload.

    Every floating-point parameter and buffer is scanned on the device it
    landed on, a few seconds for 8B parameters, before the backend reports
    itself loaded.

    Returns:
        The number of tensors scanned.

    Raises:
        ModelLoadError: At least one tensor holds a NaN or an infinity. The
            message names each one and how to repair the weights.
    """
    import torch

    corrupt: list[str] = []
    scanned = 0
    with torch.inference_mode():
        for name, tensor in itertools.chain(model.named_parameters(), model.named_buffers()):
            if not tensor.is_floating_point():
                continue
            scanned += 1
            finite = torch.isfinite(tensor)
            if not bool(finite.all()):
                bad = int(tensor.numel() - int(finite.sum()))
                corrupt.append(f"{name} ({bad} of {tensor.numel()} values)")
    if corrupt:
        more = f"; +{len(corrupt) - 8} more" if len(corrupt) > 8 else ""
        listed = "; ".join(corrupt[:8]) + more
        raise ModelLoadError(
            f"{source} holds non-finite weights and will only ever generate token 0: "
            f"{listed}. Re-download the weights (for a Hub model: "
            f"`hf download {source} --force-download`; a cached shard is intact when "
            "`sha256sum blobs/<name>` equals its name), or check the adapter's "
            "adapter_model.safetensors if the tensor is a LoRA weight."
        )
    return scanned


def build_messages(request: GenerationRequest) -> list[dict[str, Any]]:
    """Lay the request out as Qwen3-VL chat messages.

    One call into :mod:`satquery.models.prompts.layout`, which is also what the
    training record's prompt is built from — so the label-before-image layout
    the adapter was fitted to is the layout it is served on, byte for byte
    (ML_PIPELINE_RECOVERY_PLAN §2). Nothing here assembles a content list by
    hand, and a test greps to keep it that way.
    """
    return chat_prompt(
        request.system,
        [image.label for image in request.images],
        request.user,
        images=[_to_pil(image) for image in request.images],
    )


def read_layout_record(adapter: Path) -> dict[str, Any] | None:
    """The ``layout.json`` an adapter directory carries, or None when it has none.

    Raises:
        ModelLoadError: The file exists and is not a JSON object.
    """
    path = adapter / LAYOUT_FILE
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelLoadError(f"{path} is unreadable: {error}") from error
    if not isinstance(loaded, dict):
        raise ModelLoadError(f"{path} is not a JSON object")
    return loaded


def check_layout(adapter: Path, processor: Any) -> dict[str, Any]:
    """Refuse to serve an adapter under a layout or pixel budget it was not trained on.

    Level L3 of the parity proof (plan §2.3): the training run writes
    ``layout.json`` — ``LAYOUT_VERSION`` and ``layout_fingerprint`` of the
    processor it trained with — beside the adapter. Here the fingerprint is
    recomputed with the processor this process *actually loaded* and compared.
    A mismatch means the chat template, the layout or the image-token count per
    view differs from training, and the adapter would be reading a token stream
    it never saw. That must look like an error, not like a bad answer — the same
    philosophy as the hard failure on a missing ``adapter_config.json``.

    An adapter with no ``layout.json`` at all is refused too. The one adapter
    that lacks it (``runs/full-epoch-v1``) is precisely the one trained under
    the unlabelled-image layout this module exists to stop serving; every
    adapter ``scripts/train_vlm.py`` writes from now on carries the file. Serve
    the base model deliberately by unsetting ``SATQUERY_VLM_ADAPTER_PATH``.

    Returns:
        The layout record that was verified.

    Raises:
        ModelLoadError: The adapter carries no layout record, or its recorded
            layout version or fingerprint differs from what this processor
            produces.
    """
    record = read_layout_record(adapter)
    if record is None:
        raise ModelLoadError(
            f"{adapter} holds no {LAYOUT_FILE}, so the prompt layout it was trained "
            "under is unknown. Adapters from before the layout fix (runs/full-epoch-v1) "
            "were trained on unlabelled images and must not be served. Unset "
            "SATQUERY_VLM_ADAPTER_PATH to serve the base model deliberately."
        )
    version = record.get("layout_version")
    if version != LAYOUT_VERSION:
        raise ModelLoadError(
            f"{adapter} was trained under prompt layout {version!r}; this build serves "
            f"{LAYOUT_VERSION!r}. Retrain, or serve the adapter from the matching build."
        )
    expected = record.get("layout_fingerprint")
    actual = layout_fingerprint(processor)
    if expected != actual:
        raise ModelLoadError(
            f"{adapter} layout fingerprint {str(expected)[:12]}… does not match the "
            f"loaded processor's {actual[:12]}…: the chat template or the image-token "
            "count per view differs from training (expected "
            f"{record.get('image_tokens_per_view')} tokens per 448 px view). The "
            "processor must come from the adapter directory, not the base repo."
        )
    return record


class HuggingFaceBackend(LazyBackend):
    """``transformers`` serving of Qwen3-VL, with the VRAM guard in front of it."""

    kind = BackendKind.HF

    def __init__(self, config: BackendConfig) -> None:
        """Wire the backend without importing torch or reading any weights."""
        super().__init__(config)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._eos_ids: tuple[int, ...] = ()

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
        model = self._apply_adapter(model)
        model.to(device)
        model.eval()

        self._torch = torch
        self._model = model
        # Corrupt weights must fail here, not answer "!!!!" later. Scanned after
        # the adapter is applied so its tensors are covered too.
        try:
            check_finite_weights(model, self.source)
        except ModelLoadError:
            self._unload()
            raise
        # Bound through an explicitly Any local: transformers is an optional
        # extra, so its symbols are typed on a serving box and Any on one that
        # never installed it. Calling through Any is silent under both, where an
        # inline ignore would be needed on the first and unused on the second.
        auto_processor: Any = transformers.AutoProcessor
        self._processor = auto_processor.from_pretrained(
            self.processor_source, local_files_only=True
        )
        if self.config.adapter_path is not None:
            try:
                check_layout(self.config.adapter_path, self._processor)
            except ModelLoadError:
                self._unload()
                raise
        # The tokenizer that travelled with the adapter says where a turn ends;
        # resolved once, handed to every generate() call.
        self._eos_ids = eos_token_ids(self._processor)

        # The estimate above sized the load; this checks what it actually cost,
        # because a guard that is never reconciled against reality is decoration.
        try:
            guard_vram(0, device=device, budget_bytes=self.config.budget_bytes)
        except ModelLoadError:
            self._unload()
            raise

    @property
    def processor_source(self) -> str:
        """Where the processor is read from: the adapter directory when it has one.

        The training run writes ``processor_config.json``, ``tokenizer_config.json``
        and ``chat_template.jinja`` beside the adapter, and those are the settings
        the adapter was *fitted under* — the pixel budget in particular. The base
        repo's copies can differ, and a processor that tiles an image differently
        from training changes the token stream the adapter learned to read.
        """
        adapter = self.config.adapter_path
        if adapter is None:
            return self.source
        if (adapter / "processor_config.json").is_file():
            return str(adapter)
        # A hard error, not a fallback. The base repo's processor tiles a 448 px
        # view into 196 image tokens; the training profile's max_pixels gives
        # 144. Serving the adapter under the wrong count changes every image
        # position it learned to read, and nothing downstream would notice
        # (plan §2.4).
        raise ModelLoadError(
            f"{adapter} holds no processor_config.json, so the pixel budget the "
            "adapter was trained under is unknown; refusing to fall back to "
            f"{self.source}'s processor. Copy the training run's processor files "
            "beside the adapter (scripts/train_vlm.py writes them) or unset "
            "SATQUERY_VLM_ADAPTER_PATH."
        )

    def _apply_adapter(self, model: Any) -> Any:
        """Wrap the base weights in the Phase 7 LoRA adapter, when one is configured.

        **Applied over bf16 base weights, never over NF4**, even though the
        adapter was trained under NF4. Measured on this box (gfx1100, ROCm 6.4,
        torch 2.9.1): bitsandbytes 4-bit *generation* emits token soup, base
        weights alone with no adapter attached, while the same weights in bf16
        answer correctly. 4-bit training is unaffected, which is why nothing
        caught it during the run. So the serving path stays unquantised: ~16.4 GB
        of weights and a measured ~18.7 GiB peak, inside the 22 GB budget.

        Raises:
            ModelLoadError: An adapter is configured but cannot be applied. A
                silent fallback to the base model would serve stock Qwen3-VL
                under the fine-tuned model's name, and the difference does not
                look like an error — it looks like a bad answer.
        """
        adapter = self.config.adapter_path
        if adapter is None:
            return model
        if not (adapter / "adapter_config.json").is_file():
            raise ModelLoadError(
                f"{adapter} holds no adapter_config.json. Unset "
                "SATQUERY_VLM_ADAPTER_PATH to serve the base model deliberately."
            )
        try:
            from peft import PeftModel
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ModelLoadError(
                f"an adapter is configured at {adapter} but peft is not installed: "
                f"{error}"
            ) from error
        try:
            return PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            raise ModelLoadError(f"could not apply the adapter at {adapter}: {error}") from error

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
        self._eos_ids = ()
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
        tokenizer = getattr(processor, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        started = time.perf_counter()
        with torch.inference_mode():
            produced = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=not request.greedy,
                temperature=None if request.greedy else request.temperature,
                top_p=None if request.greedy else request.top_p,
                # <|im_end|> and <|endoftext|> from the adapter's own tokenizer,
                # not whatever the base repo's generation_config.json lists.
                eos_token_id=list(self._eos_ids) or None,
                pad_token_id=pad_id if isinstance(pad_id, int) else None,
                # transformers matches these against the decoded tail, and needs
                # the tokenizer to do it. Passing the strings without it is
                # silently ignored, which is the same as not stopping at all.
                stop_strings=list(request.stop) or None,
                tokenizer=tokenizer,
            )
        duration_ms = int((time.perf_counter() - started) * 1000)

        new_tokens = produced[0][prompt_tokens:]
        completion_tokens = int(new_tokens.shape[-1])
        # Decoded with the special tokens KEPT, then trimmed by hand. Qwen's box
        # sentinels — <|object_ref_start|>, <|box_start|> and their partners — are
        # special tokens, so skip_special_tokens=True deletes exactly the markup a
        # GROUNDING answer consists of. It leaves "lake(800,750),(1000,1000)",
        # which box_format.parse does still recover, but only through the bare
        # fallback it keeps for foreign checkpoints: the canonical tagged path
        # never matches, and label-to-box association stops being structural and
        # starts being a heuristic over whatever prose precedes the digits. So
        # the scaffolding is stripped by name and everything else is left alone.
        text = apply_stop(
            _strip_chat_tokens(processor.decode(new_tokens, skip_special_tokens=False)),
            request.stop,
        )

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


__all__ = [
    "LAYOUT_FILE",
    "HuggingFaceBackend",
    "build_messages",
    "check_finite_weights",
    "check_layout",
    "eos_token_ids",
    "read_layout_record",
]
