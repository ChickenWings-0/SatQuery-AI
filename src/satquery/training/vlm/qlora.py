"""QLoRA fine-tuning of Qwen3-VL-8B-Instruct on a 24 GB gfx1100 (DATA_ADAPTATION_PLAN §6).

The recipe in one paragraph: the backbone is frozen and quantised to 4-bit NF4
with double quantisation, LoRA adapters are attached to the language model's
seven linear projections *and* to the last eight blocks of the vision encoder,
and the whole thing is trained at effective batch 16 with gradient checkpointing
under a 22 GB guard.

Two choices carry the result:

* **The vision encoder is adapted, not frozen.** An LLM-only LoRA can learn to
  talk about NDVI without ever learning to *see* it. What is new about our inputs
  is the pixels — a colormapped index field on a fixed domain, a false-colour
  VV/VH/ratio composite — and the ViT is the only part of the network that looks
  at them. The last eight blocks are the ones carrying semantics rather than
  low-level structure, so that is where the capacity goes (§6.1).
* **Every heavy import is deferred.** ``torch``, ``transformers``, ``peft``,
  ``trl`` and ``bitsandbytes`` are imported inside the functions that need them.
  The profile, the target-module resolver, the VRAM arithmetic and the chat
  formatter are pure Python and are what the test suite exercises — running them
  never touches a 16 GB download.

ROCm notes: ``attn_implementation: sdpa``, because flash-attention's CK path is
unreliable on gfx1100; ``HIP_VISIBLE_DEVICES=0`` to hide the 7800X3D iGPU, which
otherwise gets picked as device 0 and fails the run with an unhelpful error.
"""

from __future__ import annotations

import collections
import contextlib
import copy
import hashlib
import json
import os
import random
import re
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

from satquery.models.loader import VRAM_BUDGET_BYTES, VramGuardError, vram_snapshot
from satquery.models.prompts import layout
from satquery.models.prompts.layout import LAYOUT_VERSION, chat_prompt
from satquery.training.corpus_builder import CorpusSample, read_jsonl

DEFAULT_PROFILE_PATH: Final[Path] = Path("configs/train/qlora_qwen3vl8b_rocm24g.yaml")
FALLBACK_PROFILE_PATH: Final[Path] = Path("configs/train/lora_qwen3vl4b_bf16.yaml")
"""Profile A and the Profile B fallback of §6.2. Switching is a config change,
not a rewrite: nothing below is 8B- or 4-bit-specific."""

LLM_TARGET_MODULES: Final[tuple[str, ...]] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
"""Attention and MLP projections in the language model (§6.1)."""

VISION_TARGET_MODULES: Final[tuple[str, ...]] = ("qkv", "proj", "fc1", "fc2")
"""The vision blocks' fused QKV, output projection and MLP (§6.1)."""

VISION_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\.)(?:visual|vision_tower|vision_model|vision_encoder)\."
    r"(?:[\w.]*?\.)?(?:blocks|layers|encoder\.layers)\.(?P<index>\d+)\."
)
"""Locates a vision block's index inside a module path.

Qwen has shipped this subtree as ``visual.blocks.N``, ``vision_tower.…layers.N``
and ``model.visual.blocks.N`` across releases, and a resolver that recognised
only one of them would silently attach zero vision adapters — a run that trains,
converges, and never learns to read a colormap."""

DEFAULT_VISION_BLOCKS: Final[int] = 8
LLM_EXCLUDED_PREFIXES: Final[tuple[str, ...]] = ("visual", "vision_tower", "vision_model")


class ProfileError(RuntimeError):
    """A training profile is missing, malformed, or asks for the impossible."""


# ---------------------------------------------------------------- the profile


class QuantizationSpec(BaseModel):
    """bitsandbytes 4-bit settings (§6.1)."""

    model_config = ConfigDict(extra="forbid")

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class LoraSpec(BaseModel):
    """Adapter shape and where it attaches (§6.1)."""

    model_config = ConfigDict(extra="forbid")

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules_llm: list[str] = Field(default_factory=lambda: list(LLM_TARGET_MODULES))
    target_modules_vision: list[str] = Field(
        default_factory=lambda: list(VISION_TARGET_MODULES)
    )
    vision_blocks: str = f"last_{DEFAULT_VISION_BLOCKS}"

    @property
    def vision_block_count(self) -> int:
        """How many trailing vision blocks to adapt; 0 disables vision adaptation.

        Raises:
            ProfileError: ``vision_blocks`` is neither ``all``, ``none`` nor
                ``last_N``.
        """
        value = self.vision_blocks.strip().lower()
        if value in {"all", "-1"}:
            return -1
        if value in {"none", "0"}:
            return 0
        match = re.fullmatch(r"last_(\d+)", value)
        if match:
            return int(match.group(1))
        raise ProfileError(f"vision_blocks must be 'last_N', 'all' or 'none', got {value!r}")


class DataSpec(BaseModel):
    """Corpus paths and the per-sample budget knobs (§6.1, §6.3)."""

    model_config = ConfigDict(extra="forbid")

    corpus: Path = Path("data/processed/corpus/train.jsonl")
    val: Path = Path("data/processed/corpus/val.jsonl")
    max_views_per_sample: int = 6
    view_size_px: int = 448
    max_pixels: int = 448 * 448
    max_seq_len: int = 4096
    expected_sources: dict[str, int] = Field(default_factory=dict)
    """Minimum *train* sample count per corpus source, e.g. ``{bigearthnet_v2: 8000}``.

    The trainer verifies its own diet: :func:`guard_composition` counts the
    JSONL it is about to train on — the lines, not the builder's report — and
    refuses to start if any source is below its floor. The previous corpus lost
    VRSBench, CDVQA and RSVQA-HR entirely and nothing noticed until the run was
    over (ML_PIPELINE_RECOVERY_PLAN fact 9, §5.3). Empty means no expectation."""
    eval_samples: int | None = 256
    """How many validation samples an eval pass reads; None reads all of them.

    Bounded, because an eval is not free and is not proportional to the training
    it reports on. The full POC val split is 3,915 samples, which at ~9.5 s per
    batch is 78 minutes — after a 10-step run that trained for six. Left
    unbounded that cost lands again at every ``eval_steps`` boundary and once
    more at the end, which is how a run that was supposed to fit in an evening
    does not. A few hundred samples is enough to watch a loss curve; the honest
    number for a checkpoint comes from the Phase 8 eval, not from here."""


class TrainSpec(BaseModel):
    """Optimiser, schedule and checkpointing (§6.1)."""

    model_config = ConfigDict(extra="forbid")

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 1.0
    bf16: bool = True
    seed: int = 42
    dataloader_num_workers: int = 8
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int = 3
    eval_steps: int = 250
    logging_steps: int = 10
    resume_from_checkpoint: str = "auto"

    @property
    def effective_batch(self) -> int:
        """Samples per optimiser step."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class QLoraProfile(BaseModel):
    """One complete training profile — the YAML of §6.1 or §6.2, validated."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = "sq-lora-v1"
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    attn_implementation: str = "sdpa"
    torch_dtype: str = "bfloat16"
    quantization: QuantizationSpec = Field(default_factory=QuantizationSpec)
    lora: LoraSpec = Field(default_factory=LoraSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    train: TrainSpec = Field(default_factory=TrainSpec)
    env: dict[str, str] = Field(
        default_factory=lambda: {
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "0",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        }
    )

    def apply_env(self) -> dict[str, str]:
        """Export the profile's environment, without overriding an explicit one.

        ``HIP_VISIBLE_DEVICES`` is the load-bearing entry: on this box the
        7800X3D's iGPU enumerates alongside the 7900 XTX, and a run that lands on
        it fails minutes in with an allocation error that names neither device.
        """
        applied: dict[str, str] = {}
        for key, value in self.env.items():
            applied[key] = os.environ.setdefault(key, str(value))
        return applied


def load_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> QLoraProfile:
    """Load and validate a training profile from YAML.

    Raises:
        ProfileError: The file is absent or is not a YAML mapping.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise ProfileError(f"training profile {resolved} does not exist")
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ProfileError(f"training profile {resolved} is not a YAML mapping")
    return QLoraProfile.model_validate(dict(loaded))


# ------------------------------------------------------------ target modules


def _module_names(model: Any) -> list[str]:
    """Every named module of a torch model, or the sequence given directly.

    Accepting a plain sequence of names is what lets the resolver be tested
    against a recorded Qwen3-VL module tree without loading 8B parameters.
    """
    if hasattr(model, "named_modules"):
        return [name for name, _ in model.named_modules() if name]
    return [str(name) for name in model]


def vision_block_index(name: str) -> int | None:
    """The vision-block index a module path sits in, or None if it is not in one."""
    match = VISION_BLOCK_PATTERN.search(name)
    return int(match.group("index")) if match else None


def select_vision_blocks(
    names: Iterable[str], last_n: int = DEFAULT_VISION_BLOCKS
) -> set[int]:
    """The block indices to adapt: the *last* ``last_n`` present in the model.

    ``last_n=-1`` selects every block, ``0`` none. Counting from the end rather
    than assuming a depth keeps the profile valid across the 4B fallback, whose
    encoder has a different number of blocks.
    """
    indices = sorted({index for name in names if (index := vision_block_index(name)) is not None})
    if last_n < 0:
        return set(indices)
    if last_n == 0 or not indices:
        return set()
    return set(indices[-last_n:])


def _is_llm_target(name: str, leaves: frozenset[str]) -> bool:
    """True for a language-model projection outside the vision tower."""
    if vision_block_index(name) is not None:
        return False
    if name.split(".")[0] in LLM_EXCLUDED_PREFIXES:
        return False
    return name.rsplit(".", 1)[-1] in leaves


def resolve_target_modules(model: Any, spec: LoraSpec | None = None) -> list[str]:
    """The full module paths LoRA attaches to, in a stable order.

    Full paths rather than bare leaf names, because ``qkv`` appears in every
    vision block and only the last eight are in scope. PEFT matches a target
    either exactly or as a dotted suffix, so a full path is the precise form.

    Args:
        model: A torch module, or an iterable of module names.
        spec: The LoRA spec; defaults select the §6.1 module set.

    Raises:
        ProfileError: The model exposes none of the requested modules, which
            means the checkpoint's layout changed and the run would train
            nothing.
    """
    resolved = spec or LoraSpec()
    names = _module_names(model)
    llm_leaves = frozenset(resolved.target_modules_llm)
    vision_leaves = frozenset(resolved.target_modules_vision)
    blocks = select_vision_blocks(names, resolved.vision_block_count)

    targets = [
        name
        for name in names
        if _is_llm_target(name, llm_leaves)
        or (
            (index := vision_block_index(name)) is not None
            and index in blocks
            and name.rsplit(".", 1)[-1] in vision_leaves
        )
    ]
    if not targets:
        raise ProfileError(
            "no LoRA target modules matched this checkpoint; the module layout "
            f"changed (looked for {sorted(llm_leaves | vision_leaves)})"
        )
    return targets


def split_targets(targets: Sequence[str]) -> tuple[list[str], list[str]]:
    """Partition resolved targets into (language model, vision encoder)."""
    vision = [name for name in targets if vision_block_index(name) is not None]
    return [name for name in targets if vision_block_index(name) is None], vision


# ------------------------------------------------------------- the VRAM guard


class ModelShape(BaseModel):
    """The dimensions the footprint arithmetic needs, defaulted to Qwen3-VL-8B."""

    model_config = ConfigDict(extra="forbid")

    param_count: int = 8_000_000_000
    hidden_size: int = 4096
    intermediate_size: int = 12_288
    num_layers: int = 36
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_blocks: int = 27
    vision_patch_px: int = 16
    vision_heads: int = 16
    vocab_size: int = 151_936


KNOWN_SHAPES: Final[dict[str, ModelShape]] = {
    "Qwen/Qwen3-VL-8B-Instruct": ModelShape(),
    "Qwen/Qwen3-VL-4B-Instruct": ModelShape(
        param_count=4_000_000_000,
        hidden_size=2560,
        intermediate_size=9728,
        num_layers=36,
        vision_hidden_size=1152,
        vision_intermediate_size=4304,
        vision_blocks=27,
    ),
}
"""Shapes for the two profiles of §6.1-§6.2, so the footprint estimate means
something for both. An unknown model falls back to the 8B shape, which
over-estimates a smaller one — the safe direction for a guard."""


def shape_for(base_model: str) -> ModelShape:
    """The dimensions to estimate against for a base model id."""
    return KNOWN_SHAPES.get(base_model, ModelShape())


SAFE_PEAK_BYTES: Final[int] = 19 * 1024**3
"""The peak a profile should plan for, below the 22 GiB the guard refuses at.

The gap is the margin for what this estimate cannot see: sequence length varies
across the corpus, and the profile's ``max_seq_len`` is a cap rather than a
typical value, so the longest samples in 65,000 will run hotter than any
100-sample rehearsal. A profile that estimates at 21.5 GiB is not "inside the
budget", it is one long batch away from an OOM eleven hours into a run."""


class MeasuredPeak(BaseModel):
    """One observed peak, and the configuration that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    base_model: str
    batch: int
    max_seq_len: int
    views: int
    max_pixels: int
    peak_bytes: int


MEASURED_PEAKS: Final[tuple[MeasuredPeak, ...]] = (
    MeasuredPeak(
        label="sanity check, 100 samples, 6/6 steps, RX 7900 XTX",
        base_model="Qwen/Qwen3-VL-8B-Instruct",
        batch=1,
        max_seq_len=4096,
        views=6,
        max_pixels=448 * 448,
        peak_bytes=int(21.49 * 1024**3),
    ),
)
"""Every peak this estimator has been calibrated against.

One point, so far. :data:`_VISION_ATTENTION_BUFFERS` is fitted to reproduce it
and everything else is derived, which means the estimate's *shape* is trustworthy
and its absolute value is trustworthy only near this configuration. Add a row
here whenever a run reports a peak, and re-fit — a second point at a different
view count would separate the quadratic term from the constant overhead, which
one point cannot."""


class VramEstimate(BaseModel):
    """Predicted peak device memory for one training step (§6.1)."""

    model_config = ConfigDict(extra="forbid")

    weight_bytes: int
    adapter_params: int
    adapter_bytes: int
    optimiser_bytes: int
    activation_bytes: int
    vision_bytes: int = 0
    total_bytes: int = 0
    budget_bytes: int = VRAM_BUDGET_BYTES
    safe_bytes: int = SAFE_PEAK_BYTES

    @property
    def fits(self) -> bool:
        """True when the predicted peak is inside the guard's budget."""
        return self.total_bytes <= self.budget_bytes

    @property
    def within_margin(self) -> bool:
        """True when the peak also leaves room for the corpus's longest samples.

        ``fits`` is the refusal threshold; this is the one worth designing to.
        A profile that only satisfies the former will run a rehearsal and die on
        a long batch hours into the real thing.
        """
        return self.total_bytes <= self.safe_bytes

    @property
    def total_gib(self) -> float:
        """The predicted peak, in GiB, for a log line."""
        return self.total_bytes / 1024**3


_BYTES_PER_PARAM_4BIT: Final[float] = 0.5 + 0.03
"""NF4 packs two parameters per byte; double quantisation adds ~0.03 B/param of
quantisation scales, which is small but not nothing at 8B."""

_BYTES_PER_PARAM_BF16: Final[int] = 2
_ADAPTER_BYTES_PER_PARAM: Final[int] = 2
_OPTIMISER_BYTES_PER_PARAM: Final[int] = 2
"""``paged_adamw_8bit`` holds two 8-bit moments per trainable parameter."""

_LOGIT_COPIES: Final[int] = 4
"""Bytes per token per vocabulary entry held at the loss: bf16 logits plus their
gradient. The float32 the cross-entropy upcasts to is not counted because recent
transformers chunks it. The logits are still the single largest activation in a
long-context VLM step over a 150k vocabulary, and omitting them is how a memory
estimate comes out several times under."""

_CHECKPOINT_RECOMPUTE_BLOCKS: Final[int] = 4
"""With gradient checkpointing only block boundaries survive the forward pass;
the peak inside a recomputed block is a few MLP-width tensors."""

_VISION_ATTENTION_BUFFERS: Final[float] = 10.5
"""Fitted. See :data:`MEASURED_PEAKS`.

The vision tower's attention is quadratic in *patches*, and six 448 px views is
4,704 patches in one sequence — two orders of magnitude more than the 1,176
tokens they become after merging into the language model. That quadratic is what
the first version of this estimator missed entirely, and it is the single largest
term in the real footprint. This constant absorbs the head count, the forward
and backward buffers in flight, and whatever SDPA does not fuse; only its *value*
is fitted, the ``patches²`` scaling it multiplies is not."""

PEAK_OVERHEAD: Final[float] = 1.35
"""Allocator fragmentation and the copies the processor makes."""

def estimate_adapter_params(spec: LoraSpec, shape: ModelShape, vision_blocks: int) -> int:
    """Trainable parameter count for the adapters this spec attaches.

    A LoRA on an ``m x n`` matrix adds ``r * (m + n)`` parameters. Summing the
    targeted matrices is exact given the shapes, and lands on the ~50 M of §6.1
    for the 8B profile.
    """
    llm = 0
    per_square = spec.r * 2 * shape.hidden_size
    for leaf in spec.target_modules_llm:
        if leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
            llm += per_square
        else:
            llm += spec.r * (shape.hidden_size + shape.intermediate_size)
    llm *= shape.num_layers

    vision = 0
    for leaf in spec.target_modules_vision:
        if leaf == "qkv":
            vision += spec.r * 4 * shape.vision_hidden_size
        elif leaf == "proj":
            vision += spec.r * 2 * shape.vision_hidden_size
        else:
            vision += spec.r * (shape.vision_hidden_size + shape.vision_intermediate_size)
    vision *= max(0, vision_blocks)
    return llm + vision


def estimate_footprint(
    profile: QLoraProfile,
    shape: ModelShape | None = None,
    tokens: int | None = None,
) -> VramEstimate:
    """Predict peak VRAM for this profile, before anything is allocated.

    The components: the quantised backbone (with the embedding and output head
    left in bf16), the adapters with their gradients and 8-bit moments, the
    language model's retained activations and logits, the **vision tower's
    quadratic attention over raw patches**, and a fragmentation multiplier.

    Calibrated against :data:`MEASURED_PEAKS` — presently one point, the 21.49
    GiB sanity check. Its first version predicted 12.0 GiB for that run, an
    understatement of 9.5 GiB, because it modelled the language model alone and
    ignored both ``max_views_per_sample`` and ``max_pixels``. Six 448 px views
    are 4,704 patches of quadratic attention, and that term dominates.

    It remains an envelope, not a measurement — the authoritative number is the
    peak ``scripts/train_vlm.py --sanity-check`` reports off the card. What it is
    for is refusing a profile that cannot fit *before* the eleven-hour run, and
    showing which §6.3 knob to turn when it does not.
    """
    dimensions = shape or shape_for(profile.base_model)
    vision_blocks = profile.lora.vision_block_count
    if vision_blocks < 0:
        vision_blocks = dimensions.vision_blocks
    vision_blocks = min(vision_blocks, dimensions.vision_blocks)

    if profile.quantization.load_in_4bit:
        # The embedding and the output head stay in bf16: quantising a 150k-row
        # vocabulary matrix costs accuracy where it is least affordable.
        head_bytes = dimensions.vocab_size * dimensions.hidden_size * _BYTES_PER_PARAM_BF16
        quantised = dimensions.param_count - dimensions.vocab_size * dimensions.hidden_size
        weight_bytes = int(quantised * _BYTES_PER_PARAM_4BIT + head_bytes)
    else:
        weight_bytes = dimensions.param_count * _BYTES_PER_PARAM_BF16

    adapter_params = estimate_adapter_params(profile.lora, dimensions, vision_blocks)
    adapter_bytes = adapter_params * _ADAPTER_BYTES_PER_PARAM
    optimiser_bytes = adapter_params * _OPTIMISER_BYTES_PER_PARAM

    sequence = tokens if tokens is not None else profile.data.max_seq_len
    batch = profile.train.per_device_train_batch_size

    # The vision tower, which the first version of this estimator omitted. Its
    # attention runs over raw patches, before the 2x2 merge that makes them
    # language-model tokens, so it is quadratic in a number several times larger
    # than `max_seq_len` — and it scales with the two knobs §6.3 says to turn
    # first, neither of which changed the estimate at all until now.
    patches_per_view = max(1, profile.data.max_pixels // (dimensions.vision_patch_px ** 2))
    patches = profile.data.max_views_per_sample * patches_per_view
    vision_attention = (
        patches
        * patches
        * _BYTES_PER_PARAM_BF16
        * dimensions.vision_heads
        * _VISION_ATTENTION_BUFFERS
    )
    vision_states = (
        patches * dimensions.vision_hidden_size * _BYTES_PER_PARAM_BF16 * max(1, vision_blocks)
    )
    vision_bytes = int((vision_attention + vision_states) * batch)

    boundaries = (
        sequence * dimensions.hidden_size * _BYTES_PER_PARAM_BF16 * dimensions.num_layers
        if profile.train.gradient_checkpointing
        else sequence * dimensions.hidden_size * _BYTES_PER_PARAM_BF16 * dimensions.num_layers * 4
    )
    recompute = (
        sequence
        * dimensions.intermediate_size
        * _BYTES_PER_PARAM_BF16
        * _CHECKPOINT_RECOMPUTE_BLOCKS
    )
    logits = sequence * dimensions.vocab_size * _LOGIT_COPIES
    activation_bytes = int((boundaries + recompute + logits) * batch)

    total = (
        weight_bytes
        + 2 * adapter_bytes
        + optimiser_bytes
        + activation_bytes
        + vision_bytes
    ) * PEAK_OVERHEAD
    return VramEstimate(
        weight_bytes=weight_bytes,
        adapter_params=adapter_params,
        adapter_bytes=adapter_bytes,
        optimiser_bytes=optimiser_bytes,
        activation_bytes=activation_bytes,
        vision_bytes=vision_bytes,
        total_bytes=int(total),
    )


def guard_budget(
    estimate: VramEstimate, device: str = "cuda:0", budget_bytes: int = VRAM_BUDGET_BYTES
) -> VramEstimate:
    """Refuse a run whose predicted peak exceeds the 22 GB budget.

    Checked against what is already resident on the card, exactly as the serving
    guard is: a browser holding 4 GB fails an 18 GB run just as surely as a bad
    profile would.

    Raises:
        VramGuardError: The run does not fit.
    """
    snapshot = vram_snapshot(device)
    in_use = snapshot.used if snapshot.available else 0
    if estimate.total_bytes + in_use > budget_bytes:
        raise VramGuardError(estimate.total_bytes, budget_bytes, in_use)
    return estimate


# ------------------------------------------------------------- data plumbing


COMPLETION_SUFFIX: Final[str] = "<|im_end|>\n"
"""What the chat template appends to the assistant text. The completion trl
tokenises is ``apply_chat_template(prompt + completion)[len(prompt):]`` —
``"{answer}<|im_end|>\\n"`` — so the loss covers the answer *and* the
end-of-turn token. Training the EOS is deliberate: the runaway decode
``loader.CHAT_STOP_STRINGS`` guards against is what an untrained stop token
produces."""

STRUCTURAL_TAIL_TOKENS: Final[int] = 2
"""``<|im_end|>`` and the trailing newline: the two completion positions that
carry no answer. Excluded from ``answer_token_accuracy`` (plan §3.6), because a
one-token answer that is wrong still scores 67 % once the model has learned to
stop."""


def completion_text(sample_assistant: str) -> str:
    """The exact string the collator tokenises as the completion."""
    return f"{sample_assistant}{COMPLETION_SUFFIX}"


def sample_to_chat(
    sample: CorpusSample,
    image_root: Path | None = None,
    max_views: int = 6,
) -> dict[str, Any]:
    """Render one corpus sample as the **prompt-completion** record ``trl`` consumes.

    ``{"prompt": [system, user], "completion": [assistant], "images": [...]}``
    — not ``messages``. The shape is the masking mechanism (ML_PIPELINE_RECOVERY_PLAN
    §1): with ``completion_only_loss=True`` trl's vision collator tokenises the
    prompt and the completion separately, concatenates, and sets every prompt
    label to −100. The previous run used ``messages`` and trained on all ~2,000
    prompt tokens per sample; the 3–100 answer tokens were ~1 % of the gradient.
    The alternatives the docs used to recommend are dead on the pinned stack:
    ``DataCollatorForCompletionOnlyLM`` no longer exists and
    ``assistant_only_loss`` raises for any dataset with an ``images`` key.

    Two trl contracts still carried, both load-bearing:

    * **``images`` is a top-level key.** It is the only thing that makes
      ``SFTTrainer`` treat the dataset as vision and select the vision collator.
      A record whose images live only in its message blocks is tokenised as
      text and dies with a bare ``StopIteration`` deep in the processor.
    * **Image blocks are bare placeholders.** ``prepare_multimodal_messages``
      counts blocks of type ``image`` with no ``image`` key and refuses the
      example unless that count equals ``len(images)``.

    The user turn's layout — label as a text part immediately before each
    view's pixels, question last — comes from :func:`~satquery.models.prompts.layout.chat_prompt`,
    the same call the serving backend makes. The old ``label`` key on the image
    block was never printed by the chat template; the adapter trained on
    unlabelled images and was served on labelled ones (plan §2).
    """
    views = sample.views[:max_views]
    return {
        "prompt": chat_prompt(sample.system, [view.label for view in views], sample.user),
        "completion": [
            {"role": "assistant", "content": [{"type": "text", "text": sample.assistant}]}
        ],
        "images": [
            str(image_root / view.path) if image_root else view.path for view in views
        ],
    }


def image_placeholder_count(record: Mapping[str, Any]) -> int:
    """Count the placeholders ``trl`` will try to fill in one record's *prompt*.

    Exactly ``trl``'s own rule — a block of type ``image`` carrying no ``image``
    payload — so a record can be checked against its own ``images`` list before
    it reaches the trainer. Counted over ``prompt``: a record that regressed to
    the ``messages`` shape counts zero here and fails the equality with
    ``len(images)`` instead of silently training on the whole sequence.
    """
    return layout.image_placeholder_count(record.get("prompt") or ())


def record_completion_text(record: Mapping[str, Any]) -> str:
    """The assistant text a prompt-completion record supervises."""
    completion = record.get("completion") or ()
    for message in completion:
        for part in message.get("content") or ():
            if part.get("type") == "text" and part.get("text") is not None:
                return str(part["text"])
    raise ProfileError("record carries no assistant text in its completion")


def stratum_of(sample: CorpusSample) -> tuple[str, str]:
    """The ``(source, task)`` cell a sample belongs to, for stratified draws."""
    return (str(sample.source.value), str(sample.task.value))


def stratified_subset(
    samples: Sequence[CorpusSample],
    n: int,
    seed: int = 0,
    key: Callable[[CorpusSample], Any] = stratum_of,
) -> list[CorpusSample]:
    """A seeded subset of *n* samples spread as evenly as possible over strata.

    Round-robin over the strata in a fixed order, each stratum shuffled by the
    seed, until *n* are drawn or every stratum is exhausted. ``--sanity-check``
    used to take the first 100 lines, which in a source-ordered corpus is one
    source and one task (plan §7.6); the Stage A memorisation probe draws its
    64 the same way with a task-aware key.
    """
    if n <= 0 or not samples:
        return []
    rng = random.Random(seed)
    buckets: dict[Any, list[CorpusSample]] = collections.defaultdict(list)
    for sample in samples:
        buckets[key(sample)].append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    order = sorted(buckets, key=str)
    chosen: list[CorpusSample] = []
    cursor = 0
    while len(chosen) < n and any(buckets.values()):
        bucket = buckets[order[cursor % len(order)]]
        if bucket:
            chosen.append(bucket.pop())
        cursor += 1
    return chosen


def load_corpus(
    path: Path,
    limit: int | None = None,
    max_views: int = 6,
    image_root: Path | None = None,
    stratify_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Read a corpus JSONL into prompt-completion records.

    Args:
        path: ``train.jsonl`` or ``val.jsonl``.
        limit: Keep this many samples — how ``--sanity-check`` gets its 100-item
            run without a second corpus file.
        max_views: Per-sample image cap, from the profile.
        image_root: Prefix for the corpus's relative view paths, when the
            pre-rendered views do not live under the working directory.
        stratify_seed: When given with *limit*, the subset is drawn by
            :func:`stratified_subset` over ``(source, task)`` rather than as the
            file's first *limit* lines.

    Raises:
        ProfileError: The corpus file does not exist.
    """
    if not path.is_file():
        raise ProfileError(f"corpus {path} does not exist; run scripts/build_corpus first")
    samples = read_jsonl(path)
    if limit is not None:
        if stratify_seed is not None:
            samples = stratified_subset(samples, limit, seed=stratify_seed)
        else:
            samples = samples[:limit]
    return [
        sample_to_chat(sample, image_root=image_root, max_views=max_views)
        for sample in samples
    ]


def corpus_composition(path: Path) -> dict[str, int]:
    """Train sample count per source, read from the JSONL lines themselves.

    From the file the trainer is about to read, not from the builder's report:
    a corpus swapped under the profile is caught here, at start-up.
    """
    counts: dict[str, int] = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line).get("source")
            counts[str(source)] += 1
    return dict(counts)


def guard_composition(path: Path, expected: Mapping[str, int]) -> dict[str, int]:
    """Refuse a corpus in which any expected source is below its floor (plan §5.3).

    Raises:
        ProfileError: A source named in *expected* has fewer train samples than
            required, or none at all.
    """
    counts = corpus_composition(path)
    short = {
        source: (counts.get(source, 0), floor)
        for source, floor in expected.items()
        if counts.get(source, 0) < floor
    }
    if short:
        table = ", ".join(f"{name}: {have} < {need}" for name, (have, need) in short.items())
        raise ProfileError(
            f"{path} does not contain the sources the profile expects ({table}). "
            f"Present: {counts}. Rebuild with every source named and "
            "--on-missing-views fail, or lower data.expected_sources deliberately."
        )
    return counts


def missing_images(records: Sequence[Mapping[str, Any]], limit: int | None = None) -> list[str]:
    """Referenced image paths that are not on disk, from a chat-formatted corpus.

    The ``images`` column decodes lazily — that is what keeps a 65 k-sample
    corpus out of RAM — so a path that points at nothing is not an error until
    the batch containing it is collated, which can be hours into a run. Checking
    here costs one ``stat`` per view and turns that into a message before the
    weights are loaded.

    Args:
        records: Chat records from :func:`load_corpus`.
        limit: Stop after inspecting this many records; None checks all of them.

    Returns:
        The absent paths, in the order they were referenced.
    """
    absent: list[str] = []
    for record in list(records)[:limit]:
        for image in record.get("images", ()):
            if isinstance(image, str | Path) and not Path(image).is_file():
                absent.append(str(image))
    return absent


def guard_images_present(
    records: Sequence[Mapping[str, Any]], label: str, sample_size: int = 200
) -> None:
    """Refuse to start a run whose corpus points at views nobody rendered.

    Checks a prefix rather than the whole corpus: a corpus is built by one pass
    over one render tree, so missing views are a property of the build, not of
    scattered individual lines, and the first few hundred records answer the
    question. A clean sample is not a proof — it is the cheap check that catches
    the failure that actually happens.

    Raises:
        ProfileError: A referenced view file is absent.
    """
    absent = missing_images(records, limit=sample_size)
    if not absent:
        return
    checked = min(len(records), sample_size)
    raise ProfileError(
        f"{label}: {len(absent)} referenced view file(s) are missing in the first "
        f"{checked} samples, e.g. {absent[:3]}. The corpus was built over patches "
        "the render pass has not covered. Rebuild it restricted to what exists:\n"
        "  uv run python scripts/build_corpus.py --on-missing-views skip ...\n"
        "or render the missing patches first (scripts/render_views.py)."
    )


def iter_corpus(path: Path) -> Iterator[CorpusSample]:
    """Stream a corpus file one validated sample at a time."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield CorpusSample.model_validate_json(line)


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str | None:
    """The checkout's HEAD commit, or None outside a repository."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


class RunPlan(BaseModel):
    """The arithmetic of one run, resolved before anything is loaded (§7.1)."""

    model_config = ConfigDict(extra="forbid")

    samples: int
    epochs: float
    effective_batch: int
    steps: int
    sanity_check: bool = False

    def describe(self) -> str:
        """One line for the log."""
        mode = "sanity check" if self.sanity_check else "run"
        return (
            f"{mode}: {self.samples} samples x {self.epochs:g} epoch(s) at "
            f"effective batch {self.effective_batch} = {self.steps} optimiser steps"
        )


SANITY_SAMPLES: Final[int] = 100
"""``--sanity-check`` truncates to this many samples for one epoch. Enough to
walk the full path — dataloader, processor, 4-bit forward, LoRA backward,
optimiser step, checkpoint write — in minutes rather than hours, which is the
only cheap way to find out whether ROCm and bitsandbytes agree today."""


def plan_run(profile: QLoraProfile, corpus_size: int, sanity_check: bool = False) -> RunPlan:
    """Resolve samples, epochs and optimiser steps for this run."""
    samples = min(corpus_size, SANITY_SAMPLES) if sanity_check else corpus_size
    epochs = 1.0 if sanity_check else float(profile.train.num_train_epochs)
    batch = profile.train.effective_batch
    steps = max(1, int(samples * epochs // batch)) if samples else 0
    return RunPlan(
        samples=samples,
        epochs=epochs,
        effective_batch=batch,
        steps=steps,
        sanity_check=sanity_check,
    )


# ------------------------------------------------------ the pre-flight guards


TOKEN_BUDGET_MARGIN: Final[int] = 8
"""Slack the token-budget guard keeps: the assistant header and any template
whitespace the text-only count does not see."""

MAX_COMPLETION_TOKENS: Final[int] = 512
"""The longest completion the mask audit accepts. Answers are 3–100 tokens; a
completion past this is a corpus bug, not a long answer."""

MASK_AUDIT_SAMPLES: Final[int] = 32


class TokenBudgetOffender(BaseModel):
    """One sample whose prompt and completion together overrun the context."""

    model_config = ConfigDict(extra="forbid")

    index: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenBudgetReport(BaseModel):
    """What :func:`token_budget_report` measured over a corpus (plan §1.5 G3)."""

    model_config = ConfigDict(extra="forbid")

    checked: int
    max_seq_len: int
    tokens_per_view: int
    margin: int
    prompt_min: int
    prompt_max: int
    completion_min: int
    completion_max: int
    offenders: list[TokenBudgetOffender]

    @property
    def ok(self) -> bool:
        """True when nothing would be truncated."""
        return not self.offenders


def _tokenise(tokenizer: Any, text: str) -> list[int]:
    """Token ids for *text* the way the collator tokenises: no special tokens added."""
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return list(ids)


def prompt_token_count(
    record: Mapping[str, Any], tokenizer: Any, tokens_per_view: int
) -> int:
    """Tokens the prompt half of a record occupies, images counted by arithmetic.

    The chat template is rendered with bare placeholders (no pixels), so each
    image lands as a single ``<|image_pad|>`` in the text; that one token is
    then replaced by *tokens_per_view*, which is what the processor would
    expand it to under the training pixel budget.
    """
    text = tokenizer.apply_chat_template(
        list(record["prompt"]), tokenize=False, add_generation_prompt=True
    )
    ids = _tokenise(tokenizer, str(text))
    pad = layout.image_token_id(tokenizer)
    placeholders = sum(1 for token in ids if token == pad)
    return len(ids) - placeholders + placeholders * tokens_per_view


def completion_token_count(record: Mapping[str, Any], tokenizer: Any) -> int:
    """Tokens the completion occupies: the answer plus the end-of-turn suffix."""
    return len(_tokenise(tokenizer, completion_text(record_completion_text(record))))


def token_budget_report(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    tokens_per_view: int,
    margin: int = TOKEN_BUDGET_MARGIN,
) -> TokenBudgetReport:
    """Measure every record's token footprint against the context length.

    Text only — no pixels are decoded — so it costs about a millisecond per
    sample and can run over the whole corpus before the weights load. Truncation
    in the collator is applied after concatenation and from the right, so an
    over-long sample loses its *completion* first: 100 % masked, a finite zero
    loss, one wasted forward/backward, and nothing in the log (plan fact 11).
    """
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    offenders: list[TokenBudgetOffender] = []
    for index, record in enumerate(records):
        prompt = prompt_token_count(record, tokenizer, tokens_per_view)
        completion = completion_token_count(record, tokenizer)
        prompt_lengths.append(prompt)
        completion_lengths.append(completion)
        total = prompt + completion + margin
        if total > max_seq_len:
            offenders.append(
                TokenBudgetOffender(
                    index=index,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                )
            )
    return TokenBudgetReport(
        checked=len(records),
        max_seq_len=max_seq_len,
        tokens_per_view=tokens_per_view,
        margin=margin,
        prompt_min=min(prompt_lengths, default=0),
        prompt_max=max(prompt_lengths, default=0),
        completion_min=min(completion_lengths, default=0),
        completion_max=max(completion_lengths, default=0),
        offenders=offenders,
    )


def guard_token_budget(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_seq_len: int,
    tokens_per_view: int,
    label: str = "corpus",
    margin: int = TOKEN_BUDGET_MARGIN,
) -> TokenBudgetReport:
    """G3 — refuse to start if any sample would lose its completion to truncation.

    Raises:
        ProfileError: At least one record overruns ``max_seq_len``; the message
            lists the first offenders by index.
    """
    report = token_budget_report(records, tokenizer, max_seq_len, tokens_per_view, margin)
    if report.ok:
        return report
    shown = ", ".join(
        f"#{o.index} ({o.prompt_tokens}+{o.completion_tokens} tok)" for o in report.offenders[:5]
    )
    raise ProfileError(
        f"{label}: {len(report.offenders)} of {report.checked} samples exceed "
        f"max_seq_len={max_seq_len} (prompt + completion + {margin}), e.g. {shown}. "
        "Each would be truncated from the right, losing its completion and training on "
        "nothing. Raise data.max_seq_len, lower max_views_per_sample, or drop them from "
        "the corpus."
    )


class MaskAuditRecord(BaseModel):
    """One audited sample: what the real collator produced for it."""

    model_config = ConfigDict(extra="forbid")

    index: int
    prompt_tokens: int
    completion_tokens: int
    image_tokens: int
    views: int
    sequence_tokens: int
    ok: bool
    problems: list[str] = Field(default_factory=list)


class MaskAuditReport(BaseModel):
    """What :func:`audit_masks` found (plan §1.5 G2); written as ``mask_audit.json``."""

    model_config = ConfigDict(extra="forbid")

    audited: int
    passed: int
    max_length: int
    tokens_per_view: int
    prompt_min: int
    prompt_max: int
    completion_min: int
    completion_max: int
    image_min: int
    image_max: int
    truncated: int
    records: list[MaskAuditRecord]

    @property
    def ok(self) -> bool:
        """True when every audited sample passed every check."""
        return self.audited > 0 and self.passed == self.audited

    def summary(self) -> str:
        """The one log line: ``mask audit: 32/32 ok · prompt … · completion … · image …``."""
        return (
            f"mask audit: {self.passed}/{self.audited} ok · prompt "
            f"{self.prompt_min:,}–{self.prompt_max:,} tok · completion "
            f"{self.completion_min}–{self.completion_max} tok · image "
            f"{self.image_min:,}–{self.image_max:,} tok · {self.truncated} truncated"
        )


def _open_images(paths: Sequence[Any]) -> list[Any]:
    """Decode a record's view paths into RGB PIL images, as the dataset cast does."""
    from PIL import Image

    opened: list[Any] = []
    for path in paths:
        if isinstance(path, str | Path):
            with Image.open(path) as image:
                opened.append(image.convert("RGB"))
        else:
            opened.append(path)
    return opened


def _unify_placeholders(record: Mapping[str, Any]) -> dict[str, Any]:
    """The record as ``datasets.Dataset.from_list`` would hand it back.

    Arrow unifies the content structs, so an image placeholder round-trips with
    a ``text: None`` beside it and a text block unchanged. The audit feeds the
    collator the same shape the trainer will, so that unification is covered
    rather than assumed harmless.
    """
    unified = copy.deepcopy(dict(record))
    for key in ("prompt", "completion"):
        for message in unified.get(key) or ():
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    part.setdefault("text", None)
    return unified


def audit_one_mask(
    record: Mapping[str, Any],
    collator: Any,
    tokenizer: Any,
    tokens_per_view: int,
    max_length: int,
    index: int = 0,
) -> MaskAuditRecord:
    """Run the real collator on one record and check what it would train on.

    The four assertions of plan §1.5 G2:

    1. the unmasked tokens are exactly the tokenised completion;
    2. every ``<|image_pad|>`` is masked and precedes the first trained token;
    3. the image-token count is ``tokens_per_view × views``;
    4. ``0 < unmasked ≤ 512`` and the sequence was not truncated.
    """
    import torch

    example = _unify_placeholders(record)
    example["images"] = _open_images(example.get("images") or ())
    views = len(example["images"])
    batch = collator([example])
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    attention = batch["attention_mask"][0]

    pad = layout.image_token_id(tokenizer)
    trained = labels != -100
    trained_ids = input_ids[trained].tolist()
    expected = _tokenise(tokenizer, completion_text(record_completion_text(record)))
    image_positions = torch.nonzero(input_ids == pad).flatten()
    image_tokens = int(image_positions.numel())
    sequence_tokens = int(attention.sum().item())
    prompt_tokens = sequence_tokens - len(trained_ids)

    problems: list[str] = []
    if trained_ids != expected:
        problems.append(
            f"unmasked tokens ({len(trained_ids)}) are not the tokenised completion "
            f"({len(expected)})"
        )
    if bool(trained[input_ids == pad].any()):
        problems.append("an <|image_pad|> position is unmasked")
    if image_tokens and bool(trained.any()):
        first_trained = int(torch.nonzero(trained).flatten()[0].item())
        if first_trained <= int(image_positions[-1].item()):
            problems.append("the first trained token precedes the last image token")
    if image_tokens != tokens_per_view * views:
        problems.append(
            f"{image_tokens} image tokens for {views} view(s); expected "
            f"{tokens_per_view} × {views} = {tokens_per_view * views}"
        )
    if not 0 < len(trained_ids) <= MAX_COMPLETION_TOKENS:
        problems.append(f"{len(trained_ids)} trained tokens (need 0 < n ≤ {MAX_COMPLETION_TOKENS})")
    if sequence_tokens >= max_length:
        problems.append(f"sequence of {sequence_tokens} tokens hit max_length={max_length}")

    return MaskAuditRecord(
        index=index,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(trained_ids),
        image_tokens=image_tokens,
        views=views,
        sequence_tokens=sequence_tokens,
        ok=not problems,
        problems=problems,
    )


def audit_masks(
    records: Sequence[Mapping[str, Any]],
    processor: Any,
    max_length: int,
    tokens_per_view: int,
    n: int = MASK_AUDIT_SAMPLES,
    seed: int = 0,
) -> MaskAuditReport:
    """G2 — prove, with the real collator, that only the completion is trained.

    Runs *before the weights load*: the processor is a 2 MB config, and the
    collator needs nothing else. Draws *n* records spread across the corpus
    (evenly spaced from a seeded offset, so a source-ordered file contributes
    every source), decodes their views, collates each with
    ``DataCollatorForVisionLanguageModeling(completion_only_loss=True)`` and
    checks the labels against the tokenised answer. This is the check that
    would have caught the 19-hour run in 30 seconds.
    """
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    tokenizer = getattr(processor, "tokenizer", processor)
    collator = DataCollatorForVisionLanguageModeling(
        processor=processor, max_length=max_length, completion_only_loss=True
    )
    if not records:
        raise ProfileError("mask audit: no records to audit")
    count = min(n, len(records))
    rng = random.Random(seed)
    stride = len(records) / count
    offset = rng.random() * stride
    indices = sorted({min(len(records) - 1, int(offset + i * stride)) for i in range(count)})

    audited = [
        audit_one_mask(records[i], collator, tokenizer, tokens_per_view, max_length, index=i)
        for i in indices
    ]
    return MaskAuditReport(
        audited=len(audited),
        passed=sum(1 for a in audited if a.ok),
        max_length=max_length,
        tokens_per_view=tokens_per_view,
        prompt_min=min(a.prompt_tokens for a in audited),
        prompt_max=max(a.prompt_tokens for a in audited),
        completion_min=min(a.completion_tokens for a in audited),
        completion_max=max(a.completion_tokens for a in audited),
        image_min=min(a.image_tokens for a in audited),
        image_max=max(a.image_tokens for a in audited),
        truncated=sum(1 for a in audited if a.sequence_tokens >= max_length),
        records=audited,
    )


def write_mask_audit(path: Path, report: MaskAuditReport) -> None:
    """Persist the audit beside the run, whatever its verdict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def guard_masks(report: MaskAuditReport) -> None:
    """Refuse to load the weights on a failed audit.

    Raises:
        ProfileError: Any audited record failed a check; the first failures are
            listed by index.
    """
    if report.ok:
        return
    failed = [a for a in report.records if not a.ok]
    shown = "; ".join(f"#{a.index}: {', '.join(a.problems)}" for a in failed[:5])
    raise ProfileError(
        f"mask audit failed on {len(failed)} of {report.audited} samples — {shown}. "
        "The collator would not train on exactly the completion. Refusing to load "
        "the weights; see ML_PIPELINE_RECOVERY_PLAN §1.5 G2."
    )


# ---------------------------------------------------------- the layout record


LAYOUT_RECORD_NAME: Final[str] = "layout.json"


def layout_record(
    processor: Any,
    corpus_path: Path | None = None,
    base_model: str | None = None,
) -> dict[str, Any]:
    """What ``layout.json`` holds: the layout the adapter was trained under.

    ``layout_version`` and ``layout_fingerprint`` are what
    :func:`satquery.models.hf_backend.check_layout` compares at load; the rest
    is provenance for the manifest.
    """
    _, tokens_per_view = layout.fixture_render(processor)
    return {
        "layout_version": LAYOUT_VERSION,
        "layout_fingerprint": layout.layout_fingerprint(processor),
        "image_tokens_per_view": tokens_per_view,
        "fixture_view_px": layout.FIXTURE_VIEW_PX,
        "base_model": base_model,
        "git_sha": git_sha(),
        "corpus_sha256": file_sha256(corpus_path) if corpus_path else None,
        "corpus_path": str(corpus_path) if corpus_path else None,
    }


def write_layout_record(directory: Path, record: Mapping[str, Any]) -> Path:
    """Write ``layout.json`` beside an adapter."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LAYOUT_RECORD_NAME
    path.write_text(json.dumps(dict(record), indent=2, default=str), encoding="utf-8")
    return path


# ------------------------------------------------------- the deferred trainer


def _torch_dtype(name: str) -> Any:
    """Resolve a dtype name against torch, imported here and nowhere else."""
    import torch

    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ProfileError(f"unknown torch dtype {name!r}")
    return dtype


def build_bnb_config(profile: QLoraProfile) -> Any:
    """The bitsandbytes 4-bit config, or None when the profile is bf16 (§6.2)."""
    if not profile.quantization.load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    # Bound through Any locals throughout this module: peft, trl, transformers
    # and bitsandbytes are the `vlm-train` extra, so every symbol here is typed
    # on the training box and Any on a checkout that never installed them. An
    # inline ignore would be required on the first and flagged as unused on the
    # second.
    quantisation: Any = BitsAndBytesConfig
    return quantisation(
        load_in_4bit=True,
        bnb_4bit_quant_type=profile.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=profile.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_torch_dtype(profile.quantization.bnb_4bit_compute_dtype),
    )


def load_processor(profile: QLoraProfile) -> Any:
    """Load the processor alone — a 2 MB config, not the 16 GB checkpoint.

    Separate from :func:`load_model` so the mask audit and the token-budget
    guard can run *before* the weights are on the card: every one of their
    failures is then a 30-second refusal instead of a discovery hours in.

    ``max_pixels`` is set on the processor rather than left to its default: it is
    the dominant VRAM knob (§6.1), and a processor that silently upsamples a
    448 px view to its own default resolution would multiply the vision token
    count by several times per image, six times per sample.
    """
    from transformers import AutoProcessor

    auto_processor: Any = AutoProcessor
    return auto_processor.from_pretrained(
        profile.base_model,
        max_pixels=profile.data.max_pixels,
        min_pixels=profile.data.view_size_px * profile.data.view_size_px // 4,
    )


def load_model(profile: QLoraProfile) -> Any:
    """Load the quantised backbone onto device 0."""
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        profile.base_model,
        dtype=_torch_dtype(profile.torch_dtype),
        attn_implementation=profile.attn_implementation,
        quantization_config=build_bnb_config(profile),
        device_map={"": 0},
    )
    model.config.use_cache = False
    return model


def load_model_and_processor(profile: QLoraProfile) -> tuple[Any, Any]:
    """Both halves, for callers that want them together."""
    processor = load_processor(profile)
    return load_model(profile), processor


def image_tokens_per_view(
    max_pixels: int, patch_px: int = 16, merge_size: int = 2
) -> int:
    """How many language-model tokens one view becomes under a pixel budget.

    Qwen3-VL tiles the image into ``patch_px`` patches and merges them
    ``merge_size × merge_size`` before the language model: ``147456 / 16² / 2²``
    is 144, the base repo's default for a 448 px view is 196 (plan fact 8). The
    arithmetic form is what the token-budget guard uses so that no pixels need
    to be decoded; the processor's own count is what the mask audit and the
    layout fingerprint measure, and the two must agree.
    """
    return max(1, max_pixels // (patch_px * patch_px) // (merge_size * merge_size))


def build_peft_model(model: Any, profile: QLoraProfile) -> Any:
    """Attach the LoRA adapters described by the profile to a loaded model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if profile.quantization.load_in_4bit:
        prepare: Any = prepare_model_for_kbit_training
        model = prepare(
            model, use_gradient_checkpointing=profile.train.gradient_checkpointing
        )
    config = LoraConfig(
        r=profile.lora.r,
        lora_alpha=profile.lora.alpha,
        lora_dropout=profile.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=resolve_target_modules(model, profile.lora),
    )
    return get_peft_model(model, config)


def trainable_parameter_report(model: Any) -> dict[str, int]:
    """Count trainable versus total parameters, split by tower.

    Printed at the head of every run: the fastest way to notice that a checkpoint
    renamed its vision subtree is that ``vision`` reads zero here.
    """
    total = trainable = vision = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            if vision_block_index(name) is not None:
                vision += count
    return {"total": total, "trainable": trainable, "trainable_vision": vision}


def sft_config_kwargs(
    profile: QLoraProfile,
    output_dir: Path,
    has_eval: bool,
    max_steps: int | None = None,
    total_steps: int | None = None,
) -> dict[str, Any]:
    """Every argument the ``SFTConfig`` is built from, as a plain mapping.

    Separated from :func:`build_trainer` so the argument *names* can be checked
    against the installed ``trl`` without constructing a trainer — which needs a
    loaded model. trl has renamed and removed ``TrainingArguments`` fields across
    minor releases (``warmup_ratio`` is gone in 1.x), and every one of those is a
    ``TypeError`` raised after the weights are on the card and before the first
    batch.
    """
    return {
        "output_dir": str(output_dir),
        "run_name": profile.run_name,
        "per_device_train_batch_size": profile.train.per_device_train_batch_size,
        "gradient_accumulation_steps": profile.train.gradient_accumulation_steps,
        "gradient_checkpointing": profile.train.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "num_train_epochs": profile.train.num_train_epochs,
        "max_steps": max_steps if max_steps is not None else -1,
        "learning_rate": profile.train.learning_rate,
        "lr_scheduler_type": profile.train.lr_scheduler_type,
        "warmup_steps": warmup_steps(profile, total_steps or max_steps or 0),
        "optim": profile.train.optim,
        "max_grad_norm": profile.train.max_grad_norm,
        "bf16": profile.train.bf16,
        "seed": profile.train.seed,
        "dataloader_num_workers": profile.train.dataloader_num_workers,
        # Checkpointing is stated in full rather than left to the defaults.
        # A resumable run needs the optimiser, the scheduler and the RNG state,
        # not just the adapter, and `save_strategy` is what decides whether
        # `save_steps` is consulted at all — a default that has moved between
        # transformers releases. `save_total_limit` bounds the run's disk
        # without bounding the run: the trainer always keeps the newest, which
        # is the one `--resume` comes back to.
        "save_strategy": profile.train.save_strategy,
        "save_steps": profile.train.save_steps,
        "save_total_limit": profile.train.save_total_limit,
        # The field that actually decides whether a run is resumable. Left at
        # its default it is already False, but "the optimiser and scheduler
        # states are on disk" is the whole point of the checkpoint here, and a
        # default is a weak place to keep a load-bearing fact: `save_only_model:
        # True` writes an adapter that can be *served* and cannot be *resumed*,
        # and the difference does not show up until the resume silently restarts
        # Adam from zero moment estimates.
        "save_only_model": False,
        "eval_steps": profile.train.eval_steps,
        "eval_strategy": "steps" if has_eval else "no",
        "logging_steps": profile.train.logging_steps,
        "max_length": profile.data.max_seq_len,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        # The masking fix (ML_PIPELINE_RECOVERY_PLAN §1.4). EXPLICIT True, never
        # None: None auto-detects from the record shape, and a regression to the
        # `messages` shape would silently revert to full-sequence loss — the
        # exact failure being fixed. With True, a `messages` record makes trl
        # raise instead.
        "completion_only_loss": True,
        # Explicit False, with the reason: trl 1.12.0 raises
        # ValueError("Assistant-only loss is not yet supported for vision
        # datasets") for any dataset carrying an `images` key
        # (sft_trainer.py:1052). Stated here so a well-meant "fix" that flips it
        # fails the unit test rather than the run.
        "assistant_only_loss": False,
        # The vision prompt-completion collator raises NotImplementedError if
        # this is set (sft_trainer.py:693-697).
        "pad_to_multiple_of": None,
        # The default, stated: keep_end is rejected for vision datasets because
        # the image tokens live in the prompt and would be the first to go.
        "truncation_mode": "keep_start",
        # packing and padding_free stay unset: both are rejected for vision
        # datasets (sft_trainer.py:1043-1051).
    }


def warmup_steps(profile: QLoraProfile, total_steps: int) -> int:
    """Convert the profile's warmup *ratio* into the warmup *steps* trl takes.

    §6.1 states warmup as a ratio, and ``TrainingArguments`` used to accept one
    directly. ``trl``'s ``SFTConfig`` (1.x) exposes only ``warmup_steps``, so
    passing ``warmup_ratio`` raises ``TypeError`` before the first batch. The
    profile keeps the ratio — it is the frozen recipe, and it is the form that
    stays correct when the corpus or the epoch count changes — and the
    translation happens here, at the boundary, where the total step count is
    known.

    A ratio that rounds below one step still warms up for one: starting a cosine
    schedule at the peak learning rate on a 4-bit backbone is how the first
    optimiser step undoes the initialisation.
    """
    if total_steps <= 0 or profile.train.warmup_ratio <= 0:
        return 0
    return max(1, round(total_steps * profile.train.warmup_ratio))


class InterruptRequest:
    """A one-way flag shared by the signal handler and the trainer callback.

    Deliberately not a ``threading.Event``: the only writer is a signal handler,
    which must do the least possible work, and the only reader is the training
    loop between steps.
    """

    def __init__(self) -> None:
        """Start un-requested."""
        self.requested = False
        self.signal_name: str | None = None

    def request(self, signal_name: str) -> None:
        """Record that a stop was asked for, and by which signal."""
        self.requested = True
        self.signal_name = signal_name


def install_interrupt_handler(
    request: InterruptRequest, signals: Sequence[int] | None = None
) -> Callable[[], None]:
    """Turn Ctrl+C into "checkpoint at the next step, then stop cleanly".

    Without this, SIGINT raises ``KeyboardInterrupt`` from wherever the loop
    happens to be — mid-backward, mid-collate — and the process dies with
    everything since the last periodic save thrown away, no adapter written and
    no manifest. Which is the one thing an overnight run on a desktop cannot
    afford, because the way that run *ends* is somebody pressing Ctrl+C.

    The handler only sets a flag. The actual save happens in the callback from
    :func:`graceful_stop_callback`, at a step boundary, where the optimiser and
    scheduler states are consistent and the trainer's own checkpoint writer can
    do its job.

    The original handlers are restored on the *first* signal, so a second Ctrl+C
    aborts immediately the old way. That matters: the graceful path still has to
    finish the current optimiser step and write a checkpoint, and an operator
    who has decided not to wait must always be able to leave.

    Returns:
        A callable that restores the previous handlers, for the caller's
        ``finally``.
    """
    import signal as signal_module

    watched = tuple(signals or (signal_module.SIGINT, signal_module.SIGTERM))
    previous: dict[int, Any] = {}

    def handle(number: int, frame: Any) -> None:
        name = signal_module.Signals(number).name
        request.request(name)
        restore()
        print(
            f"\n{name} received — finishing the current optimiser step, writing a "
            "checkpoint, then stopping. Press Ctrl+C again to abort now and lose "
            "everything since the last checkpoint.",
            flush=True,
        )

    def restore() -> None:
        """Put the handlers that were there before back."""
        for number, handler in previous.items():
            # pragma: no cover - only fails off the main thread
            with contextlib.suppress(OSError, ValueError):
                signal_module.signal(number, handler)
        previous.clear()

    for number in watched:
        try:
            previous[number] = signal_module.signal(number, handle)
        except (OSError, ValueError):  # pragma: no cover - non-main thread
            continue
    return restore


def graceful_stop_callback(request: InterruptRequest) -> Any:
    """A ``TrainerCallback`` that checkpoints and stops when *request* is set.

    ``should_save`` before ``should_training_stop``, both on the same step: the
    trainer writes the checkpoint on its way out, so the run that resumes starts
    from where the interrupt landed rather than from the last periodic save.

    Checked at ``on_step_end`` — an *optimiser* step, not a micro-batch. With
    ``gradient_accumulation_steps: 16`` that is up to sixteen forward/backward
    passes of latency between Ctrl+C and the process exiting. It is the earliest
    point at which the accumulated gradients have been applied and the state on
    disk would be one a resume can trust.
    """
    from transformers import TrainerCallback

    base: Any = TrainerCallback

    class _GracefulStop(base):  # type: ignore[misc]
        """Turns the interrupt flag into the trainer's own stop-and-save controls."""

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            """Ask for a checkpoint and an exit if a signal has arrived."""
            if request.requested:
                control.should_save = True
                control.should_training_stop = True
            return control

    return _GracefulStop()


def latest_checkpoint(output_dir: Path) -> Path | None:
    """The newest ``checkpoint-N`` in *output_dir*, by step number.

    By step, not by mtime: ``save_total_limit`` deletes older directories, and a
    filesystem restored from a backup or copied between disks can carry mtimes
    that no longer reflect the order the checkpoints were written in.
    """
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.removeprefix("checkpoint-")))


def checkpoint_step(checkpoint: Path) -> int:
    """The optimiser step a checkpoint directory holds."""
    return int(checkpoint.name.removeprefix("checkpoint-"))


def describe_resume(output_dir: Path, setting: str) -> str:
    """One line saying what this invocation will do about an existing checkpoint."""
    target = resume_target(output_dir, setting)
    if target is None:
        existing = latest_checkpoint(output_dir)
        if existing is not None:
            return (
                f"starting from scratch, ignoring {existing.name} "
                f"(resume={setting!r})"
            )
        return "starting from scratch (no checkpoint present)"
    if target is True:
        found = latest_checkpoint(output_dir)
        assert found is not None  # resume_target only returns True when one exists
        return f"resuming from {found.name} (step {checkpoint_step(found)})"
    return f"resuming from {target}"


def build_trainer(
    profile: QLoraProfile,
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    output_dir: Path,
    max_steps: int | None = None,
    total_steps: int | None = None,
    interrupt: InterruptRequest | None = None,
) -> Any:
    """Assemble the ``trl`` SFT trainer for this profile.

    Args:
        profile: The validated training profile.
        model: The PEFT-wrapped, quantised model.
        processor: The processor the weights were loaded with.
        train_dataset: Chat-formatted training records.
        eval_dataset: Chat-formatted validation records, or None.
        output_dir: Checkpoint and log destination.
        max_steps: Hard stop, for a sanity check or a truncated run.
        total_steps: The run's planned optimiser steps, used to place the warmup.
            Defaults to *max_steps* when that is the shorter of the two.
        interrupt: Shared stop flag; when given, the trainer checkpoints and
            exits cleanly on the next step boundary after a signal arrives.
    """
    from trl import SFTConfig

    config = SFTConfig(
        **sft_config_kwargs(
            profile=profile,
            output_dir=output_dir,
            has_eval=eval_dataset is not None,
            max_steps=max_steps,
            total_steps=total_steps,
        )
    )
    trainer_class = satquery_sft_trainer_class()
    trainer = trainer_class(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        callbacks=[graceful_stop_callback(interrupt)] if interrupt else None,
    )
    check_trainer_masking(trainer)
    return trainer


def check_trainer_masking(trainer: Any) -> None:
    """G1 — the trainer self-check (plan §1.5).

    Private attributes on a pinned trl (``pyproject.toml`` pins ``trl==1.12.0``;
    the pin is part of the fix): the trainer must have recognised the dataset
    as vision, must be masking the prompt, and must be using the vision
    collator whose prompt-completion path does the masking.

    Raises:
        ProfileError: Any of the three is not so.
    """
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

    problems: list[str] = []
    if getattr(trainer, "_is_vision_dataset", None) is not True:
        problems.append("the dataset was not recognised as vision (no top-level `images` key?)")
    if getattr(trainer, "completion_only_loss", None) is not True:
        problems.append("completion_only_loss did not resolve to True")
    if not isinstance(trainer.data_collator, DataCollatorForVisionLanguageModeling):
        problems.append(
            f"the collator is {type(trainer.data_collator).__name__}, not "
            "DataCollatorForVisionLanguageModeling"
        )
    if problems:
        raise ProfileError(
            "the trainer would not mask the prompt: " + "; ".join(problems) + ". "
            "Refusing to start; see ML_PIPELINE_RECOVERY_PLAN §1."
        )


def answer_token_mask(shift_labels: Any, tail: int = STRUCTURAL_TAIL_TOKENS) -> Any:
    """The supervised positions minus the last *tail* per row.

    ``shift_labels != -100`` is trl's own accuracy mask; this removes the final
    *tail* unmasked positions of every row — ``<|im_end|>`` and the newline —
    so what remains is the answer. A row with *tail* or fewer supervised tokens
    contributes nothing rather than going negative.
    """
    import torch

    mask = shift_labels != -100
    if tail <= 0:
        return mask
    # Rank each supervised position from the end of its row: the last one is 1.
    reversed_counts = torch.flip(torch.cumsum(torch.flip(mask.long(), dims=[1]), dim=1), dims=[1])
    return mask & (reversed_counts > tail)


def satquery_sft_trainer_class() -> Any:
    """``SFTTrainer`` plus ``answer_token_accuracy`` (plan §3.6), built on demand.

    Defined inside a factory so importing this module never imports trl.

    41 % of the corpus's answers are ≤ 3 characters, so their completion is one
    or two answer tokens plus ``<|im_end|>`` and a newline. Once the model has
    learned the two structural tokens those samples sit at 50–67 % token
    accuracy with the *answer* still wrong. This subclass reports accuracy over
    completion positions excluding the final two, alongside trl's own number.

    Under ``loss_type="chunked_nll"`` — the default, kept because it runs the
    ``lm_head`` only on unmasked positions — no logits are ever materialised,
    so the metric is computed from the post-norm hidden state the chunked loss
    itself consumes, captured by a forward hook on the decoder's final norm,
    projected through ``lm_head`` for the completion positions only (a few
    hundred rows of a 152 k-wide matmul, under ``no_grad``). When trl's
    non-chunked path is in use the logits exist and are used directly.
    """
    import torch
    from trl import SFTTrainer

    base: Any = SFTTrainer

    class SatQuerySFTTrainer(base):  # type: ignore[misc]
        """trl's trainer with the structural-token exclusion of plan §3.6."""

        def __init__(self, *args: Any, answer_accuracy: bool = True, **kwargs: Any) -> None:
            """Wire the norm hook lazily; nothing runs until the first forward."""
            super().__init__(*args, **kwargs)
            self.answer_accuracy = answer_accuracy
            self._captured_hidden: Any | None = None
            self._norm_hook: Any | None = None

        def _final_norm(self) -> Any | None:
            """The decoder's final norm module, whose output feeds ``lm_head``."""
            model = self.model
            base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
            decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
            norm = getattr(decoder, "norm", None)
            return norm if isinstance(norm, torch.nn.Module) else None

        def _ensure_hook(self) -> bool:
            if self._norm_hook is not None:
                return True
            norm = self._final_norm()
            if norm is None:
                return False

            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                self._captured_hidden = output.detach()

            self._norm_hook = norm.register_forward_hook(capture)
            return True

        def compute_loss(
            self,
            model: Any,
            inputs: Any,
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            """The trl loss, then the answer-only accuracy from the same forward."""
            mode = "train" if self.model.training else "eval"
            labels = inputs.get("labels") if "shift_labels" not in inputs else None
            hooked = self.answer_accuracy and labels is not None and self._ensure_hook()
            self._captured_hidden = None
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            if hooked:
                accuracy = self._answer_accuracy(outputs, labels)
                if accuracy is not None:
                    self._metrics[mode]["answer_token_accuracy"].append(accuracy)
            self._captured_hidden = None
            return (loss, outputs) if return_outputs else loss

        def _answer_accuracy(self, outputs: Any, labels: Any) -> float | None:
            with torch.no_grad():
                shift_labels = labels[..., 1:]
                mask = answer_token_mask(shift_labels)
                total = int(mask.sum().item())
                if total == 0:
                    return None
                logits = getattr(outputs, "logits", None)
                if logits is not None:
                    predictions = logits[..., :-1, :].argmax(dim=-1)
                else:
                    hidden = self._captured_hidden
                    if hidden is None:
                        return None
                    shift_hidden = hidden[:, :-1, :][mask]
                    head = self.model.get_output_embeddings()
                    predictions = head(shift_hidden.to(head.weight.dtype)).argmax(dim=-1)
                    correct = (predictions == shift_labels[mask]).sum()
                    return float(correct.item() / total)
                correct = ((predictions == shift_labels) & mask).sum()
                return float(correct.item() / total)

    return SatQuerySFTTrainer


def resume_target(output_dir: Path, setting: str) -> bool | str | None:
    """Resolve ``resume_from_checkpoint`` to what the trainer expects.

    ``auto`` resumes when a checkpoint is present and starts fresh when it is
    not, which is what makes an interrupted overnight run restartable with the
    same command line.
    """
    value = setting.strip().lower()
    if value in {"", "false", "no", "none"}:
        return None
    if value == "auto":
        return True if any(output_dir.glob("checkpoint-*")) else None
    if value in {"true", "yes"}:
        return True
    # An explicit path, checked here rather than by the trainer: a typo in a
    # checkpoint name otherwise surfaces after the weights are on the card.
    path = Path(setting)
    if not (path / "trainer_state.json").is_file():
        raise ProfileError(
            f"{setting} is not a resumable checkpoint (no trainer_state.json). "
            f"Available in {output_dir}: "
            f"{sorted(p.name for p in output_dir.glob('checkpoint-*')) or 'none'}"
        )
    return setting


def write_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Record what produced an adapter, next to the adapter.

    The registry records which adapter answered a query (Master.md §8 Phase 7);
    this is the other half of that provenance — which profile, corpus and prompt
    version produced the adapter in the first place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")


__all__ = [
    "COMPLETION_SUFFIX",
    "LAYOUT_RECORD_NAME",
    "MASK_AUDIT_SAMPLES",
    "MAX_COMPLETION_TOKENS",
    "STRUCTURAL_TAIL_TOKENS",
    "TOKEN_BUDGET_MARGIN",
    "MaskAuditRecord",
    "MaskAuditReport",
    "TokenBudgetOffender",
    "TokenBudgetReport",
    "answer_token_mask",
    "audit_masks",
    "audit_one_mask",
    "check_trainer_masking",
    "completion_text",
    "completion_token_count",
    "corpus_composition",
    "file_sha256",
    "git_sha",
    "guard_composition",
    "guard_images_present",
    "guard_masks",
    "guard_token_budget",
    "image_tokens_per_view",
    "layout_record",
    "load_model",
    "load_processor",
    "missing_images",
    "prompt_token_count",
    "record_completion_text",
    "satquery_sft_trainer_class",
    "stratified_subset",
    "stratum_of",
    "token_budget_report",
    "write_layout_record",
    "write_mask_audit",
    "InterruptRequest",
    "checkpoint_step",
    "describe_resume",
    "graceful_stop_callback",
    "install_interrupt_handler",
    "latest_checkpoint",
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_VISION_BLOCKS",
    "FALLBACK_PROFILE_PATH",
    "LLM_TARGET_MODULES",
    "MEASURED_PEAKS",
    "SAFE_PEAK_BYTES",
    "SANITY_SAMPLES",
    "VISION_TARGET_MODULES",
    "DataSpec",
    "LoraSpec",
    "KNOWN_SHAPES",
    "MeasuredPeak",
    "ModelShape",
    "ProfileError",
    "QLoraProfile",
    "QuantizationSpec",
    "RunPlan",
    "TrainSpec",
    "VramEstimate",
    "build_bnb_config",
    "build_peft_model",
    "build_trainer",
    "estimate_adapter_params",
    "estimate_footprint",
    "guard_budget",
    "iter_corpus",
    "load_corpus",
    "load_model_and_processor",
    "load_profile",
    "plan_run",
    "resolve_target_modules",
    "resume_target",
    "image_placeholder_count",
    "sample_to_chat",
    "select_vision_blocks",
    "sft_config_kwargs",
    "shape_for",
    "split_targets",
    "trainable_parameter_report",
    "vision_block_index",
    "warmup_steps",
    "write_run_manifest",
]
