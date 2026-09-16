"""Batch inference over corpus rows, written out as it goes.

The runner builds one :class:`~satquery.models.loader.GenerationRequest` per
sample from the row's own ``system`` and ``user`` turns and its rendered views
— never from the assistant turn — and sends it through whatever backend the
caller hands it. The backend is the production one
(:func:`satquery.models.loader.get_backend`), so ``SATQUERY_VLM_BACKEND=hf``
scores the adapter over bf16 and ``llamacpp`` scores the Q4_K_M GGUF with no
code change; the two columns of one table come from one code path.

Every prediction is appended to ``predictions.jsonl`` the moment it exists.
A run that dies at sample 312 of 500 is resumed with ``resume=True`` and
starts at 313, which matters when one sample is five seconds and the box is
shared with training.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from satquery.core.logging import get_logger
from satquery.models.loader import BackendKind, GenerationRequest, PromptImage, VlmBackend
from satquery.training.corpus_builder import CorpusSample

__all__ = [
    "Prediction",
    "build_request",
    "read_predictions",
    "reference_backend",
    "run",
]

log = get_logger(__name__)


@dataclass(frozen=True)
class Prediction:
    """One generated answer and what it cost, as a ``predictions.jsonl`` row."""

    id: str
    source: str
    task: str
    pair_type: str
    reference: str
    prediction: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: bool = False
    backend: str = ""
    model_id: str = ""
    error: str | None = None

    def to_json(self) -> str:
        """One JSONL line."""
        return json.dumps(asdict(self), ensure_ascii=False)


def _view_image(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def build_request(
    sample: CorpusSample,
    root: Path,
    max_views: int = 6,
    max_new_tokens: int = 256,
) -> GenerationRequest:
    """The serving request for one corpus row.

    View paths in the corpus are relative to the repository root
    (``data/processed/views/...``); *root* is where they resolve. Views are
    taken in slot order, at most *max_views*, exactly as the trainer's
    ``sample_to_chat`` takes them.

    Raises:
        FileNotFoundError: A view the row names is not on disk.
    """
    views = sorted(sample.views, key=lambda view: view.slot)[:max_views]
    images = tuple(
        PromptImage(label=view.label, rgb=_view_image(root / view.path)) for view in views
    )
    return GenerationRequest(
        system=sample.system,
        user=sample.user,
        images=images,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )


def read_predictions(path: Path) -> dict[str, Prediction]:
    """The rows already in ``predictions.jsonl``, keyed by sample id."""
    if not path.exists():
        return {}
    rows: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                rows[payload["id"]] = Prediction(**payload)
    return rows


def run(
    samples: Iterable[CorpusSample],
    backend: VlmBackend,
    out: Path,
    *,
    root: Path,
    max_views: int = 6,
    max_new_tokens: int = 256,
    resume: bool = False,
    progress: Callable[[int, int, Prediction], None] | None = None,
) -> Iterator[Prediction]:
    """Generate an answer for every sample, appending each to *out*.

    A generation that raises is recorded as a row with ``error`` set and an
    empty prediction — it scores as wrong, which is what it is — rather than
    ending the run; the exception text is in the row for the post-mortem.

    Args:
        samples: The rows to score.
        backend: Any loaded-or-lazy VLM backend.
        out: ``predictions.jsonl``; created, or appended to when resuming.
        root: Directory the corpus view paths resolve against.
        max_views: Views per prompt (the training budget is six).
        max_new_tokens: Decode budget; the longest reference in the corpus
            is a caption of ~120 tokens, so 256 leaves room without letting
            a runaway decode cost a minute.
        resume: Skip ids already present in *out*.
        progress: Called as ``(done, total, prediction)`` after each row.

    Yields:
        Every prediction, resumed rows included, in sample order.
    """
    rows = list(samples)
    existing = read_predictions(out) if resume else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and out.exists() else "w"
    with out.open(mode, encoding="utf-8") as handle:
        for index, sample in enumerate(rows, start=1):
            if sample.id in existing:
                prediction = existing[sample.id]
            else:
                prediction = _predict(sample, backend, root, max_views, max_new_tokens)
                handle.write(prediction.to_json() + "\n")
                handle.flush()
            if progress is not None:
                progress(index, len(rows), prediction)
            yield prediction


def _predict(
    sample: CorpusSample,
    backend: VlmBackend,
    root: Path,
    max_views: int,
    max_new_tokens: int,
) -> Prediction:
    started = time.perf_counter()
    # The reference backend (self-check) answers per sample, not per prompt:
    # user turns repeat across patches, so it is told which row is up.
    bind = getattr(backend, "bind_sample", None)
    if bind is not None:
        bind(sample.id)
    try:
        request = build_request(sample, root, max_views=max_views, max_new_tokens=max_new_tokens)
        result = backend.generate(request)
    except Exception as error:  # noqa: BLE001 - one bad sample must not end a 500-sample run
        log.warning("eval.sample_failed", sample_id=sample.id, error=str(error))
        return Prediction(
            id=sample.id,
            source=sample.source.value,
            task=sample.task.value,
            pair_type=sample.pair_type.value,
            reference=sample.assistant,
            prediction="",
            latency_ms=int((time.perf_counter() - started) * 1000),
            backend=str(getattr(backend, "kind", "")),
            model_id=str(getattr(backend, "model_id", "")),
            error=f"{type(error).__name__}: {error}",
        )
    return Prediction(
        id=sample.id,
        source=sample.source.value,
        task=sample.task.value,
        pair_type=sample.pair_type.value,
        reference=sample.assistant,
        prediction=result.text,
        latency_ms=result.duration_ms or int((time.perf_counter() - started) * 1000),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        truncated=result.truncated,
        backend=str(result.backend),
        model_id=result.model_id,
    )


class _ReferenceBackend:
    """Answers every request with the reference it was built from.

    Not a model. It exists so the whole pipeline — sampling, request
    construction, scoring, reporting — runs on a laptop with no weights and
    produces a table of all-ones, which is the self-check that the scorers
    accept their own references. A real run that scores below this ceiling
    is the model's doing; a self-check that does not reach it is ours.
    """

    kind: BackendKind = BackendKind.NONE
    model_id: str = "reference"

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers
        self._current: str | None = None

    def bind_sample(self, sample_id: str) -> None:
        """Select the row the next :meth:`generate` answers for."""
        self._current = sample_id

    @property
    def is_loaded(self) -> bool:
        """Always ready."""
        return True

    def load(self) -> None:
        """Nothing to load."""

    def unload(self) -> None:
        """Nothing to unload."""

    def generate(self, request: GenerationRequest) -> Any:
        """Return the bound row's reference; the request itself is ignored."""
        from satquery.models.loader import GenerationResult

        text = self._answers.get(self._current or "", "")
        return GenerationResult(
            text=text, backend=BackendKind.NONE, model_id="reference", device="cpu"
        )


def reference_backend(samples: Iterable[CorpusSample]) -> VlmBackend:
    """A backend that returns each sample's own reference — the scorer self-check."""
    return _ReferenceBackend({sample.id: sample.assistant for sample in samples})
