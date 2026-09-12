"""The offline demo path: a GGUF quantisation behind llama.cpp's server.

Why this exists at all, given the transformers backend works: the demo is given
in a room whose network cannot be relied on, on a machine that may already have
the CV models resident. A Q4_K_M GGUF of the same 8B model runs in about 6 GB
instead of 16 and starts in seconds instead of a minute, and llama.cpp's
``llama-server`` speaks an OpenAI-shaped API that needs no Python ML stack at
all — so this path also survives a broken ROCm install, which is the failure mode
that actually bites on Fedora after a kernel update (Master.md §8 Phase 4).

It is the *same interface*, deliberately. :class:`LlamaCppBackend` and
:class:`~satquery.models.hf_backend.HuggingFaceBackend` are interchangeable, so
switching to the offline path is one environment variable and nothing else. The
answers differ — quantisation costs accuracy — which is why Phase 4's verification
compares them rather than assuming they agree.

The transport is :mod:`urllib` from the standard library. An HTTP client
dependency for four calls against a process on loopback would be a dependency
carried into the offline demo for no benefit.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from typing import Any, Final

import numpy as np
from PIL import Image

from satquery.models.loader import (
    BackendConfig,
    BackendKind,
    GenerationRequest,
    GenerationResult,
    LazyBackend,
    ModelLoadError,
    PromptImage,
    apply_stop,
)
from satquery.models.prompts.layout import user_turn_content

CHAT_PATH: Final[str] = "/v1/chat/completions"
HEALTH_PATH: Final[str] = "/health"
PROBE_TIMEOUT_S: Final[float] = 3.0
"""Long enough for a loaded server on loopback, short enough that a wrong URL
costs one degraded step rather than the request's whole timeout budget."""

JPEG_QUALITY: Final[int] = 92
"""Matches the artifact store, so the offline path sees the same encoding
artefacts the evidence gallery shows and the corpus was built with."""


class LlamaCppError(ModelLoadError):
    """The llama.cpp server could not be reached or returned an error."""


def encode_data_uri(image: PromptImage) -> str:
    """Encode a rendered view as the ``data:`` URI llama.cpp's server accepts."""
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image.rgb, dtype=np.uint8), mode="RGB").save(
        buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0
    )
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _image_url_block(uri: Any) -> dict[str, Any]:
    """The OpenAI-shaped image block llama-server accepts."""
    return {"type": "image_url", "image_url": {"url": uri}}


def build_messages(request: GenerationRequest) -> list[dict[str, Any]]:
    """Lay the request out as OpenAI-shaped multimodal chat messages.

    The block *type* is llama.cpp's (``image_url`` with a data URI), but the
    *ordering* — label immediately before its own image, question last — comes
    from :func:`~satquery.models.prompts.layout.user_turn_content`, the same
    function the transformers backend and the training records use. The binding
    between a named view and its pixels is what the whole rendering strategy
    rests on, and it has to be identical on every path or the backends stop
    being comparable.
    """
    content = user_turn_content(
        [image.label for image in request.images],
        request.user,
        images=[encode_data_uri(image) for image in request.images],
        image_block=_image_url_block,
    )
    return [
        {"role": "system", "content": request.system},
        {"role": "user", "content": content},
    ]


class LlamaCppBackend(LazyBackend):
    """An interchangeable client for a llama.cpp ``llama-server`` instance.

    "Loading" here means confirming the server is up and holding the model, not
    reading weights into this process — but it is the same lifecycle from the
    caller's side, which is the point of the shared base class.
    """

    kind = BackendKind.LLAMACPP

    def __init__(self, config: BackendConfig) -> None:
        """Wire the client to its server address."""
        super().__init__(config)
        self.base_url = (config.server_url or "http://127.0.0.1:8080").rstrip("/")

    # -- lifecycle ---------------------------------------------------------

    def _load(self) -> None:
        """Confirm the server answers before a step commits to using it.

        Raises:
            LlamaCppError: The server is unreachable or not ready.
        """
        try:
            self._request(HEALTH_PATH, payload=None, timeout=PROBE_TIMEOUT_S)
        except LlamaCppError as error:
            raise LlamaCppError(
                f"no llama.cpp server answering at {self.base_url}: {error}. "
                f"Start one with scripts/serve_vlm.sh."
            ) from error

    def _unload(self) -> None:
        """Nothing is resident in this process; the server owns the weights."""
        return

    # -- generation --------------------------------------------------------

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Send one completion request and read the answer back.

        Raises:
            LlamaCppError: The server errored or returned an unusable body.
        """
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": build_messages(request),
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
            "stream": False,
        }
        if not request.greedy:
            body["top_p"] = request.top_p
        if request.stop:
            body["stop"] = list(request.stop)

        started = time.perf_counter()
        response = self._request(
            CHAT_PATH, payload=body, timeout=self.config.request_timeout_s
        )
        duration_ms = int((time.perf_counter() - started) * 1000)

        choices = response.get("choices") or []
        if not choices:
            raise LlamaCppError(f"llama.cpp returned no choices: {response}")
        choice = choices[0]
        # llama-server strips the sequence it stopped on, but only the one it
        # matched, and only when it stopped on a string rather than the budget.
        # Trimming again here costs nothing and is what makes the two backends
        # return the same text for the same runaway decode.
        text = apply_stop(
            str((choice.get("message") or {}).get("content") or ""), request.stop
        )
        usage = response.get("usage") or {}

        return GenerationResult(
            text=text,
            backend=self.kind,
            model_id=str(response.get("model") or self.model_id),
            device=self.base_url,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            duration_ms=duration_ms,
            truncated=choice.get("finish_reason") == "length",
        )

    # -- transport ---------------------------------------------------------

    def _request(
        self, path: str, payload: dict[str, Any] | None, timeout: float
    ) -> dict[str, Any]:
        """POST JSON (or GET when *payload* is None) and decode the response.

        Raises:
            LlamaCppError: The call failed, timed out, or returned non-JSON.
        """
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        http_request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(http_request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:  # pragma: no cover - server-dependent
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise LlamaCppError(f"{url} returned HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LlamaCppError(f"{url} is unreachable: {error}") from error

        try:
            decoded: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as error:  # pragma: no cover - server-dependent
            raise LlamaCppError(f"{url} returned a non-JSON body: {raw[:200]}") from error
        return decoded


__all__ = [
    "CHAT_PATH",
    "HEALTH_PATH",
    "LlamaCppBackend",
    "LlamaCppError",
    "build_messages",
    "encode_data_uri",
]
