"""The one place a Qwen3-VL user turn is laid out — for training and for serving.

ML_PIPELINE_RECOVERY_PLAN §2 records the skew this module closes. The system
text, the user text and the view labels were already produced by one builder on
both paths; what differed was the *layout of the user turn*. Serving interleaved
``label_i, image_i``; training emitted ``image_1 … image_n, user`` with the
labels attached under a key the chat template never prints. The adapter was
fitted to unlabelled pixels and served on labelled ones.

So the layout is a function, not a convention, and every caller goes through
it: :func:`~satquery.training.vlm.qlora.sample_to_chat` builds the prompt half
of a training record with :func:`chat_prompt`, and
:func:`satquery.models.hf_backend.build_messages` hands the *same* call's
result to ``apply_chat_template``. Neither is allowed to assemble a content list
by hand again — a test greps for it.

Pure Python, no torch: importable by the training package, the serving package
and the test suite alike. The only function that touches a processor is
:func:`layout_fingerprint`, which takes one as an argument and is what the
serving path uses to refuse an adapter trained under a different layout or
pixel budget (plan §2.3, level L3).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any, Final

LAYOUT_VERSION: Final[str] = "label-before-image/v1"
"""Names the layout this module produces. Written into ``layout.json`` beside
every adapter and compared at load: an adapter carrying a different version was
trained on a different token stream and must not be served under this one."""

ImageBlock = Callable[[Any], dict[str, Any]]
"""Maps one image payload to the content block a particular consumer expects."""


def _placeholder_block(_: Any) -> dict[str, Any]:
    """The trl contract: a bare ``{"type": "image"}`` and nothing else.

    ``prepare_multimodal_messages`` counts blocks of type ``image`` that carry no
    ``image`` key and refuses the example unless that count equals
    ``len(images)``. Any extra key is tolerated by the template but a payload
    key here would make the count zero.
    """
    return {"type": "image"}


def _payload_block(payload: Any) -> dict[str, Any]:
    """The transformers contract: the PIL image rides inside the block."""
    return {"type": "image", "image": payload}


def user_turn_content(
    labels: Sequence[str],
    user_text: str,
    images: Sequence[Any] | None = None,
    image_block: ImageBlock | None = None,
) -> list[dict[str, Any]]:
    """``[label_1, image_1, …, label_n, image_n, user_text]`` as content blocks.

    Each view is named by a text part *immediately before* its own pixels, so
    the label and the image are adjacent in the token stream. Naming views only
    in the system prompt leaves the model to align six unlabelled tensors
    against a list, which it does unreliably and which breaks the moment a view
    is unavailable and the later slots renumber (DATA_ADAPTATION_PLAN §2.4).

    Args:
        labels: One label per view, in slot order — the strings
            :mod:`satquery.render.view_labels` produced.
        user_text: The analyst's question; always the last block.
        images: The per-view payloads. ``None`` emits bare placeholders (the
            training record; trl fills them from the record's ``images`` list).
            Given, each is wrapped by *image_block*.
        image_block: How a payload becomes a block. Defaults to transformers'
            ``{"type": "image", "image": payload}``; the llama.cpp client passes
            its ``image_url`` mapper so the *ordering* still comes from here.

    Raises:
        ValueError: ``images`` was given with a different length from ``labels``.
    """
    if images is not None and len(images) != len(labels):
        raise ValueError(
            f"{len(labels)} view label(s) but {len(images)} image payload(s); every view "
            "is labelled immediately before its own pixels, so the two must pair up"
        )
    block = image_block or (_placeholder_block if images is None else _payload_block)
    content: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        content.append({"type": "text", "text": label})
        content.append(block(None if images is None else images[index]))
    content.append({"type": "text", "text": user_text})
    return content


def chat_prompt(
    system: str,
    labels: Sequence[str],
    user_text: str,
    images: Sequence[Any] | None = None,
    image_block: ImageBlock | None = None,
) -> list[dict[str, Any]]:
    """``[system turn, user turn]``.

    The prompt half of a prompt-completion training record and, identically, the
    message list :mod:`satquery.models.hf_backend` hands to ``apply_chat_template``
    with ``add_generation_prompt=True``. Same function, same bytes.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {
            "role": "user",
            "content": user_turn_content(labels, user_text, images, image_block),
        },
    ]


def strip_image_payloads(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same messages with every image block reduced to a bare placeholder.

    What the L1 parity test compares: the serving path's messages, payloads
    removed, must equal the training record's prompt exactly.
    """
    stripped: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(dict(message))
            continue
        blocks = [
            {"type": "image"} if block.get("type") == "image" else dict(block)
            for block in content
        ]
        stripped.append({**message, "content": blocks})
    return stripped


def image_placeholder_count(messages: Sequence[dict[str, Any]]) -> int:
    """Count the placeholders trl will try to fill in a message list.

    Exactly trl's own rule — a block of type ``image`` carrying no ``image``
    payload — so a record can be checked against its own ``images`` list before
    it reaches the trainer. A ``text: None`` beside the placeholder (what
    ``datasets.Dataset.from_list`` adds when it unifies the content structs) does
    not make a block a payload, in trl's reading or in this one.
    """
    return sum(
        1
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image" and "image" not in part
    )


# ------------------------------------------------------------ the fingerprint

FIXTURE_SYSTEM: Final[str] = "layout fingerprint fixture — system"
FIXTURE_LABELS: Final[tuple[str, ...]] = (
    "Image 1 (optical true colour, Sentinel-2)",
    "Image 2 (optical NDVI heatmap, Sentinel-2, fixed scale -1 to +1)",
)
FIXTURE_USER: Final[str] = "layout fingerprint fixture — user"
FIXTURE_VIEW_PX: Final[int] = 448
"""The view size every corpus is rendered at (DataSpec.view_size_px). The
fingerprint's image-token count is measured on a canvas of exactly this size, so
it is the number the adapter was trained under — 144 at the training profile's
``max_pixels``, 196 under the base repo's default (plan fact 8)."""

IMAGE_PAD_TOKEN: Final[str] = "<|image_pad|>"


def fixture_image() -> Any:
    """A deterministic 448 px RGB image. Its pixels do not matter; its size does."""
    from PIL import Image

    return Image.new("RGB", (FIXTURE_VIEW_PX, FIXTURE_VIEW_PX), (64, 128, 192))


def image_token_id(processor: Any) -> int:
    """The id ``<|image_pad|>`` tokenises to under this processor."""
    tokenizer = getattr(processor, "tokenizer", processor)
    token_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
    if not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"{IMAGE_PAD_TOKEN} is not a token of this processor")
    return token_id


def fixture_render(processor: Any) -> tuple[str, int]:
    """The templated fixture prompt and its image-token count under *processor*.

    The text is ``apply_chat_template(..., tokenize=False, add_generation_prompt=True)``
    over :func:`chat_prompt` of the fixture — the exact call the serving path
    makes. The count is how many ``<|image_pad|>`` tokens the processor expands
    one 448 px fixture view into, which is the pixel budget in effect.
    """
    image = fixture_image()
    messages = chat_prompt(
        FIXTURE_SYSTEM, FIXTURE_LABELS, FIXTURE_USER, images=[image] * len(FIXTURE_LABELS)
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    input_ids = encoded["input_ids"]
    ids = input_ids[0] if isinstance(input_ids[0], list) else input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
    pad = image_token_id(processor)
    per_view, remainder = divmod(sum(1 for token in ids if token == pad), len(FIXTURE_LABELS))
    if remainder:
        raise ValueError("the fixture views did not expand to equal token counts")
    return str(text), per_view


def layout_fingerprint(processor: Any) -> str:
    """sha256 over the fixture's templated text and its per-view image-token count.

    Deterministic — a fixed fixture, no randomness — and cheap: one template
    render and one 448 px image through the image processor, on CPU. Equal
    fingerprints mean the chat template, the layout version and the pixel
    budget all agree between the run that wrote the adapter and the process
    about to serve it. Any one of them differing changes the token stream the
    adapter reads, and shows up here rather than as a worse answer.
    """
    text, tokens_per_view = fixture_render(processor)
    digest = hashlib.sha256()
    digest.update(LAYOUT_VERSION.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tokens_per_view).encode("ascii"))
    return digest.hexdigest()


__all__ = [
    "FIXTURE_LABELS",
    "FIXTURE_SYSTEM",
    "FIXTURE_USER",
    "FIXTURE_VIEW_PX",
    "IMAGE_PAD_TOKEN",
    "LAYOUT_VERSION",
    "ImageBlock",
    "chat_prompt",
    "fixture_image",
    "fixture_render",
    "image_placeholder_count",
    "image_token_id",
    "layout_fingerprint",
    "strip_image_payloads",
    "user_turn_content",
]
