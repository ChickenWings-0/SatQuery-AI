"""A processor stand-in for the layout, token-budget and fingerprint tests.

Renders the Qwen chat layout as text, splits special tokens the way the real
tokenizer does, and expands every image block into a configurable number of
``<|image_pad|>`` tokens — the two properties the layout fingerprint and the
token-budget guard actually depend on. Nothing here needs transformers.
"""

from __future__ import annotations

import re
from typing import Any

from satquery.models.prompts import layout

_SPECIAL = re.compile(r"(<\|[a-z_]+\|>)")


class FakeTokenizer:
    """Whitespace tokens, special tokens split out, ``<|image_pad|>`` looked up."""

    IMAGE_PAD = 151655

    def convert_tokens_to_ids(self, token: str) -> int:
        """Only the image pad has a stable id; everything else is 1."""
        return self.IMAGE_PAD if token == layout.IMAGE_PAD_TOKEN else 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        """Tokenise: one id per special token, one per whitespace word."""
        ids: list[int] = []
        for chunk in _SPECIAL.split(text):
            if _SPECIAL.fullmatch(chunk):
                ids.append(self.IMAGE_PAD if chunk == layout.IMAGE_PAD_TOKEN else 1)
            else:
                # Newlines are tokens of their own, as they are for Qwen: the
                # trailing "\n" after <|im_end|> is a supervised position.
                words = chunk.replace("\n", " \n ").split(" ")
                ids.extend(hash(word) % 1000 + 2 for word in words if word)
        return {"input_ids": ids}


class FakeProcessor:
    """Renders the Qwen layout as text and expands every image to N pad tokens.

    Enough of ``apply_chat_template`` for :func:`layout.layout_fingerprint` and
    the token-budget guard, and nothing else.
    """

    def __init__(self, tokens_per_view: int = 144, template_tag: str = "v1") -> None:
        """Choose the pixel budget (tokens per view) and a template identity."""
        self.tokens_per_view = tokens_per_view
        self.template_tag = template_tag
        self.tokenizer = FakeTokenizer()

    def _render(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        lines = [f"<template {self.template_tag}>"]
        for message in messages:
            lines.append(f"<|im_start|>{message['role']}")
            for part in message["content"]:
                if part.get("type") == "image" or "image" in part:
                    lines.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif part.get("text") is not None:
                    lines.append(str(part["text"]))
            lines.append("<|im_end|>")
        if add_generation_prompt:
            lines.append("<|im_start|>assistant")
        return "\n".join(lines)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        **_: Any,
    ) -> Any:
        """Render, and optionally tokenise with every image expanded to N pads."""
        text = self._render(messages, add_generation_prompt)
        if not tokenize:
            return text
        ids: list[int] = []
        for token in self.tokenizer(text)["input_ids"]:
            repeat = self.tokens_per_view if token == FakeTokenizer.IMAGE_PAD else 1
            ids.extend([token] * repeat)
        return {"input_ids": [ids]} if return_dict else [ids]



__all__ = ["FakeProcessor", "FakeTokenizer"]
