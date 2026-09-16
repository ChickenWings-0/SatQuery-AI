"""The validation benchmark (ROADMAP_REMAINING_FIXES.md, Track 1).

Four small modules, each usable on its own:

* :mod:`~satquery.eval.sampler` — a reproducible, stratified slice of the
  held-out ``val.jsonl``;
* :mod:`~satquery.eval.runner` — batch inference through whichever
  :class:`~satquery.models.loader.VlmBackend` the machine serves, written out
  incrementally so a crashed run resumes where it stopped;
* :mod:`~satquery.eval.scorers` — pure functions from ``(reference,
  prediction, fact sheet)`` to numbers, one family per task type;
* :mod:`~satquery.eval.report` — the aggregate tables the slides are built
  from, in Markdown, JSON and CSV, plus the worst cases as evidence.

The numbers this package produces are the numbers the team presents, so the
scorers reuse the production parsers and validator rather than re-deriving
them: a box is a box if :func:`satquery.models.prompts.box_format.parse`
says so, and a citation is a citation if
:func:`satquery.evidence.citation_validator.validate` binds it.
"""

from __future__ import annotations

__all__: list[str] = []
