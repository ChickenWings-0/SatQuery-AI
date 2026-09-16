"""A reproducible, stratified slice of the held-out corpus.

``val.jsonl`` is 5,902 rows and a full pass at four to five seconds a sample
is most of a working day. The benchmark takes ``per_source`` rows from each
source instead, and it takes them **round-robin across the source's task
types** so that 100 BigEarthNet rows are not 90 ``SCENE_CLASSIFY`` and ten of
everything else. The chosen ids are written next to the results, and a later
run — the un-adapted baseline in particular — can be pinned to exactly that
slice with :func:`select_ids`, which is what makes the "before" and "after"
columns of one table comparable.

Rows are parsed as :class:`~satquery.training.corpus_builder.CorpusSample`,
the same model the trainer read, so what the runner sends is byte-for-byte
what training saw (``tests/integration/test_train_serve_parity.py``).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from satquery.training.corpus_builder import CorpusSample

__all__ = ["load_samples", "read_ids", "select_ids", "stratified_sample", "write_ids"]


def load_samples(path: Path) -> Iterator[CorpusSample]:
    """Yield every row of a corpus JSONL file, validated.

    Raises:
        FileNotFoundError: *path* does not exist.
        pydantic.ValidationError: A row is not a valid corpus sample.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield CorpusSample.model_validate_json(line)


def stratified_sample(
    samples: Iterable[CorpusSample], per_source: int, seed: int
) -> list[CorpusSample]:
    """Pick up to *per_source* rows per source, balanced across task types.

    Within each source the rows are grouped by task, each group is shuffled
    with its own seeded generator, and the groups are drained round-robin
    until the source's quota is met or every group is empty. The result is
    ordered by source name, then by the draw order, so the same inputs and
    seed give the same list on any machine.

    Args:
        samples: The corpus rows to draw from.
        per_source: Rows to take from each source; a source with fewer rows
            contributes all of them.
        seed: The shuffle seed.

    Returns:
        The chosen rows, at most ``per_source`` per source.
    """
    by_source: dict[str, dict[str, list[CorpusSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        by_source[sample.source.value][sample.task.value].append(sample)

    chosen: list[CorpusSample] = []
    for source in sorted(by_source):
        groups = by_source[source]
        queues: dict[str, list[CorpusSample]] = {}
        for task in sorted(groups):
            rows = list(groups[task])
            random.Random(f"{seed}:{source}:{task}").shuffle(rows)
            queues[task] = rows
        taken = 0
        while taken < per_source and any(queues.values()):
            for task in sorted(queues):
                if taken >= per_source:
                    break
                if queues[task]:
                    chosen.append(queues[task].pop())
                    taken += 1
    return chosen


def select_ids(samples: Iterable[CorpusSample], ids: Sequence[str]) -> list[CorpusSample]:
    """Return the rows whose ``id`` is in *ids*, in the order *ids* lists them.

    Raises:
        KeyError: An id in *ids* is not in *samples* — a baseline run pinned
            to a slice the corpus no longer contains is not comparable, so it
            is refused rather than silently shortened.
    """
    wanted = set(ids)
    found = {sample.id: sample for sample in samples if sample.id in wanted}
    missing = [sample_id for sample_id in ids if sample_id not in found]
    if missing:
        raise KeyError(f"{len(missing)} pinned ids are not in the corpus, e.g. {missing[0]!r}")
    return [found[sample_id] for sample_id in ids]


def write_ids(samples: Sequence[CorpusSample], path: Path) -> None:
    """Persist the slice as ``{"ids": [...], "per_source": {...}}``."""
    per_source: dict[str, int] = defaultdict(int)
    for sample in samples:
        per_source[sample.source.value] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ids": [sample.id for sample in samples],
                "per_source": dict(sorted(per_source.items())),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


def read_ids(path: Path) -> list[str]:
    """Read the ids a previous run pinned."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload["ids"] if isinstance(payload, dict) else payload
    return [str(sample_id) for sample_id in ids]
