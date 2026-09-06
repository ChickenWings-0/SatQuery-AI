"""SQLite persistence for audit traces (Master.md §6.4 storage row).

A trace is the requirement-5 deliverable, so it outlives the request that
produced it: ``GET /v1/trace/{id}`` has to work after a restart, and a judge has
to be able to ask for the run they saw twenty minutes ago.

The row keeps the whole trace as canonical JSON plus a handful of extracted
columns. Those columns exist to make the common queries — recent traces, traces
for one policy key — indexable without parsing every payload, and nothing reads
them back into the object: the JSON is the record.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from satquery.schemas.trace import AuditTrace

SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id       TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    duration_ms    INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    task           TEXT NOT NULL,
    policy_key     TEXT NOT NULL,
    planner        TEXT NOT NULL,
    confidence     REAL NOT NULL,
    payload        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_created_at ON traces (created_at DESC);
CREATE INDEX IF NOT EXISTS traces_policy_key ON traces (policy_key);
"""


@dataclass(frozen=True)
class TraceSummary:
    """The indexed columns of one stored trace, without its payload."""

    trace_id: str
    created_at: datetime
    duration_ms: int
    task: str
    policy_key: str
    planner: str
    confidence: float


class TraceStore:
    """A SQLite-backed store of :class:`AuditTrace` records."""

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the database at *path*.

        ``:memory:`` is accepted and is what the tests use.
        """
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._memory: sqlite3.Connection | None = None
        if self.path == ":memory:":
            # An in-memory database dies with its connection, so this one is
            # held open for the store's lifetime rather than per operation.
            self._memory = sqlite3.connect(self.path, check_same_thread=False)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection, committing on success."""
        if self._memory is not None:
            yield self._memory
            self._memory.commit()
            return
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def put(self, trace: AuditTrace) -> None:
        """Insert or replace one trace."""
        payload = trace.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO traces (trace_id, created_at, duration_ms, "
                "schema_version, task, policy_key, planner, confidence, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.created_at.isoformat(),
                    trace.duration_ms,
                    trace.schema_version,
                    trace.resolved_task.primary.value,
                    trace.plan.policy_key,
                    trace.plan.planner,
                    trace.confidence.overall,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )

    def get(self, trace_id: str) -> AuditTrace | None:
        """Return one trace, or None when it is unknown."""
        with self._connect() as connection, closing(
            connection.execute("SELECT payload FROM traces WHERE trace_id = ?", (trace_id,))
        ) as cursor:
            row = cursor.fetchone()
        if row is None:
            return None
        payload: dict[str, Any] = json.loads(row[0])
        return AuditTrace.model_validate(payload)

    def recent(self, limit: int = 20) -> list[TraceSummary]:
        """The most recently created traces, newest first."""
        with self._connect() as connection, closing(
            connection.execute(
                "SELECT trace_id, created_at, duration_ms, task, policy_key, planner, "
                "confidence FROM traces ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        ) as cursor:
            rows = cursor.fetchall()
        return [
            TraceSummary(
                trace_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                duration_ms=int(row[2]),
                task=row[3],
                policy_key=row[4],
                planner=row[5],
                confidence=float(row[6]),
            )
            for row in rows
        ]

    def delete(self, trace_id: str) -> bool:
        """Remove one trace. Returns True when a row was deleted."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
            return cursor.rowcount > 0

    def __len__(self) -> int:
        """Number of stored traces."""
        with self._connect() as connection, closing(
            connection.execute("SELECT COUNT(*) FROM traces")
        ) as cursor:
            return int(cursor.fetchone()[0])
