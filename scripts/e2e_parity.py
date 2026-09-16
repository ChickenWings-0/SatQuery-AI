#!/usr/bin/env python
"""End-to-end API and DAG parity check against a *live* SatQuery server.

ROADMAP_REMAINING_FIXES.md, Track 3. Three demo scenarios — single-image
grounding, bi-temporal change, cross-modal optical + SAR — plus two negative
cases, each submitted twice: once through the synchronous ``POST /v1/analyze``
and once through ``POST /v1/jobs`` + ``GET /v1/jobs/{id}/events`` (SSE). For
every scenario the script asserts:

* the plan is the policy-table entry the query should select, step by step,
  with any substitution reported as ``fallback_of`` rather than silently
  planned (a missing ``SATQUERY_CD_CHECKPOINT`` turns ``siamese_change_detector``
  into ``image_diff_change`` — the plan must *say* so);
* the SSE stream follows API_CONTRACT §5 — ``queued`` → ``stage``* → ``plan``
  (once) → every planned step started and completed exactly once → ``done``;
* every citation in the answer resolves to a scalar the named step measured,
  and the number in the answer text is the number that was measured;
* the two ``AuditTrace``s are **byte-identical** once the fields that cannot be
  equal — ``trace_id``, ``created_at``, durations, cache hits — are normalised
  away. That is PRODUCT.md's "byte-identical reruns" claim, tested.

The server is whatever ``--base-url`` points at: the bf16 adapter through the
transformers backend, a Q4_K_M GGUF behind llama-server, or ``make demo-cpu``
with the VLM disabled (templated answers). The checks are the same; only the
answer text differs. Nothing here needs torch, so the script runs on the demo
laptop as well as the training box.

Inputs default to the synthetic corpus from :mod:`scripts.make_synthetic_fixtures`
(rendered into a temp directory, no dataset, no network). Point ``--scenes-dir``
at a directory holding real clipped GeoTIFFs with the same file names
(``s2_pre.tif``, ``s2_post.tif``, ``s1_vvvh.tif``, ``benchmark_rgb.png`` …) to
run the same checks over real imagery.

Usage::

    make e2e                                   # starts the API, runs this, stops it
    uv run --no-sync python scripts/e2e_parity.py --base-url http://127.0.0.1:8000
    uv run --no-sync python scripts/e2e_parity.py --scenario change --json out.json
    uv run --no-sync python scripts/e2e_parity.py --scenes-dir data/e2e --verbose

Exit status is the number of failed scenarios, so it composes with ``make``.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import mimetypes
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------- vocabulary

TERMINAL: Final[frozenset[str]] = frozenset({"done", "error"})
STAGES: Final[tuple[str, ...]] = (
    "ingesting",
    "validating",
    "rendering",
    "planning",
    "executing",
    "aggregating",
    "done",
)
"""The §5 stage order; a stream may skip stages but never reorder them."""

VOLATILE_KEYS: Final[frozenset[str]] = frozenset({"created_at", "duration_ms", "cache_hit"})
"""Trace fields that two runs of the same analysis can never share.

``cache_hit`` is here because the second submission legitimately hits the
executor's result cache for deterministic tools; the *scalars* it returns are
then identical by construction, which is exactly what the parity check
demonstrates."""

CITATION_SOURCE: Final[re.Pattern[str]] = re.compile(r"^step:(\d+)/scalars\.([A-Za-z0-9_.]+)$")
"""``Citation.source`` grammar, API_CONTRACT §3.x: one step, one dotted scalar path."""

NUMBER: Final[re.Pattern[str]] = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    """One demo family: what is uploaded, what is asked, what must come back."""

    name: str
    title: str
    files: tuple[str, ...]
    query: str
    policy_key: str
    tools: tuple[str, ...]
    """The policy-table steps in order. An execution may substitute a tool only
    if its ``fallback_of`` names the planned one."""
    options: dict[str, Any] = field(default_factory=dict)
    expect_artifact_types: tuple[str, ...] = ()
    expect_overall: tuple[str, ...] = ("PASS",)
    expect_concurrency: bool = False
    """The plan has independent steps, so at least one pair of them must have
    run concurrently (overlapping start/complete frames) — the executor walks a
    DAG, not a list."""


SCENARIOS: Final[dict[str, Scenario]] = {
    "grounding": Scenario(
        name="grounding",
        title="Single-image grounding (VHR PNG)",
        files=("benchmark_rgb.png",),
        query="Where is the airplane? Highlight it with a bounding box.",
        policy_key="GROUNDING|SINGLE|*",
        tools=("spectral_renderer", "text_grounding", "vlm_vqa"),
        expect_artifact_types=("BBOX_SET",),
        # A benchmark PNG carries no georeference; the checks pass with a
        # MISSING_GEOREFERENCE warning, and the answer must say so too.
        expect_overall=("PASS", "PASS_WITH_WARNINGS"),
    ),
    "change": Scenario(
        name="change",
        title="Bi-temporal change (Sentinel-2 pair)",
        files=("s2_pre.tif", "s2_post.tif"),
        query="What changed between these two images?",
        policy_key="CHANGE_VQA|BI_TEMPORAL|optical",
        tools=(
            "spectral_renderer",
            "siamese_change_detector",
            "change_statistics",
            "spectral_index_analyzer",
            "vlm_change_vqa",
        ),
        expect_artifact_types=("CHANGE_MASK",),
        # spectral_index_analyzer (4) needs only the renderer; it must run
        # while the change detector (2) / change_statistics (3) are busy.
        expect_concurrency=True,
    ),
    "crossmodal": Scenario(
        name="crossmodal",
        title="Cross-modal optical + SAR",
        files=("s2_pre.tif", "s1_vvvh.tif"),
        # "sar"/"optical" in the text would select CROSS_MODAL_COMPARE (rule 30);
        # "together" on a CROSS_MODAL pair selects CROSS_MODAL_VQA (rule 31).
        query="Is this area built-up? Use both sensors together.",
        policy_key="CROSS_MODAL_VQA|CROSS_MODAL|mixed",
        tools=(
            "spectral_renderer",
            "spectral_index_analyzer",
            "sar_backscatter_analyzer",
            "crossmodal_consistency",
            "physics_agreement",
            "vlm_vqa",
        ),
        expect_concurrency=True,
    ),
}

# The negative cases must *degrade or reject identically* on both paths and
# never 5xx. What "identically" means is decided from the synchronous response
# and then demanded of the streaming one.
NEGATIVE: Final[dict[str, Scenario]] = {
    "disjoint": Scenario(
        name="disjoint",
        title="Negative: pair with no spatial overlap",
        files=("s2_pre.tif", "s2_disjoint.tif"),
        query="What changed between these two images?",
        policy_key="",
        tools=(),
        expect_overall=("FAIL",),
    ),
    "uncited": Scenario(
        name="uncited",
        title="Negative: a tool disabled mid-plan must degrade, not error",
        files=("s2_pre.tif",),
        query="Describe this scene and identify water bodies.",
        policy_key="",
        tools=(),
        options={"disable_tools": ["spectral_index_analyzer"]},
        expect_overall=("PASS",),
    ),
}


# ------------------------------------------------------------------- client


class ApiError(Exception):
    """A non-2xx answer, with the §6 error envelope when the server sent one."""

    def __init__(self, status: int, body: str) -> None:
        """Keep the status and parse the body once; the envelope may be absent."""
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}
        self.envelope: dict[str, Any] = parsed if isinstance(parsed, dict) else {}

    @property
    def code(self) -> str | None:
        """The §6 error code, looking through FastAPI's ``detail`` wrapper when present."""
        error = self.envelope.get("error") or self.envelope.get("detail")
        if isinstance(error, dict):
            inner = error.get("error", error)
            code = inner.get("code") if isinstance(inner, dict) else None
            return str(code) if code else None
        return None


class Client:
    """Just enough HTTP for two endpoints and one event stream.

    Standard library only, so the script runs on the demo laptop without the
    ML environment.
    """

    def __init__(self, base_url: str, timeout: float) -> None:
        """Bind to one server; *timeout* bounds every request, a bf16 load included."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- transport ---------------------------------------------------------

    def _open(self, request: urllib.request.Request) -> Any:
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)  # noqa: S310 - loopback API
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise ApiError(error.code, body) from None

    def get_json(self, path: str) -> Any:
        """GET *path* and decode the JSON body."""
        with self._open(urllib.request.Request(self.base_url + path)) as response:
            return json.load(response)

    def post_multipart(
        self, path: str, files: Sequence[Path], fields: dict[str, str]
    ) -> tuple[int, Any]:
        """POST *files* as ``images`` plus *fields* (hand-framed multipart); (status, body)."""
        boundary = f"----satquery-e2e-{uuid.uuid4().hex}"
        body = bytearray()
        for name, value in fields.items():
            body += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        for path_ in files:
            mime = mimetypes.guess_type(path_.name)[0] or "application/octet-stream"
            if path_.suffix.lower() in {".tif", ".tiff"}:
                mime = "image/tiff"
            body += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="images"; '
                f'filename="{path_.name}"\r\nContent-Type: {mime}\r\n\r\n'
            ).encode()
            body += path_.read_bytes() + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with self._open(request) as response:
            return response.status, json.load(response)

    def events(self, path: str) -> Iterator[tuple[int, str, dict[str, Any]]]:
        """Yield ``(id, event, data)`` frames until the stream closes; drop heartbeats."""
        request = urllib.request.Request(
            self.base_url + path, headers={"Accept": "text/event-stream"}
        )
        with self._open(request) as response:
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("text/event-stream"):
                raise ApiError(response.status, f"not an event stream: {content_type}")
            seq: int | None = None
            name: str | None = None
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line or line.startswith(":"):
                    continue
                if line.startswith("id: "):
                    seq = int(line.removeprefix("id: "))
                elif line.startswith("event: "):
                    name = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    if name is None or seq is None:
                        raise AssertionError("SSE data frame arrived before its id/event lines")
                    yield seq, name, json.loads(line.removeprefix("data: "))
                    seq = name = None

    # -- the two submission paths -------------------------------------------

    def analyze(self, files: Sequence[Path], query: str, options: dict[str, Any]) -> dict[str, Any]:
        """``POST /v1/analyze`` and return the response body."""
        fields = {"query": query}
        if options:
            fields["options"] = json.dumps(options)
        status, body = self.post_multipart("/v1/analyze", files, fields)
        if status != 200:
            raise ApiError(status, json.dumps(body))
        return body

    def submit_job(
        self, files: Sequence[Path], query: str, options: dict[str, Any]
    ) -> dict[str, Any]:
        """``POST /v1/jobs`` and return the 202 body."""
        fields = {"query": query}
        if options:
            fields["options"] = json.dumps(options)
        status, body = self.post_multipart("/v1/jobs", files, fields)
        if status != 202:
            raise ApiError(status, json.dumps(body))
        return body


# ------------------------------------------------------------------ checks


class Check:
    """A named assertion list.

    Failures are collected, not raised, so one scenario reports everything
    that is wrong with it at once.
    """

    def __init__(self) -> None:
        """Start with nothing asserted."""
        self.passed: list[str] = []
        self.failed: list[str] = []

    def ok(self, condition: bool, label: str, detail: str = "") -> bool:
        """Record *label* as passed or failed (with *detail*); return *condition*."""
        if condition:
            self.passed.append(label)
        else:
            self.failed.append(f"{label}{(' — ' + detail) if detail else ''}")
        return condition


def normalise(trace: dict[str, Any], trace_id: str) -> dict[str, Any]:
    """Strip what two honest runs cannot share, then the rest must be equal."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in VOLATILE_KEYS}
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str):
            return node.replace(trace_id, "<trace_id>")
        return node

    return walk(copy.deepcopy(trace))


def canonical(obj: Any) -> str:
    """One stable serialisation, so two traces can be compared as text."""
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, default=str)


def check_plan(check: Check, trace: dict[str, Any], scenario: Scenario) -> None:
    """The plan is the policy-table entry; every step ran it or a declared fallback."""
    plan = trace["plan"]
    check.ok(
        plan["policy_key"] == scenario.policy_key,
        "plan.policy_key is the expected policy-table entry",
        f"got {plan['policy_key']!r}, expected {scenario.policy_key!r}",
    )
    planned = [step["tool"] for step in plan["steps"]]
    check.ok(
        planned == list(scenario.tools),
        "plan.steps are the policy-table tools in order",
        f"got {planned}, expected {list(scenario.tools)}",
    )
    numbers = [step["step"] for step in plan["steps"]]
    check.ok(numbers == list(range(1, len(numbers) + 1)), "plan steps are numbered 1..n")
    check.ok(
        all(d < step["step"] for step in plan["steps"] for d in step.get("depends_on", [])),
        "every depends_on points at an earlier step (topological order)",
    )

    executions = {execution["step"]: execution for execution in trace["executions"]}
    check.ok(
        sorted(executions) == numbers,
        "every planned step has exactly one execution",
        f"executed {sorted(executions)}",
    )
    for step in plan["steps"]:
        execution = executions.get(step["step"])
        if execution is None:
            continue
        same = execution["tool"] == step["tool"]
        substituted = execution.get("fallback_of") == step["tool"]
        check.ok(
            same or substituted,
            f"step {step['step']} ran {step['tool']} or a declared fallback",
            f"ran {execution['tool']!r} with fallback_of={execution.get('fallback_of')!r}",
        )
        if substituted:
            check.passed.append(
                f"  ↳ step {step['step']}: {step['tool']} → {execution['tool']} (declared fallback)"
            )
        check.ok(
            execution["status"] in {"OK", "DEGRADED", "SKIPPED", "FAILED", "CACHED"},
            f"step {step['step']} has a contract status",
            execution["status"],
        )


def check_compatibility(check: Check, trace: dict[str, Any], scenario: Scenario) -> None:
    """The check battery ran and its verdict is the one the scenario expects."""
    report = trace["compatibility"]
    checks = report["checks"]
    names = [c["name"] if isinstance(c, dict) and "name" in c else c.get("check") for c in checks]
    check.ok(len(checks) >= 1, "compatibility report lists its checks", f"{len(checks)} checks")
    check.ok(
        report["overall"] in scenario.expect_overall,
        f"compatibility.overall is one of {'/'.join(scenario.expect_overall)}",
        f"got {report['overall']!r}; failed: "
        + ", ".join(
            str(n) for n, c in zip(names, checks, strict=True) if c.get("status") == "FAIL"
        ),
    )
    if len(scenario.files) == 2:
        check.ok(
            len(checks) >= 8,
            "a pair runs the full check battery",
            f"only {len(checks)}: {names}",
        )


def check_answer(check: Check, trace: dict[str, Any]) -> None:
    """Every number in the answer is bound to a measurement the trace contains."""
    answer = trace["answer"]
    check.ok(bool(answer["text"].strip()), "answer.text is non-empty")
    check.ok(
        not answer.get("uncited_numeric_spans"),
        "no uncited numeric spans in the answer",
        f"uncited: {answer.get('uncited_numeric_spans')}",
    )
    executions = {execution["step"]: execution for execution in trace["executions"]}
    fact_sheet = trace.get("fact_sheet", {})
    for citation in answer.get("citations", []):
        for source in citation["source"].split("|"):
            match = CITATION_SOURCE.match(source)
            if not check.ok(match is not None, "citation.source follows the grammar", source):
                continue
            assert match is not None
            step, path = int(match.group(1)), match.group(2)
            execution = executions.get(step)
            if not check.ok(execution is not None, f"citation step {step} exists", source):
                continue
            assert execution is not None
            scalars = execution.get("scalars", {})
            measured = scalars.get(path)
            if measured is None:
                # Some tools namespace their scalars; the fact sheet is the
                # flattened view the aggregator actually cited from.
                measured = fact_sheet.get(f"{execution['tool']}.{path}")
            if not check.ok(
                measured is not None,
                f"citation {source} names a measured scalar",
                f"step {step} ({execution['tool']}) scalars: {sorted(scalars)}",
            ):
                continue
            if isinstance(measured, int | float) and isinstance(citation["value"], int | float):
                check.ok(
                    abs(float(measured) - float(citation["value"]))
                    <= 1e-6 * max(1.0, abs(float(measured))),
                    f"cited value equals the measurement for {source}",
                    f"cited {citation['value']} vs measured {measured}",
                )
            check.ok(
                citation["claim"] in answer["text"],
                "citation.claim is a literal span of the answer",
                repr(citation["claim"]),
            )


def check_artifacts(check: Check, trace: dict[str, Any], scenario: Scenario) -> None:
    """The expected evidence types exist and are addressed under this trace."""
    types = {artifact["type"] for artifact in trace["artifacts"]}
    check.ok(bool(types), "the run produced artifacts", "none")
    for expected in scenario.expect_artifact_types:
        check.ok(expected in types, f"a {expected} artifact was produced", f"have {sorted(types)}")
    for artifact in trace["artifacts"]:
        if artifact.get("url"):
            check.ok(
                artifact["url"].startswith(f"/v1/artifacts/{trace['trace_id']}/"),
                f"artifact {artifact['id']} url is addressed by trace id",
                artifact["url"],
            )


def check_events(
    check: Check, events: list[tuple[int, str, dict[str, Any]]], scenario: Scenario
) -> dict[str, Any] | None:
    """API_CONTRACT §5, asserted frame by frame. Returns the ``done`` payload."""
    ids = [seq for seq, _, _ in events]
    names = [name for _, name, _ in events]
    check.ok(ids == list(range(len(ids))), "SSE ids are the dense sequence 0..n-1", str(ids[:8]))
    check.ok(bool(names) and names[0] == "queued", "stream opens with queued", str(names[:3]))
    check.ok(bool(names) and names[-1] in TERMINAL, "stream closes on done|error", str(names[-3:]))
    check.ok(sum(n in TERMINAL for n in names) == 1, "exactly one terminal event")
    if "plan" not in names:
        check.ok(False, "a plan event was emitted", str(names))
        return None
    plan_at = names.index("plan")
    check.ok(names.count("plan") == 1, "plan is emitted exactly once")
    check.ok(
        set(names[1:plan_at]) <= {"stage"},
        "only stage narration precedes the plan",
        str(names[:plan_at]),
    )

    stages = [data["stage"] for _, name, data in events if name == "stage"]
    order = [STAGES.index(s) for s in stages if s in STAGES]
    check.ok(order == sorted(order), "stages arrive in contract order", str(stages))
    pcts = [data["pct"] for _, name, data in events if name == "stage"]
    check.ok(pcts == sorted(pcts), "stage pct is monotonic", str(pcts))

    for streaming in ("step_started", "step_completed", "artifact", "answer_delta"):
        if streaming in names:
            check.ok(names.index(streaming) > plan_at, f"{streaming} follows the plan")

    plan = next(data for _, name, data in events if name == "plan")
    planned = [step["step"] for step in plan["steps"]]
    started = [data["step"] for _, name, data in events if name == "step_started"]
    completed = [data["step"] for _, name, data in events if name == "step_completed"]
    check.ok(
        sorted(started) == planned, "every planned step started once", f"{started} vs {planned}"
    )
    check.ok(
        sorted(completed) == planned,
        "every planned step completed once",
        f"{completed} vs {planned}",
    )

    # A step completes only after it started, and only after every dependency
    # completed — the stream is the DAG being walked, not a log being replayed.
    position = {
        (name, data.get("step")): i
        for i, (_, name, data) in enumerate(events)
        if name.startswith("step_")
    }
    dependencies = {step["step"]: step.get("depends_on", []) for step in plan["steps"]}
    for step in planned:
        s, c = position.get(("step_started", step)), position.get(("step_completed", step))
        if s is None or c is None:
            continue
        check.ok(s < c, f"step {step} started before it completed")
        for dependency in dependencies[step]:
            dc = position.get(("step_completed", dependency))
            check.ok(
                dc is not None and dc < s,
                f"step {step} started only after dependency {dependency} completed",
            )

    if scenario.expect_concurrency:
        # Transitive closure of depends_on: two steps are independent when
        # neither is an ancestor of the other. Of those pairs, at least one
        # must have overlapping [started, completed) frame windows.
        ancestors: dict[int, set[int]] = {}
        for step in planned:
            closure: set[int] = set()
            stack = list(dependencies[step])
            while stack:
                dependency = stack.pop()
                if dependency not in closure:
                    closure.add(dependency)
                    stack.extend(dependencies[dependency])
            ancestors[step] = closure
        windows = {
            step: (position[("step_started", step)], position[("step_completed", step)])
            for step in planned
            if ("step_started", step) in position and ("step_completed", step) in position
        }
        independent = [
            (a, b)
            for a in windows
            for b in windows
            if a < b and b not in ancestors[a] and a not in ancestors[b]
        ]
        overlapping = [
            (a, b)
            for a, b in independent
            if windows[a][0] < windows[b][1] and windows[b][0] < windows[a][1]
        ]
        check.ok(
            bool(overlapping),
            "independent steps ran concurrently (DAG, not a list)",
            f"independent pairs {independent} never overlapped; windows {windows}",
        )

    done = next((data for _, name, data in events if name == "done"), None)
    if done is not None:
        check.ok(done.get("trace") is not None, "done carries the full trace (include_trace=true)")
    return done


def check_planned_tools_match_plan_event(
    check: Check, events: list[tuple[int, str, dict[str, Any]]], trace: dict[str, Any]
) -> None:
    """What was streamed (plan, artifacts) is what the final trace records."""
    plan = next((data for _, name, data in events if name == "plan"), None)
    if plan is None:
        return
    check.ok(
        [s["tool"] for s in plan["steps"]] == [s["tool"] for s in trace["plan"]["steps"]],
        "the plan event and the trace's plan name the same tools",
    )
    streamed = [data["id"] for _, name, data in events if name == "artifact"]
    check.ok(
        streamed == [artifact["id"] for artifact in trace["artifacts"]],
        "streamed artifact ids equal the final artifact list, in order",
        f"{streamed} vs {[a['id'] for a in trace['artifacts']]}",
    )


# ---------------------------------------------------------------- scenarios


@dataclass
class Outcome:
    """Everything one scenario produced, for the console and the JSON report."""

    scenario: str
    title: str
    passed: list[str]
    failed: list[str]
    sync_trace_id: str | None = None
    job_trace_id: str | None = None
    answer: str | None = None
    diff: str | None = None
    wall_ms: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return not self.failed


def run_positive(
    client: Client, scenario: Scenario, scenes: dict[str, Path], verbose: bool
) -> Outcome:
    """Submit one demo scenario through both paths and assert everything about it."""
    check = Check()
    files = [scenes[name] for name in scenario.files]
    options = {"include_trace": True, "include_rendered_views": True, **scenario.options}
    wall: dict[str, int] = {}

    # -- synchronous ----------------------------------------------------------
    started = time.perf_counter()
    try:
        sync = client.analyze(files, scenario.query, options)
    except ApiError as error:
        check.ok(False, "POST /v1/analyze returned 200", str(error))
        return Outcome(scenario.name, scenario.title, check.passed, check.failed)
    wall["analyze_ms"] = int((time.perf_counter() - started) * 1000)
    sync_trace = sync.get("trace")
    if not check.ok(sync_trace is not None, "/v1/analyze response carries the trace"):
        return Outcome(scenario.name, scenario.title, check.passed, check.failed)
    assert sync_trace is not None
    check.ok(sync["trace_id"] == sync_trace["trace_id"], "response.trace_id == trace.trace_id")

    # -- streaming ------------------------------------------------------------
    started = time.perf_counter()
    try:
        accepted = client.submit_job(files, scenario.query, options)
    except ApiError as error:
        check.ok(False, "POST /v1/jobs returned 202", str(error))
        return Outcome(
            scenario.name, scenario.title, check.passed, check.failed, sync_trace["trace_id"]
        )
    job_id = accepted["job_id"]
    check.ok(
        accepted["events_url"] == f"/v1/jobs/{job_id}/events", "202 body addresses the event stream"
    )
    events = list(client.events(accepted["events_url"]))
    wall["jobs_ms"] = int((time.perf_counter() - started) * 1000)
    done = check_events(check, events, scenario)
    if done is None or done.get("trace") is None:
        error = next((data for _, name, data in events if name == "error"), None)
        check.ok(False, "the job produced a done event with a trace", json.dumps(error)[:300])
        return Outcome(
            scenario.name,
            scenario.title,
            check.passed,
            check.failed,
            sync_trace["trace_id"],
            job_id,
        )
    job_trace = done["trace"]
    check.ok(job_trace["trace_id"] == job_id, "the job id is the trace id (§4.2)")

    # -- the stored trace is the streamed trace ------------------------------
    stored = client.get_json(f"/v1/traces/{job_id}")
    check.ok(
        canonical(stored) == canonical(job_trace),
        "GET /v1/traces/{id} returns the streamed trace verbatim",
    )
    polled = client.get_json(f"/v1/jobs/{job_id}")
    check.ok(
        polled["status"] == "succeeded" and polled["pct"] == 100, "poll reports succeeded/100 %"
    )

    # -- the same assertions on both traces ------------------------------------
    for label, trace in (("sync", sync_trace), ("stream", job_trace)):
        before = len(check.failed)
        check_plan(check, trace, scenario)
        check_compatibility(check, trace, scenario)
        check_answer(check, trace)
        check_artifacts(check, trace, scenario)
        if len(check.failed) > before:
            check.failed[before:] = [f"[{label}] {f}" for f in check.failed[before:]]
    check_planned_tools_match_plan_event(check, events, job_trace)

    # -- byte-identical --------------------------------------------------------
    left = canonical(normalise(sync_trace, sync_trace["trace_id"]))
    right = canonical(normalise(job_trace, job_id))
    diff = None
    if not check.ok(
        left == right, "normalised traces are byte-identical across /v1/analyze and /v1/jobs"
    ):
        diff = "\n".join(
            difflib.unified_diff(
                left.splitlines(), right.splitlines(), "analyze", "jobs", lineterm="", n=2
            )
        )
        if verbose:
            print(diff)
    check.ok(
        sync["answer"]["text"] == done["answer"]["text"],
        "answer text is identical on both paths",
        f"{sync['answer']['text'][:80]!r} vs {done['answer']['text'][:80]!r}",
    )

    return Outcome(
        scenario.name,
        scenario.title,
        check.passed,
        check.failed,
        sync_trace["trace_id"],
        job_id,
        sync["answer"]["text"],
        diff,
        wall,
    )


def run_negative(
    client: Client, scenario: Scenario, scenes: dict[str, Path], verbose: bool
) -> Outcome:
    """Both paths must agree on how an input is refused or degraded, and neither may 5xx."""
    check = Check()
    files = [scenes[name] for name in scenario.files]
    options = {"include_trace": True, **scenario.options}

    sync_status: int | str
    try:
        sync = client.analyze(files, scenario.query, options)
        sync_status = 200
        sync_code = None
    except ApiError as error:
        sync, sync_status, sync_code = {}, error.status, error.code
    check.ok(
        isinstance(sync_status, int) and sync_status < 500,
        "/v1/analyze never 5xx",
        str(sync_status),
    )

    try:
        accepted = client.submit_job(files, scenario.query, options)
        events = list(client.events(accepted["events_url"]))
        job_status: int = 202
        job_code = next((data.get("code") for _, name, data in events if name == "error"), None)
        done = next((data for _, name, data in events if name == "done"), None)
    except ApiError as error:
        events, job_status, job_code, done = [], error.status, error.code, None
    check.ok(job_status < 500, "/v1/jobs never 5xx", str(job_status))

    if sync_status != 200:
        # Rejected in-request on the sync path → must be rejected in-request on
        # the jobs path with the same status and error code (no phantom job).
        check.ok(
            job_status == sync_status,
            "both paths reject with the same HTTP status",
            f"analyze {sync_status} vs jobs {job_status}",
        )
        check.ok(
            job_code == sync_code,
            "both paths reject with the same error code",
            f"{sync_code} vs {job_code}",
        )
        check.ok(sync_code is not None, "the rejection carries a §6 error code", str(sync_code))
        return Outcome(
            scenario.name,
            scenario.title,
            check.passed,
            check.failed,
            answer=f"rejected: {sync_code}",
        )

    # Accepted on the sync path → the job must run to done (never error) with
    # the same degraded shape.
    check.ok(
        done is not None and job_code is None, "the job ended in done, not error", str(job_code)
    )
    sync_trace = sync.get("trace") or {}
    job_trace = (done or {}).get("trace") or {}
    if sync_trace and job_trace:
        for label, trace in (("sync", sync_trace), ("stream", job_trace)):
            statuses = sorted({e["status"] for e in trace["executions"]})
            check.ok(
                bool(set(statuses) & {"DEGRADED", "SKIPPED", "FAILED"})
                or trace["compatibility"]["overall"] != "PASS",
                f"[{label}] the run is visibly degraded or flagged",
                f"statuses {statuses}, overall {trace['compatibility']['overall']}",
            )
            check.ok(bool(trace["answer"]["text"]), f"[{label}] a degraded run still answers")
        left = canonical(normalise(sync_trace, sync_trace["trace_id"]))
        right = canonical(normalise(job_trace, job_trace["trace_id"]))
        check.ok(left == right, "normalised degraded traces are byte-identical across paths")
        check.ok(
            sync_trace["compatibility"]["overall"] in scenario.expect_overall,
            f"compatibility.overall is one of {'/'.join(scenario.expect_overall)}",
            sync_trace["compatibility"]["overall"],
        )
    return Outcome(
        scenario.name,
        scenario.title,
        check.passed,
        check.failed,
        sync_trace.get("trace_id"),
        job_trace.get("trace_id"),
        (sync.get("answer") or {}).get("text"),
    )


# ------------------------------------------------------------------ scenes


def resolve_scenes(scenes_dir: Path | None, needed: set[str]) -> dict[str, Path]:
    """Real scenes from ``--scenes-dir`` when given, else the synthetic corpus."""
    if scenes_dir is not None:
        found = {name: scenes_dir / name for name in needed}
        missing = [name for name, path in found.items() if not path.exists()]
        if missing:
            raise SystemExit(f"--scenes-dir {scenes_dir} is missing: {', '.join(missing)}")
        return found
    from scripts.make_synthetic_fixtures import build_fixtures  # rasterio; server-side dep

    output = Path(tempfile.mkdtemp(prefix="satquery-e2e-"))
    built = {scene.path.name: scene.path for scene in build_fixtures(output)}
    missing = [name for name in needed if name not in built]
    if missing:
        raise SystemExit(f"synthetic corpus lacks: {', '.join(missing)}")
    return {name: built[name] for name in needed}


# ------------------------------------------------------------------ report


def print_outcome(outcome: Outcome, verbose: bool) -> None:
    """One block per scenario: verdict, ids, answer, and every failure."""
    mark = "PASS" if outcome.ok else "FAIL"
    print(f"\n[{mark}] {outcome.title}")
    if outcome.sync_trace_id or outcome.job_trace_id:
        print(
            f"       analyze={outcome.sync_trace_id}  jobs={outcome.job_trace_id}  "
            f"{outcome.wall_ms or ''}"
        )
    if outcome.answer:
        print(f"       answer: {outcome.answer[:140]!r}")
    if verbose:
        for line in outcome.passed:
            print(f"   ✓ {line}")
    for line in outcome.failed:
        print(f"   ✗ {line}")
    if outcome.diff and not verbose:
        head = outcome.diff.splitlines()[:40]
        print("   " + "\n   ".join(head))
        if len(outcome.diff.splitlines()) > 40:
            print("   … (run with --verbose for the full diff)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[*SCENARIOS, *NEGATIVE, "all", "positive", "negative"],
        help="Repeatable. Default: all.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Real GeoTIFFs named like the synthetic corpus.",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the full outcome report here."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-request timeout in seconds (a bf16 load is slow).",
    )
    parser.add_argument(
        "--wait", type=float, default=120.0, help="Seconds to wait for /v1/health before giving up."
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def wait_for_health(client: Client, seconds: float) -> dict[str, Any]:
    """Poll ``/v1/health`` until it answers or *seconds* run out."""
    deadline = time.monotonic() + seconds
    last: str = ""
    while time.monotonic() < deadline:
        try:
            return client.get_json("/v1/health")
        except (ApiError, urllib.error.URLError, OSError) as error:
            last = str(error)
            time.sleep(1.0)
    raise SystemExit(f"{client.base_url}/v1/health not reachable after {seconds:.0f}s: {last}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected scenarios; the exit status is the number that failed."""
    args = parse_args(argv)
    selected = set(args.scenario or ["all"])
    positives = (
        list(SCENARIOS)
        if selected & {"all", "positive"}
        else [s for s in SCENARIOS if s in selected]
    )
    negatives = (
        list(NEGATIVE) if selected & {"all", "negative"} else [s for s in NEGATIVE if s in selected]
    )

    client = Client(args.base_url, timeout=args.timeout)
    health = wait_for_health(client, args.wait)
    print(
        f"server {args.base_url}  status={health.get('status')}  "
        f"tools={health.get('tools_available')}/{health.get('tools_total')}"
    )

    needed = {name for key in positives for name in SCENARIOS[key].files}
    needed |= {name for key in negatives for name in NEGATIVE[key].files}
    scenes = resolve_scenes(args.scenes_dir, needed)

    outcomes: list[Outcome] = []
    for key in positives:
        outcome = run_positive(client, SCENARIOS[key], scenes, args.verbose)
        print_outcome(outcome, args.verbose)
        outcomes.append(outcome)
    for key in negatives:
        outcome = run_negative(client, NEGATIVE[key], scenes, args.verbose)
        print_outcome(outcome, args.verbose)
        outcomes.append(outcome)

    failed = [o for o in outcomes if not o.ok]
    print(f"\n{len(outcomes) - len(failed)}/{len(outcomes)} scenarios passed")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "base_url": args.base_url,
                    "health": health,
                    "scenes": {k: str(v) for k, v in scenes.items()},
                    "outcomes": [o.__dict__ for o in outcomes],
                },
                indent=2,
                default=str,
            )
        )
        print(f"report → {args.json}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
