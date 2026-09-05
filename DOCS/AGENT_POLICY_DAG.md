# SatQuery AI — Agentic Controller: Classification, Policy Table & Execution Semantics (FROZEN)

**Status:** FROZEN as of Phase 0 (schemas) / implemented in Phase 3.
**Authority:** Derived strictly from `DOCS/Master.md` §3, §3.1, §4.4, §5, §8-Phase-3. Companion to `DOCS/API_CONTRACT.md`.
**Scope:** Everything between "a validated set of images + a query" and "a populated `AuditTrace`".

> **Governing constraint (Master.md §3.1):** control flow is **deterministic**. The LLM performs *slot filling* and *answer synthesis* only. It never chooses a tool, never orders steps, and never invents a capability. Byte-identical reruns of the plan are a hard requirement, because the observable trace is what the rubric scores.

---

## 1. The Routing Key

Every plan is selected by a single composite key:

```
policy_key = "{TaskType}|{PairType}|{ModalityKey}"
```

`ModalityKey` is derived, not free:

| PairType | ModalityKey |
|---|---|
| `SINGLE` | the single image's `modality` (`optical` \| `sar` \| `panchromatic`) |
| `BI_TEMPORAL` | the **shared** modality of both images; if they differ, the pair is re-classified `CROSS_MODAL` |
| `CROSS_MODAL` | the literal string `mixed` |

Lookup order (first hit wins):

1. Exact key.
2. Wildcard on modality: `"{TaskType}|{PairType}|*"`.
3. Generic fallback: `"*|*|*"` — sets `plan.planner = "fallback_generic_v1"` and adds warning `GENERIC_PLAN_USED`.

A key that resolves only at step 3 is **visibly flagged in the trace**. This is deliberate: a judge should be able to see when the router was confident and when it was guessing.

---

## 2. Task Classification

A four-stage cascade. Each stage may terminate the cascade.

### Stage A — PairType gate

Narrow the candidate `TaskType` set using the table in `API_CONTRACT.md` §2.1. A `SINGLE` input can never produce `CHANGE_*`; a `BI_TEMPORAL` input can never produce `CROSS_MODAL_*`. This gate alone removes most misclassification risk and costs nothing.

### Stage B — Rule layer (`rules_v1`)

Ordered, case-insensitive regex over the normalised query. First match within the gated candidate set wins.

| Pri | Pattern (abridged) | TaskType | conf |
|---|---|---|---|
| 10 | `\b(map\|mask\|highlight\|where exactly\|show the area)\b` **and** change-word | `CHANGE_MAP` | 0.90 |
| 11 | `\b(describe\|summari[sz]e\|caption\|what happened\|narrate)\b` **and** change-word | `CHANGE_CAPTION` | 0.90 |
| 12 | change-word = `\b(chang(e\|ed\|es)\|differ\|appear\|disappear\|new\|removed\|lost\|gained\|grew\|expand\|shrink\|before and after\|between the two)\b` | `CHANGE_VQA` | 0.92 |
| 20 | `\b(how many\|count\|number of\|total of)\b` | `COUNT` | 0.93 |
| 21 | `\b(segment\|delineate\|outline\|extent of\|boundary of)\b` | `SEGMENTATION` | 0.88 |
| 22 | `\b(where\|locate\|find\|point (to\|out)\|bounding box\|highlight)\b` | `GROUNDING` | 0.88 |
| 30 | `\b(sar\|radar\|backscatter\|microwave)\b` **and** `\b(optical\|spectral\|visible\|multispectral)\b` | `CROSS_MODAL_COMPARE` | 0.91 |
| 31 | `\b(complement\|combin\|together\|both sensors\|fuse)\b` and PairType=CROSS_MODAL | `CROSS_MODAL_VQA` | 0.86 |
| 40 | `\b(describe\|caption\|summari[sz]e\|overview\|what does this (image )?show)\b` | `CAPTION` | 0.90 |
| 50 | `\b(classif\|land ?cover\|land ?use\|what (type\|kind) of (land\|terrain\|area\|region))\b` | `SCENE_CLASSIFY` | 0.85 |
| 99 | default | `VQA` (SINGLE) / `CROSS_MODAL_VQA` (CROSS_MODAL) / `CHANGE_VQA` (BI_TEMPORAL) | 0.70 |

Priority ordering is load-bearing: `CHANGE_MAP` and `CHANGE_CAPTION` are checked before `CHANGE_VQA` because their patterns are strict subsets of the change space; `COUNT` before `GROUNDING` because "how many buildings are there" contains a grounding-shaped noun phrase.

### Stage C — Embedding kNN (`embed_v1`)

Runs only if Stage B confidence `< 0.75`. Sentence-embedding kNN (k=5, cosine) over a committed labelled seed set of ~200 queries at `configs/query_seed_set.jsonl`. Confidence = mean similarity of the winning class. Terminates the cascade if `>= 0.70`.

### Stage D — LLM slot-fill (`llm_slotfill_v1`)

**Always runs** — it is the only source of slot values. It is given the candidate set from Stage A and must return JSON conforming to §3, via constrained/structured decoding.

It may additionally *propose* a `TaskType` only when Stage B+C produced `< 0.55`. When that proposal is used:
- `resolved_task.classifier` = `"rules_v1+embed_v1+llm_slotfill_v1(task_proposed)"`
- confidence cap `llm_task_proposal` is applied (§7)

### Termination

| Final confidence | Behaviour |
|---|---|
| `>= 0.40` | proceed with the resolved task |
| `< 0.40` and generic fallback enabled | `TaskType = UNSUPPORTED`, generic DAG, warning `LOW_CLASSIFICATION_CONFIDENCE` |
| `< 0.40` and generic fallback disabled | `422 QUERY_UNCLASSIFIABLE` |

### Secondary tasks

`resolved_task.secondary` is **derived, not classified** — it is the set of tasks whose artifacts the primary task's DAG produces as a by-product. Static implication map:

| Primary | Secondary |
|---|---|
| `CHANGE_VQA`, `CHANGE_CAPTION` | `["CHANGE_MAP"]` |
| `COUNT` | `["GROUNDING"]` |
| `SEGMENTATION` | `[]` |
| `CROSS_MODAL_COMPARE` | `["CROSS_MODAL_VQA"]` |
| all others | `[]` |

---

## 3. Slot Schemas

Returned by Stage D, stored at `resolved_task.slots`. Unfilled optional slots are `null`, never absent.

```jsonc
// common to every task
{ "target_class": "built_up",          // normalised land-cover / object term, or null
  "spatial_constraint": null,          // "north-east quadrant", "along the river", or null
  "output_format": "text+mask" }       // "text" | "text+mask" | "text+boxes"

// COUNT               + { "object_class": "aircraft" }                       // required
// GROUNDING           + { "referring_expression": "the large white building near the runway",
//                         "max_boxes": 20 }                                  // required
// SEGMENTATION        + { "target_class": "water" }                          // required
// CHANGE_*            + { "metric": "area_delta",                            // area_delta|presence|direction|null
//                         "direction": "increase" }                          // increase|decrease|any
// CROSS_MODAL_*       + { "focus": "complementarity" }                       // complementarity|agreement|sar_only|optical_only
// SCENE_CLASSIFY      + { "taxonomy": "bigearthnet19" }                      // bigearthnet19|bigearthnet6|free
```

**Normalisation is mandatory.** `target_class` and `object_class` are mapped onto a controlled vocabulary at `configs/class_vocabulary.yaml` before reaching the planner ("buildings", "built up", "urban area", "settlement" → `built_up`). An unmappable term is kept verbatim, passed to the VLM only, and never used for a threshold decision.

---

## 4. The Policy Table

`configs/policy_table.yaml`. Tool names are exactly the registry keys from `API_CONTRACT.md` §3.6 / Master.md §5.

### 4.1 Input addressing

| Token | Resolves to |
|---|---|
| `pre`, `post` | the `img_*` with that role (BI_TEMPORAL) |
| `optical`, `sar` | the `img_*` with that role (CROSS_MODAL) |
| `single` | the sole `img_0` |
| `all` | every input, in order |
| `@N` | every artifact produced by step `N` |
| `@N:CHANGE_MASK` | artifacts of that type from step `N` |

### 4.2 Full table

Notation: `A -> B` sequential, `(A || B)` parallel, all steps depend on the renderer unless stated.

| # | `policy_key` | DAG |
|---|---|---|
| 1 | `VQA\|SINGLE\|optical` | `spectral_renderer -> (raster_statistics \|\| spectral_index_analyzer) -> vlm_vqa` |
| 2 | `VQA\|SINGLE\|sar` | `spectral_renderer -> (raster_statistics \|\| sar_backscatter_analyzer) -> vlm_vqa` |
| 3 | `VQA\|SINGLE\|panchromatic` | `spectral_renderer -> raster_statistics -> vlm_vqa` |
| 4 | `CAPTION\|SINGLE\|optical` | `spectral_renderer -> spectral_index_analyzer -> vlm_caption` |
| 5 | `CAPTION\|SINGLE\|sar` | `spectral_renderer -> sar_backscatter_analyzer -> vlm_caption` |
| 6 | `CAPTION\|SINGLE\|panchromatic` | `spectral_renderer -> raster_statistics -> vlm_caption` |
| 7 | `GROUNDING\|SINGLE\|*` | `spectral_renderer -> text_grounding -> vlm_vqa` |
| 8 | `SEGMENTATION\|SINGLE\|optical` | `spectral_renderer -> (spectral_index_analyzer \|\| semantic_segmenter) -> vlm_vqa` |
| 9 | `COUNT\|SINGLE\|*` | `spectral_renderer -> text_grounding -> object_counter -> vlm_vqa` |
| 10 | `SCENE_CLASSIFY\|SINGLE\|optical` | `spectral_renderer -> spectral_index_analyzer -> vlm_vqa` |
| 11 | `SCENE_CLASSIFY\|SINGLE\|sar` | `spectral_renderer -> sar_backscatter_analyzer -> vlm_vqa` |
| 12 | `CHANGE_VQA\|BI_TEMPORAL\|optical` | `spectral_renderer -> siamese_change_detector -> change_statistics -> spectral_index_analyzer(pre,post) -> vlm_change_vqa` |
| 13 | `CHANGE_VQA\|BI_TEMPORAL\|sar` | `spectral_renderer -> sar_backscatter_analyzer(pre,post) -> siamese_change_detector -> change_statistics -> vlm_change_vqa` |
| 14 | `CHANGE_CAPTION\|BI_TEMPORAL\|*` | same as 12/13, terminal tool `vlm_change_vqa` in caption mode |
| 15 | `CHANGE_MAP\|BI_TEMPORAL\|*` | `spectral_renderer -> siamese_change_detector -> change_statistics` **(no VLM — templated answer)** |
| 16 | `CROSS_MODAL_VQA\|CROSS_MODAL\|mixed` | `spectral_renderer -> (spectral_index_analyzer \|\| sar_backscatter_analyzer) -> crossmodal_consistency -> physics_agreement -> vlm_vqa` |
| 17 | `CROSS_MODAL_COMPARE\|CROSS_MODAL\|mixed` | same as 16, terminal `vlm_vqa` in compare mode |
| 18 | `VQA\|CROSS_MODAL\|mixed` | same as 16 |
| 19 | `CAPTION\|CROSS_MODAL\|mixed` | `spectral_renderer -> (spectral_index_analyzer \|\| sar_backscatter_analyzer) -> physics_agreement -> vlm_caption` |
| 20 | `GROUNDING\|CROSS_MODAL\|mixed` | `spectral_renderer -> text_grounding(optical) -> crossmodal_consistency -> vlm_vqa` |
| 21 | `SCENE_CLASSIFY\|CROSS_MODAL\|mixed` | `spectral_renderer -> (spectral_index_analyzer \|\| sar_backscatter_analyzer) -> physics_agreement -> vlm_vqa` |
| 22 | `*\|*\|*` (generic) | `spectral_renderer -> raster_statistics -> vlm_vqa` |

Entry 15 is deliberately VLM-free: a pure change-map request is fully answerable from deterministic outputs, so it must not pay VLM latency or risk VLM hallucination. Its `answer.template_fallback` is `true` **by design**, not by failure — the confidence cap `template_fallback` is waived for this key (`caps_waived: ["template_fallback"]` in the entry).

### 4.3 YAML format

```yaml
version: policy_table_v1
entries:
  - key: "CHANGE_VQA|BI_TEMPORAL|optical"
    caps_waived: []
    steps:
      - tool: spectral_renderer
        inputs: [pre, post]
        params: { views: [TC, FCIR, NDBI], size_px: 448 }
      - tool: siamese_change_detector
        inputs: [pre, post]
        depends_on: [1]
        params: { threshold: 0.5, tile: 512, overlap: 64, tta: false }
      - tool: change_statistics
        inputs: ["@2:CHANGE_MASK"]
        depends_on: [2]
        params: { min_component_px: 25 }
      - tool: spectral_index_analyzer
        inputs: [pre, post]
        depends_on: [1]
        params: { indices: [ndvi, ndbi, ndwi], suffixes: [_pre, _post] }
      - tool: vlm_change_vqa
        inputs: ["@1", "@2:CHANGE_MASK"]
        depends_on: [1, 3, 4]
        params: { mode: vqa, max_new_tokens: 384, temperature: 0.0 }
```

Steps are listed in topological order; `step` numbers are assigned by position (1-indexed). `depends_on` must reference only earlier steps — the loader rejects the table otherwise.

**Extension rule (Master.md §5):** adding a tool must never require touching `planner.py`. New behaviour arrives as a registry entry plus one or more policy-table rows.

---

## 5. Capability Matching

Runs after table lookup, before execution. For each step, in order:

```
spec = registry[step.tool]
1. availability   spec.available == true                       else -> FALLBACK
2. pair type      plan.pair_type in spec.accepts.pair_types    else -> FALLBACK
3. image count    min_images <= n(resolved inputs) <= max_images  else -> FALLBACK
4. modality       {m(i) for i in inputs} subset of spec.accepts.modalities  else -> FALLBACK
5. bands          every spec.accepts.required_bands resolvable
                  via the per-sensor alias map                 else -> FALLBACK
6. georeference   spec.accepts.requires_georeference implies
                  every input.is_georeferenced                  else -> FALLBACK
7. size           min_size_px <= min(w,h) and max(w,h) <= max_size_px  else -> FALLBACK
8. gsd            gsd_m within spec.accepts.gsd_range_m         else -> WARN, still run
```

Checks 1-7 are **hard**. Check 8 is **soft**: an out-of-range GSD emits `check: gsd_ratio` style warning and reduces `input_quality`, but does not block — the Cartosat-2S transfer case (0.65 m against tools trained at 0.5-10 m) must degrade, not refuse.

**Band alias map** — logical names used in `required_bands`, resolved per sensor. Canonical table lives in `DATA_ADAPTATION_PLAN.md` §2.2; both consumers read the same YAML (`configs/band_aliases.yaml`).

**FALLBACK resolution:** substitute `spec.fallback` and re-run checks 1-7 against it. **One level only** — a fallback's fallback is not followed. If the fallback also fails, the step becomes `SKIPPED`, and every step transitively depending on it becomes `SKIPPED`. If a `vlm_*` terminal step is skipped, the aggregator emits a templated answer with `template_fallback: true`.

**Declared fallback chains (frozen):**

| Tool | Fallback | Why |
|---|---|---|
| `siamese_change_detector` | `image_diff_change` | CVA / log-ratio baseline; no weights needed |
| `semantic_segmenter` | `spectral_index_analyzer` | index thresholding yields a coarse class map |
| `text_grounding` | `semantic_segmenter` | class mask -> connected-component boxes |
| `crossmodal_consistency` | `physics_agreement` | deterministic rule layer, no DOFA weights needed |
| `object_counter` | *(none)* | trivially deterministic; cannot fail |
| `vlm_vqa` / `vlm_caption` / `vlm_change_vqa` | *(none)* | -> templated answer via aggregator |
| `spectral_renderer`, `raster_statistics`, `spectral_index_analyzer`, `sar_backscatter_analyzer`, `change_statistics`, `physics_agreement` | *(none)* | pure-numpy; a failure here is a bug, not a condition |

---

## 6. Executor Semantics

### 6.1 Scheduling

- Deterministic topological order; ties broken by ascending `step`.
- Steps whose dependencies are all satisfied run concurrently, bounded by:
  - `max_parallel_tools` = **3** (CPU-bound tools)
  - `max_parallel_gpu_tools` = **1** — a single global semaphore. With 24 GB shared between an 8B VLM and CV models, concurrent GPU tools are the fastest route to `VRAM_EXHAUSTED`.
- Per-step timeout: `max(3 * spec.est_ms, 5000)` ms, overridable per entry. Timeout is treated exactly like a failure (fallback, then `SKIPPED`).

### 6.2 Cache key

```
cache_key = sha256(
    tool_name || tool_version ||
    canonical_json(sorted(params)) ||
    concat(sorted(input_content_hashes)) ||
    common_grid_hash
)
```

`input_content_hashes` are `InputManifest.sha256` for `img_*` and the content hash for `art_*`.

**Explicitly excluded from the key:** `trace_id`, `created_at`, the raw query text, upload filenames, and any wall-clock value. A cache hit sets `execution.cache_hit = true` and `duration_ms` to the *replayed* duration (0-2 ms), never the original.

### 6.3 Status propagation

- A step consuming an artifact from a `DEGRADED` step is itself **at most** `DEGRADED`, even if it succeeds outright.
- `FAILED` and `SKIPPED` do not propagate as failures — they propagate as `SKIPPED` on dependents.
- The request as a whole never returns 5xx for a tool-level failure (`API_CONTRACT.md` §4.1 contract guarantee). `TOOL_FAILED_NO_FALLBACK` (500) is reserved for a step declared non-optional in the policy entry — currently only `spectral_renderer`.

### 6.4 FactSheet merge

```
fact_sheet["{tool_name}.{scalar_key}"] = value
```

- Namespacing is unconditional, including for single-tool plans.
- If the same tool runs twice in one plan (e.g. `spectral_index_analyzer` on `pre` and `post`), the step's `params.suffixes` disambiguate at the scalar level (`ndbi_mean_pre`, `ndbi_mean_post`) — the namespace is **not** further qualified by step, so scalar keys within a tool must be unique across its invocations. The policy table is responsible for supplying distinct suffixes; the loader validates this statically.
- Only scalars validating against `spec.scalars_schema` are merged. A non-conforming scalar is dropped and a warning `SCALAR_SCHEMA_VIOLATION` is recorded — it must never reach the answer, because the FactSheet is the citation ground truth.

---

## 7. Evidence, Citation & Confidence

### 7.1 `CitationValidator`

1. **Extract** numeric spans from `answer.text`:
   `[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(%|km2|km²|m2|m²|m|dB|px)?`
2. **Exclude** from validation: bare 4-digit integers in `1900-2100` (years), the `N` in `Image N`, list markers, and digits inside a quoted `source` label.
3. **Resolve** each remaining span against the FactSheet:
   - numeric tolerance: relative `1 %` for `|v| >= 10`, absolute `0.05` for `|v| < 10`
   - unit compatibility is **required**: `%` matches only `*_pct`; `dB` only `*_db`; `km2` only `*_km2`; `m2` only `*_m2`; unitless matches unitless
4. **Matched** -> append a `Citation` with `source: "step:{N}/scalars.{key}"` and the resolved `value`.
   **Unmatched** -> append the raw span to `uncited_numeric_spans`.
5. **Policy** `citation_policy`:
   - `flag` (**default**) — keep the sentence, list the span. The honesty signal is worth more than a tidy answer, and stripping mid-sentence produces incoherent text.
   - `strip` — remove the containing sentence. Available for benchmark runs where an unsupported number is scored as wrong.

### 7.2 `weighted_tool_agreement_v1`

```
overall = 0.20*task_classification
        + 0.20*input_quality
        + 0.35*tool_mean
        + 0.25*cross_tool_agreement
```

**`task_classification`** = `resolved_task.confidence`.

**`input_quality`** = `pair_factor * mean_i(q_i)` where

```
q_i        = (1 - nodata_pct_i/100) * georef_i * gsd_i
georef_i   = 1.00 if is_georeferenced else 0.85
gsd_i      = 1.00 if gsd within every executed tool's gsd_range_m else 0.80
pair_factor= 1.00 (PASS) | 0.85 (PASS_WITH_WARNINGS)
```

**`tool_mean`** = mean of `execution.confidence` over **evidence-producing** steps only (`category` in `analysis`, `cv`, `fusion`). VLM steps are excluded — they synthesise evidence, they do not produce it, and including them would let a fluent answer inflate confidence in weak measurements. Contributions: `OK` -> `confidence`; `DEGRADED` -> `0.5 * confidence`; `FAILED` -> `0`; `SKIPPED` -> excluded from the mean. If no evidence step ran, `tool_mean = 0.50`.

**`cross_tool_agreement`**

- Fewer than two tools contributed a comparable scalar pair -> **`0.75`** (neutral prior; deliberately *not* `1.0` — an uncorroborated measurement must not score as if it were confirmed).
- Otherwise, over each pair in `AGREEMENT_PAIRS`: `1 - min(1, |a - b| / scale)`, averaged.

`AGREEMENT_PAIRS` (`configs/agreement_pairs.yaml`, frozen set):

| Scalar A | Scalar B | scale |
|---|---|---|
| `siamese_change_detector.changed_area_pct` | `image_diff_change.changed_area_pct` | 15.0 |
| `spectral_index_analyzer.built_up_fraction_pct` | `change_statistics.changed_area_pct` | 20.0 |
| `crossmodal_consistency.agreement_mean` | `physics_agreement.agreement_score` | 0.50 |
| `semantic_segmenter.water_fraction_pct` | `spectral_index_analyzer.water_fraction_pct` | 10.0 |
| `object_counter.count` | `text_grounding.n_boxes` | 3.0 |

**Caps** — applied after the weighted sum, in order. Each records its name in `confidence.caps_applied`. A cap named in the policy entry's `caps_waived` is skipped.

| Cap name | Condition | Clamp |
|---|---|---|
| `compatibility_warnings` | `overall_compat == PASS_WITH_WARNINGS` | `<= 0.80` |
| `llm_task_proposal` | Stage D proposed the task | `<= 0.75` |
| `degraded_execution` | any execution `DEGRADED` | `<= 0.70` |
| `template_fallback` | `answer.template_fallback == true` | `<= 0.65` |
| `uncited_claims` | `uncited_numeric_spans` non-empty | `<= 0.60` |
| `failed_execution` | any execution `FAILED` | `<= 0.50` |
| `generic_plan` | `planner == "fallback_generic_v1"` | `<= 0.45` |

Result rounded to 4 decimal places. Components are reported unrounded-then-rounded identically, so the frontend can display the arithmetic and it will add up.

---

## 8. Determinism Guarantees

**Guaranteed identical** across two runs with the same images, query, `seed`, registry version and policy-table version:

`plan.policy_key` · `plan.steps` (including `params`) · every `executions[].params` · every `executions[].scalars` · `fact_sheet` · `compatibility.checks` · `confidence.components` (excluding any VLM-derived term)

**Not guaranteed:** `answer.text` (mitigated by `temperature: 0.0` and a fixed seed on every `vlm_*` step) · `duration_ms` · `created_at` · `trace_id` · `cache_hit` · artifact URLs (content-addressed, so identical content yields identical paths — but eviction may differ).

The Phase 3 reproducibility test asserts exactly the guaranteed set, field by field.

---

## 9. Phase 3 Golden Tests

The 20 assertions required by Master.md §8 Phase 3. Fixtures come from `scripts/make_synthetic_fixtures.py`.

**Routing (query -> policy_key)**

1. `"what changed between these two images"` + BI_TEMPORAL optical -> `CHANGE_VQA|BI_TEMPORAL|optical`
2. `"show me a map of the changed areas"` + BI_TEMPORAL -> `CHANGE_MAP|BI_TEMPORAL|optical`
3. `"describe what happened between the two dates"` + BI_TEMPORAL -> `CHANGE_CAPTION|BI_TEMPORAL|optical`
4. `"how many buildings are there"` + SINGLE optical -> `COUNT|SINGLE|optical`, slot `object_class=built_up`
5. `"where is the airport"` + SINGLE -> `GROUNDING|SINGLE|optical`
6. `"describe this scene"` + SINGLE SAR -> `CAPTION|SINGLE|sar`
7. `"what does the SAR show that the optical misses"` + CROSS_MODAL -> `CROSS_MODAL_COMPARE|CROSS_MODAL|mixed`
8. `"outline the water bodies"` + SINGLE optical -> `SEGMENTATION|SINGLE|optical`
9. `"what land cover types are present"` + SINGLE optical -> `SCENE_CLASSIFY|SINGLE|optical`
10. `"how many changed regions"` + BI_TEMPORAL -> `CHANGE_VQA` (change-word wins over count at Stage A gate), secondary `["CHANGE_MAP"]`
11. `"what changed"` + **SINGLE** -> gate forbids `CHANGE_*`; resolves to `VQA|SINGLE|optical` with warning
12. Gibberish query -> confidence `< 0.40` -> `UNSUPPORTED` + `fallback_generic_v1` + `GENERIC_PLAN_USED`

**Structure**

13. Every generated plan is a DAG (no cycles) and `depends_on` references only earlier steps.
14. Every tool named in `policy_table.yaml` exists in `registry.yaml` (static lint, runs in CI).
15. Every `required_bands` entry in every `ToolSpec` resolves in `band_aliases.yaml` for at least one sensor (static lint).

**Execution**

16. Same inputs twice -> identical `plan`, identical `executions[].params`, identical `fact_sheet` (§8).
17. `siamese_change_detector` forced to raise -> `image_diff_change` runs, execution recorded `DEGRADED` with `fallback_of` set, HTTP status still `200`.
18. `semantic_segmenter` marked `available: false` -> capability matching substitutes `spectral_index_analyzer`; no exception escapes.
19. Second identical request -> `cache_hit: true` on every deterministic step, and `duration_ms` drops by `> 10x`.

**Evidence**

20. An answer with an injected uncited number (`"about 42% of the area"` with no matching scalar) -> that span appears in `uncited_numeric_spans`, `caps_applied` contains `uncited_claims`, and `confidence.overall <= 0.60`.

---

## 10. Resolved Ambiguities & Deltas from `Master.md`

| # | Item | Master.md state | Resolution | Rationale |
|---|---|---|---|---|
| 1 | Classifier architecture | §8 Phase 3: "rules + embeddings over a labelled query set"; §4.4 shows `"rules_v1+llm_slotfill_v1"` | Four-stage cascade (§2) with both named strings reachable | The two references describe different stages of the same cascade; the `classifier` string names whichever stages actually ran. |
| 2 | `cross_tool_agreement` with one tool | Undefined | Neutral prior **0.75**, not 1.0 | Scoring an uncorroborated measurement as fully corroborated would make single-tool plans systematically overconfident. |
| 3 | VLM steps in `tool_mean` | Undefined | **Excluded** | The VLM synthesises evidence rather than producing it; including it lets fluency inflate confidence in weak measurements. |
| 4 | `CHANGE_MAP` and templated answers | §5 implies every task ends at a VLM | Entry 15 is VLM-free with `template_fallback` cap waived | A pure change-map request is fully answerable deterministically; paying VLM latency and hallucination risk for it is strictly worse. |
| 5 | Fallback depth | §8 Phase 3 says "fallback chains" | **One level only** | Unbounded chains make latency and the trace unpredictable; the frozen table (§5) covers every real case in one hop. |
| 6 | GPU concurrency | Not specified | `max_parallel_gpu_tools = 1` | 24 GB shared between an 8B VLM and CV models; concurrent GPU steps are the shortest path to `VRAM_EXHAUSTED` (Master.md §9). |
| 7 | Repeated tool in one plan | Not specified | Namespace by tool only; policy table must supply distinct scalar suffixes, statically validated | Namespacing by step would make citation keys unstable across plan edits. |
| 8 | `citation_policy` default | §4.4 says "stripped or flagged" | Default **`flag`** | Stripping mid-sentence produces incoherent text; the visible honesty signal is the demo asset (§4.4: "Demo this to judges"). |
| 9 | Soft vs hard capability checks | Not specified | GSD is soft; all others hard | Cartosat-2S at 0.65 m falls outside tools trained at 0.5-10 m. Refusing the hidden evaluation set would be a catastrophic failure mode; degrading is correct. |
| 10 | Generic-plan visibility | §3.1 says free-form fallback is "flagged as such" | `planner: "fallback_generic_v1"` + warning + `generic_plan` cap `<= 0.45` | Makes the flag machine-readable and self-limiting rather than cosmetic. |
