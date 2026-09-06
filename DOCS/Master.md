# SatQuery AI — Master Architecture & Implementation Plan
**SIH Problem 26167 · ISRO / Space Applications Centre · Agentic Vision-Language Assistant for Remote Sensing**

---

## 1. Context

We are building the complete backend for SatQuery AI: an agentic vision-language assistant that answers natural-language queries over single, cross-modal (optical+SAR), and bi-temporal remote-sensing imagery. A teammate builds the frontend and consumes our REST API.

The problem statement explicitly **disqualifies monolithic generic VLMs**. The scored novelty is the *agentic, query-driven framework*: a controller that parses the query, validates geospatial input compatibility, selects and sequences specialist tools from a registry, fuses their outputs, and returns an **evidence-grounded answer plus an auditable execution trace**. Internal chain-of-thought is explicitly *not* evaluated — only the observable trace is.

Two consequences drive every decision below:
1. **The trace is a first-class product, not logging.** It is a versioned, schema-validated API artifact built by the executor, not reconstructed from logs.
2. **The VLM is one tool among many, not the system.** Every claim in the final answer must be traceable to a deterministic tool output. This is our anti-hallucination story and our differentiator.

The repo is currently empty (`ca22b20 Initial commit`, only `.gitattributes`). This is greenfield.

### The hidden-evaluation gap (drives model/data choices)
The ISRO/SAC hidden set is **Cartosat-2S optical (~0.65 m pan / ~1.6 m MX, 4 bands) + RISAT SAR**. The mandated training dataset is **BigEarthNet (Sentinel-1/2, 10 m, 12 bands)** — a ~10–15× GSD gap and a different band set. We therefore run **two-track adaptation**: BigEarthNet for sensor/physics grounding (mandate), and VHR corpora (VRSBench, DIOR-RSVG, LEVIR-CD @0.5 m) for resolution/task alignment with Cartosat. Both are merged into one instruction corpus and one adapter.

---

## 2. Verified Environment & Hard Constraints

Measured on this machine:

| Fact | Value | Consequence |
|---|---|---|
| GPU | **AMD Radeon RX 7900 XTX, 24 GB (gfx1100)** | ROCm target. All scoping below assumes a 24 GB budget. |
| iGPU | AMD Radeon Graphics (Ryzen 7800X3D) | **Gotcha:** ROCm enumerates it as a second agent and breaks PyTorch. Must pin `HIP_VISIBLE_DEVICES=0`. |
| CPU / RAM | Ryzen 7 7800X3D, 16 threads / 31 GB | Fine for tiling, raster IO, ONNX CPU fallback. Cap dataloader workers at 8. |
| Disk free | 377 GB | Full BigEarthNet-v2 (549,488 S1+S2 pairs) will **not** fit comfortably. Use a stratified subset (§7.2). |
| Python | 3.14.7 only, no conda, no Docker | Torch/rasterio wheels are unreliable on 3.14. **Pin Python 3.11.** |
| OS | Windows 11 → **user is migrating to Fedora Linux** | Unblocks native ROCm PyTorch training, Docker, and vLLM. Plan assumes Fedora. |

### Decisions confirmed with the user
- **Base VLM:** Qwen3-VL-8B-Instruct + our own LoRA.
- **Training:** 100% local on the 7900 XTX via ROCm. No cloud dependency.
- **Serving:** hybrid — local-first (must survive a dead network at judging), cloud optional.
- **Timeline:** 1–2 months.

### Phase −1: Fedora + ROCm bring-up (do this before Phase 0)

```bash
# 1. Kernel driver + permissions (Fedora's in-tree amdgpu supports gfx1100)
sudo usermod -aG video,render $USER   # then log out/in
sudo dnf install rocminfo rocm-smi
rocminfo | grep -i gfx                 # MUST print gfx1100

# 2. Python 3.11 + uv
sudo dnf install python3.11 python3.11-devel
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv && source .venv/bin/activate

# 3. PyTorch ROCm (bundles its own ROCm userspace — do NOT mix with dnf rocm libs)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4

# 4. Verify
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True  AMD Radeon RX 7900 XTX
```

Required env (goes in `.env` and `scripts/rocm_env.sh`):
```bash
export HIP_VISIBLE_DEVICES=0            # hide the iGPU — critical
export ROCR_VISIBLE_DEVICES=0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # only if rocminfo/torch disagree
```

**Escape hatch if native install fights back:** `docker run --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined rocm/pytorch:latest`. Keep this in `scripts/` from day one — do not burn days on userspace library mismatches.

**Do not use OpenMMLab / mmcv.** Its custom CUDA ops are a known ROCm build trap. We use pure-PyTorch + `torchgeo` + PyTorch Lightning instead (§6.3).

---

## 3. System Architecture

```
                         POST /v1/analyze  (multipart: images[] + query)
                                        │
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ ① INGESTION & VALIDATION            src/satquery/ingest/                      │
│   rasterio open → InputManifest (CRS, transform, GSD, bands, dtype, nodata)   │
│   modality classifier (optical | SAR | pan | unknown)                         │
│   pair-type inference (SINGLE | CROSS_MODAL | BI_TEMPORAL)                    │
│   co-registration: CRS reproject · bounds IoU · GSD ratio · phase-corr offset │
│   ⇒ InputManifest[]  +  CompatibilityReport      ◀── mandatory req. 5         │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ ② SPECTRAL RENDERING & EVIDENCE      src/satquery/render/                     │
│   N-band raster → named 3-channel views: TrueColor, FalseColorIR, SWIR,       │
│   NDVI, NDWI, NDBI, SAR-FalseColor(VV,VH,VV/VH), SAR-dB                       │
│   ⇒ feeds BOTH the VLM (as named images) AND the frontend (as PNG evidence)   │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ ③ QUERY UNDERSTANDING → PLANNER      src/satquery/agent/                      │
│   task classifier → TaskType enum + slots (target class, spatial, temporal)   │
│   deterministic policy table: (TaskType × PairType × Modality) → Tool DAG     │
│   capability matching against ToolSpec contracts                              │
│   ⇒ QueryPlan (a DAG, not free-form ReAct — reproducible & auditable)         │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ ④ TOOL REGISTRY & EXECUTOR           src/satquery/registry/ · tools/          │
│   async DAG execution · per-tool timeout · content-hash cache · fallbacks     │
│   every tool returns ToolResult{artifacts, scalars, confidence, params, ms}   │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ ⑤ EVIDENCE AGGREGATOR                src/satquery/evidence/                  │
│   FactSheet ← union of all tool scalars (the ONLY numbers allowed in output)  │
│   VLM synthesises answer constrained by FactSheet + rendered views            │
│   CitationValidator: every number/claim must resolve to step:N/scalar.path    │
│   calibrated confidence = f(tool confidences, cross-tool agreement)           │
└───────────────────────────────────────┬───────────────────────────────────────┘
┌───────────────────────────────────────▼───────────────────────────────────────┐
│ ⑥ API LAYER                          src/satquery/api/                       │
│   AnalyzeResponse {answer, artifacts[], confidence, trace}                    │
│   SSE progress stream · /v1/artifacts/{id} · OpenAPI 3.1 → frontend contract  │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Why a deterministic planner, not free-form ReAct
The rubric scores the *observable* trace and ignores CoT. A policy-table planner gives us: byte-identical reruns for the same input (auditability), sub-second planning latency, unit-testable routing, and no failure mode where an LLM invents a tool that doesn't exist. The LLM is used for *slot filling* (structured JSON output) and *answer synthesis* — never for control flow. Free-form fallback exists only for unclassifiable queries and is flagged as such in the trace.

---

## 4. Core Schemas (`src/satquery/schemas/`)

Pydantic v2. These are frozen at the end of Phase 0 and are the frontend contract.

### 4.1 `InputManifest`
```python
class InputManifest(BaseModel):
    id: str  # "img_0"
    role: Literal["single", "pre", "post", "optical", "sar"] | None
    filename: str
    sha256: str
    size_bytes: int
    driver: str  # "GTiff" | "PNG" | ...
    modality: Literal["optical", "sar", "panchromatic", "unknown"]
    modality_confidence: float
    sensor_guess: str | None  # "Sentinel-2 L2A" | "Cartosat-2S" | "RISAT-1"
    crs: str | None  # "EPSG:32643"
    transform: list[float] | None  # 6-tuple affine
    bounds_native: list[float] | None
    bounds_wgs84: list[float] | None
    gsd_m: float | None
    width: int
    height: int
    band_count: int
    dtype: str
    band_names: list[str] | None  # ["B02","B03",...] | ["VV","VH"]
    nodata_pct: float
    acquisition_time: datetime | None  # from TIFF tags if present
    is_georeferenced: bool
    warnings: list[str]
```

### 4.2 `CompatibilityReport` (mandatory requirement 5)
```python
class CheckResult(BaseModel):
    name: str  # "crs_match" | "bounds_overlap_iou" | "gsd_ratio"
    # | "coregistration_offset_px" | "band_sufficiency"
    # | "modality_distinct" | "temporal_ordering"
    status: Literal["PASS", "WARN", "FAIL", "SKIP"]
    value: float | str | None
    threshold: str | None  # "iou >= 0.80"
    detail: str


class CompatibilityReport(BaseModel):
    pair_type: Literal["SINGLE", "CROSS_MODAL", "BI_TEMPORAL", "INCOMPATIBLE"]
    pair_type_source: Literal["metadata", "heuristic", "user_declared"]
    checks: list[CheckResult]
    overall: Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]
    actions_taken: list[str]  # "reprojected img_1 EPSG:4326→EPSG:32643 (bilinear)"
    # "resampled img_1 20.0m→10.0m"
    # "cropped both to intersection 1024x1024"
    common_grid: dict | None  # {crs, transform, width, height, gsd_m}
```

### 4.3 `ToolSpec` (registry entry) & `ToolResult`
```python
class ToolSpec(BaseModel):
    name: str
    version: str
    category: Literal["analysis", "vlm", "cv", "geo", "fusion"]
    description: str
    accepts: InputContract  # pair_types[], modalities[], min/max images,
    # required_bands[], gsd_range_m, min_size_px
    produces: list[ArtifactType]  # CHANGE_MASK | BBOX_SET | SEGMENTATION | HEATMAP
    # | TEXT | SCALARS | OVERLAY_PNG | GEOJSON
    scalars_schema: dict  # JSON-schema of the scalars it contributes to FactSheet
    device: Literal["rocm", "cpu", "auto"]
    est_ms: int
    fallback: str | None  # name of a cheaper deterministic tool


class ToolResult(BaseModel):
    tool: str
    version: str
    status: Literal["OK", "DEGRADED", "FAILED", "SKIPPED"]
    params: dict  # exact execution parameters (auditable)
    scalars: dict  # merged into FactSheet
    artifacts: list[ArtifactRef]
    confidence: float
    duration_ms: int
    device_used: str
    error: str | None
```

### 4.4 `AuditTrace` — the mandatory deliverable
```jsonc
{
  "trace_id": "b3f1…", "schema_version": "1.0",
  "created_at": "2026-09-05T12:00:00Z", "duration_ms": 4820,
  "query": { "raw": "How much built-up area appeared between these two dates?",
             "normalized": "...", "language": "en" },

  "resolved_task": { "primary": "CHANGE_VQA", "secondary": ["CHANGE_MAP"],
                     "slots": { "target_class": "built_up", "metric": "area_delta" },
                     "confidence": 0.94, "classifier": "rules_v1+llm_slotfill_v1" },

  "inputs":        [ /* InputManifest[] */ ],
  "compatibility": { /* CompatibilityReport */ },

  "plan": { "planner": "policy_table_v1", "policy_key": "CHANGE_VQA|BI_TEMPORAL|optical",
            "steps": [ { "step": 1, "tool": "spectral_renderer",  "depends_on": [] },
                       { "step": 2, "tool": "changeformer_levircd","depends_on": [1] },
                       { "step": 3, "tool": "change_statistics",   "depends_on": [2] },
                       { "step": 4, "tool": "vlm_change_vqa",      "depends_on": [1,3] } ] },

  "executions": [
    { "step": 2, "tool": "changeformer_levircd", "version": "1.0.0+ckpt:9a3f",
      "status": "OK", "device_used": "rocm:0", "duration_ms": 1830,
      "params": { "threshold": 0.5, "tile": 512, "overlap": 64, "tta": false },
      "input_refs": ["img_0","img_1"], "output_refs": ["art_mask_01"],
      "scalars": { "changed_area_pct": 7.4, "changed_area_km2": 1.07, "n_components": 23 },
      "confidence": 0.88 }
  ],

  "artifacts": [
    { "id": "art_mask_01", "type": "CHANGE_MASK", "mime": "image/png",
      "url": "/v1/artifacts/b3f1/art_mask_01.png",
      "geotiff_url": "/v1/artifacts/b3f1/art_mask_01.tif",
      "geo": { "crs": "EPSG:32643", "transform": [10,0,499980,0,-10,2100000] },
      "stats": { "positive_px": 88_000, "total_px": 1_188_000 },
      "produced_by_step": 2 }
  ],

  "fact_sheet": { "changed_area_pct": 7.4, "ndbi_mean_pre": -0.12, "ndbi_mean_post": 0.08 },

  "answer": { "text": "Approximately 7.4% of the scene (1.07 km²) transitioned to built-up…",
              "citations": [ { "claim": "7.4% of the scene",
                               "source": "step:2/scalars.changed_area_pct" },
                             { "claim": "NDBI rose from -0.12 to 0.08",
                               "source": "step:3/scalars.ndbi_mean_pre|post" } ],
              "uncited_numeric_spans": [] },

  "confidence": { "overall": 0.86, "method": "weighted_tool_agreement_v1",
                  "components": { "task_classification": 0.94, "input_quality": 0.97,
                                  "tool_mean": 0.88, "cross_tool_agreement": 0.81 } },
  "warnings": [], "errors": []
}
```

`uncited_numeric_spans` is the honesty signal: any number the VLM emits that the `CitationValidator` cannot resolve to a tool scalar is listed there and stripped or flagged. Demo this to judges.

### 4.5 API surface
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/analyze` | multipart: `images[]`, `query`, `options` → `AnalyzeResponse` (sync, ≤30 s) |
| `POST` | `/v1/jobs` | same payload → `{job_id}` for long jobs |
| `GET` | `/v1/jobs/{id}` · `/v1/jobs/{id}/events` | poll / SSE progress (`stage`, `step`, `pct`) |
| `POST` | `/v1/validate` | ingestion + compatibility only — cheap, lets the frontend pre-flight uploads |
| `GET` | `/v1/artifacts/{trace}/{artifact}` | PNG / GeoTIFF / GeoJSON bytes |
| `GET` | `/v1/traces/{id}` | full `AuditTrace` |
| `GET` | `/v1/registry` | live tool registry — frontend renders "capabilities" panel |
| `GET` | `/v1/health` | device, loaded models, VRAM |

---

## 5. Tool Registry (target state)

| Tool | Category | Backs requirement | Implementation |
|---|---|---|---|
| `spectral_renderer` | geo | all | rasterio + numpy composites & indices |
| `raster_statistics` | analysis | VQA grounding | band stats, histograms, land-cover fractions |
| `spectral_index_analyzer` | analysis | optical reasoning | NDVI/NDWI/NDBI/NBR + thresholded class maps |
| `sar_backscatter_analyzer` | analysis | SAR reasoning | σ⁰ dB, VV/VH ratio, GLCM texture, speckle stats |
| `crossmodal_consistency` | fusion | **req 4** | DOFA embeddings + index/backscatter agreement map |
| `siamese_change_detector` | cv | **req 3** | ChangeFormer/TinyCD-style, trained LEVIR-CD + OSCD |
| `change_statistics` | analysis | **req 3** | connected components, area, direction-of-change |
| `semantic_segmenter` | cv | req 2 (option B) | SegFormer/UPerNet on LoveDA + OpenEarthMap |
| `text_grounding` | cv/vlm | **req 2** | Qwen3-VL native bbox + DIOR-RSVG LoRA |
| `object_counter` | analysis | RSVQA counting | components on grounding/segmentation output |
| `vlm_vqa` · `vlm_caption` · `vlm_change_vqa` | vlm | **req 2, 3** | adapted Qwen3-VL-8B, FactSheet-constrained |
| `earthdial_baseline` | vlm | ablation | EarthDial-4B (MIT) zero-shot comparison |

Every tool implements one protocol (`tools/base.py`) and is registered declaratively in `configs/registry.yaml`. Adding a tool must never require touching the planner — only the policy table.

---

## 6. Model & Library Selection

### 6.1 Core VLM — Qwen3-VL-8B-Instruct + LoRA
Chosen for: native normalized-bbox grounding (directly serves the region-grounding requirement), a real multi-image interface (essential for cross-modal and bi-temporal), first-class llama.cpp support for the local Vulkan/HIP demo path, and it is the exact base validated by the Sept-2026 multispectral/SAR adaptation study.

### 6.2 Domain adaptation — Spectral Rendering + LoRA (the key design choice)
**Do not modify patch embeddings** to accept 12-band input. Instead render each raster into a small set of **named, interpretable 3-channel views** and feed them through the model's existing multi-image interface with explicit naming in the prompt (`"Image 3 (optical NDVI heatmap)"`, `"Image 5 (SAR false-colour VV/VH/ratio)"`).

This one decision pays for itself three times over:
- Handles Sentinel-2 (12 band), Sentinel-1 (2 band), Cartosat-2S (4 band) and RISAT (1 band) with **zero architecture change** — critical for the hidden test set.
- The same renderer produces the **human-viewable evidence PNGs** the frontend displays.
- Published result on this exact recipe: BigEarthNet-v2 micro-F1 **0.5921 zero-shot → 0.8275 adapted**, within 1.7 pts of a dedicated specialist encoder. That delta *is* our "we domain-adapted a VLM" evidence.

**Training profile A (default) — QLoRA, scoped to 24 GB gfx1100:**
| Setting | Value | Rationale |
|---|---|---|
| Quantization | bitsandbytes NF4 + double-quant, bf16 compute | 8B → ~5.5 GB, leaves ~18 GB for activations |
| LoRA | r=16, α=32, dropout=0.05 | ~50 M trainable, ~200 MB adapter |
| Targets | LLM `q,k,v,o,gate,up,down` + **last 8 ViT blocks** | ViT adaptation is what teaches SAR/index semantics |
| Attention | `sdpa` | flash-attn CK is unreliable on RDNA3 — do not fight it |
| Batch | `per_device=1`, `grad_accum=16` | effective batch 16 |
| Memory | gradient checkpointing ON, `expandable_segments:True` | peak ≈ 18–21 GB |
| Images | ≤6 rendered views/sample, `max_pixels≈448²` per view | the dominant VRAM knob — tune here first |
| Optimizer | `paged_adamw_8bit`, lr 1e-4 cosine, 3% warmup, 2 epochs | |
| Expected | ~2.5 k steps ≈ **4–7 h** on the 7900 XTX | one overnight run per iteration |

**Profile B (fallback):** Qwen3-VL-**4B** bf16 LoRA (8 GB weights, very comfortable) — switch to this if bitsandbytes-ROCm misbehaves. Both profiles are YAML configs; the training script is device- and size-agnostic.

### 6.3 Specialist architectures
- **Change detection:** Siamese encoder + lightweight MLP/diff decoder (ChangeFormer / TinyCD lineage — both pure PyTorch). Train on **LEVIR-CD (0.5 m, matches Cartosat GSD)** + **OSCD (10 m, matches Sentinel)** for resolution diversity. Backbone initialised from **SSL4EO-S12** weights via `torchgeo`.
- **Cross-modal optical–SAR:** **DOFA** (`torchgeo.models.dofa_base_patch16_224`) — a wavelength-conditioned hypernetwork patch embedding that accepts *any* channel count. One encoder covers S2/S1/Cartosat/RISAT. Fused with a deterministic **physics agreement layer** (optical NDVI/NDBI vs SAR σ⁰/VV-VH ratio) that produces explainable statements like *"NDVI 0.72 with σ⁰_VV −18 dB → vegetated, not built-up; sensors agree."* Deterministic, sensor-independent, and immune to hallucination — high value per unit of effort.
- **Grounding/segmentation:** Qwen3-VL native boxes (LoRA-tuned on DIOR-RSVG) as primary; SegFormer on LoveDA/OpenEarthMap for masks. SAM2 box→mask refinement is a **stretch goal only**.

### 6.4 Stack
| Layer | Choice |
|---|---|
| API | FastAPI · Uvicorn · Pydantic v2 · `sse-starlette` · OpenAPI 3.1 |
| Geospatial | **rasterio** (bundles GDAL — no system GDAL) · rioxarray · pyproj · shapely · geopandas · scikit-image (`phase_cross_correlation`) · opencv-python-headless |
| ML | PyTorch **ROCm 6.4** · transformers · peft · bitsandbytes · accelerate · **torchgeo** · timm · Lightning |
| Serving | llama.cpp (HIP/Vulkan) for GGUF · vLLM-ROCm optional · ONNX Runtime (CPU fallback) |
| Storage | filesystem artifact store + SQLite (jobs/traces) → Postgres if needed |
| Ops | `uv` · structlog (JSON) · pytest + pytest-asyncio · ruff + mypy |

**Explicitly rejected:** mmcv/OpenMMLab (ROCm build trap), monolithic off-the-shelf VLM as the system (disqualified by the rubric), free-form ReAct control flow (non-reproducible trace).

---

## 7. Repository Layout

```
satquery-ai/
├── pyproject.toml · .env.example · README.md
├── scripts/            rocm_env.sh · setup.sh · download_datasets.py
│                       make_synthetic_fixtures.py · export_openapi.py · serve_vlm.sh
├── configs/            app.yaml · registry.yaml · policy_table.yaml
│                       train/qlora_qwen3vl8b_rocm24g.yaml · train/lora_qwen3vl4b_bf16.yaml
├── src/satquery/
│   ├── core/           config · logging · errors · cache · geo_utils
│   ├── schemas/        manifest · compatibility · plan · tool · trace · api   ← FROZEN CONTRACT
│   ├── ingest/         reader · manifest_builder · modality · coregistration · compatibility
│   ├── render/         composites · indices · colormaps · tiling · overlays · artifact_store
│   ├── registry/       spec · registry · capability_match
│   ├── tools/          base.py + one module per tool
│   ├── agent/          query_parser · task_classifier · planner · executor · aggregator
│   ├── evidence/       fact_sheet · citation_validator · confidence
│   ├── models/         loader · llamacpp_client · hf_backend · prompts/
│   ├── trace/          builder · store
│   ├── eval/           vrsbench · rsvqa · cdvqa · metrics · report
│   └── api/            app · routers/ · deps · sse · errors
├── training/           data/builders/{bigearthnet,vrsbench,rsvqa,cdvqa,dior_rsvg}.py
│                       build_corpus.py · train_lora.py · merge_export.py
├── tests/              fixtures/ (synthetic GeoTIFFs) · unit/ · integration/ · contract/
└── data/               (gitignored) raw/ processed/ artifacts/ checkpoints/
```

---

## 8. Phased Roadmap

Ordered **risk-first**: the end-to-end system exists by Phase 3 with zero heavy ML, then each subsequent phase closes exactly one mandatory requirement. If the schedule slips, we ship a complete-but-shallower system rather than a half-wired deep one.

---

### Phase 0 — Foundations & Frozen API Contract · 2–3 days
**Objective:** repo, tooling, and every Pydantic schema exist; a **mock `/v1/analyze`** returns a realistic full `AuditTrace`. The frontend teammate is unblocked on day 2 and never waits on us again.

**Build:** `pyproject.toml` (Python 3.11, uv) · `scripts/rocm_env.sh` + `setup.sh` · all of `src/satquery/schemas/` · `core/config.py` (pydantic-settings) · `core/logging.py` (structlog JSON) · `api/app.py` with all routes wired to fixtures · `scripts/export_openapi.py` · CI (ruff, mypy, pytest).

**Verify:** `uvicorn` serves `/docs`; `POST /v1/analyze` with two sample files returns a schema-valid trace; `openapi.json` committed and handed to the teammate; `pytest tests/contract/` asserts every example payload validates.

▶ **Next prompt:** *"Execute Phase 0. Scaffold the SatQuery AI repo for Python 3.11 + uv on Fedora/ROCm, implement all Pydantic v2 schemas in src/satquery/schemas/ exactly as specified in the plan, and build a FastAPI app whose endpoints return realistic mocked AuditTrace fixtures. Export openapi.json and add contract tests."*

---

### Phase 1 — Geospatial Ingestion & Compatibility Engine · 4–5 days
**Objective:** real GeoTIFF ingestion producing `InputManifest` + `CompatibilityReport`, including automatic reprojection/resampling to a common grid. This is mandatory requirement 5's input-validation half and it is pure, fully testable, GPU-free code.

**Build:** `ingest/reader.py` (rasterio, safe dtype/nodata handling, GeoTIFF+PNG/JPEG) · `manifest_builder.py` · `modality.py` (band-count/dtype/statistics heuristics + sensor fingerprints for S1/S2/Cartosat-2S/RISAT) · `coregistration.py` (CRS reproject via pyproj, bounds IoU, GSD ratio, sub-pixel offset via `skimage.phase_cross_correlation`) · `compatibility.py` (the check battery + common-grid computation) · `scripts/make_synthetic_fixtures.py` generating known-CRS/known-offset rasters.

**Verify:** unit tests over synthetic fixtures — mismatched CRS auto-reprojects and reports the action; 30 % overlap → `WARN`; disjoint bounds → `FAIL` with a clean error; a 3 px injected shift is recovered to ±0.5 px; single 12-band S2 file classified `optical`, 2-band VV/VH classified `sar`. `POST /v1/validate` is now real.

▶ **Next prompt:** *"Execute Phase 1. Implement the geospatial ingestion and compatibility engine: reader, manifest builder, modality classifier, co-registration checks, and common-grid resampling. Add scripts/make_synthetic_fixtures.py and a full unit test suite. Wire POST /v1/validate to the real implementation."*

---

### Phase 2 — Spectral Rendering & Evidence Artifact Layer · 3–4 days
**Objective:** the shared substrate for *both* VLM input and frontend evidence. One renderer, two consumers.

**Build:** `render/composites.py` (TrueColor, FalseColorIR, SWIR, SAR-FalseColor; percentile stretch, per-sensor band maps) · `indices.py` (NDVI/NDWI/NDBI/NBR + SAR σ⁰ dB, VV/VH ratio) · `colormaps.py` · `overlays.py` (mask/bbox/heatmap compositing onto basemap) · `tiling.py` (windowed reads for large rasters) · `render/artifact_store.py` (content-addressed store, PNG + georeferenced GeoTIFF + GeoJSON, serves `/v1/artifacts/...`).

**Verify:** golden-image tests (SSIM vs committed references) for each composite; NDVI on a synthetic raster with known reflectances matches the analytic value; artifact GeoTIFFs reopen in rasterio with the correct CRS/transform; a 10 000×10 000 raster renders in <5 s and under 2 GB RAM.

▶ **Next prompt:** *"Execute Phase 2. Implement the spectral rendering pipeline (composites, indices, colormaps, overlays, tiling) and the content-addressed artifact store serving PNG/GeoTIFF/GeoJSON via /v1/artifacts. Include golden-image and analytic-value tests."*

---

### Phase 3 — Tool Registry + Agentic Controller + Audit Trace · 5–6 days ⭐
**Objective:** **the whole system runs end-to-end** — query in, grounded answer + real artifacts + real trace out — using only deterministic tools. This is the single most important milestone; everything after it is swapping in stronger tools behind a stable interface.

**Build:** `tools/base.py` protocol · `registry/` (YAML-driven registry + capability matching) · `agent/task_classifier.py` (rules + embeddings over a labelled query set) · `agent/query_parser.py` (slot extraction) · `agent/planner.py` (`configs/policy_table.yaml`: TaskType × PairType × Modality → DAG) · `agent/executor.py` (async DAG, timeouts, content-hash cache, fallback chains) · `agent/aggregator.py` · `evidence/{fact_sheet,citation_validator,confidence}.py` · `trace/builder.py` + SQLite store · SSE progress. Deterministic tools: `raster_statistics`, `spectral_index_analyzer`, `sar_backscatter_analyzer`, `image_diff_change` (CVA/log-ratio baseline), `change_statistics`. Answers via templates for now — no VLM yet.

**Verify:** 20 golden query→plan assertions (`"what changed"`+bi-temporal → change DAG; `"how many buildings"`+single → count DAG); the same input twice produces identical plans and identical scalars; a deliberately failing tool degrades to its fallback and is recorded `DEGRADED` in the trace, not a 500; `CitationValidator` rejects an injected uncited number; every response validates against the frozen schema.

▶ **Next prompt:** *"Execute Phase 3. Build the tool registry, deterministic policy-table planner, async DAG executor with caching and fallbacks, evidence aggregator with FactSheet and CitationValidator, and the AuditTrace builder with SQLite persistence. Implement the five deterministic tools and template-based answers. Add golden plan tests and reproducibility tests."*

---

### Phase 4 — VLM Serving + Grounded VQA & Captioning · 4–5 days
**Objective:** close **mandatory requirement 2** (VQA + captioning) with a zero-shot Qwen3-VL-8B, served locally on ROCm, generating answers *constrained by the FactSheet*.

**Build:** `models/loader.py` (ROCm device management, VRAM guard, lazy load/unload) · `models/hf_backend.py` (transformers, bf16) · `models/llamacpp_client.py` (GGUF via HIP/Vulkan — the offline-demo path) · `models/prompts/` (versioned templates; the FactSheet-constrained system prompt is the core IP) · tools `vlm_vqa`, `vlm_caption` · `scripts/serve_vlm.sh`.

**Verify:** measure and record zero-shot accuracy on 200 held-out VRSBench VQA items (this becomes the "before" column of the ablation table); confirm both backends produce comparable answers; VRAM stays under 22 GB; kill the network and confirm the local GGUF path still serves; assert every numeric span in generated answers resolves to a citation.

▶ **Next prompt:** *"Execute Phase 4. Integrate Qwen3-VL-8B-Instruct as a registry tool with two interchangeable backends (transformers/ROCm and llama.cpp GGUF), a versioned FactSheet-constrained prompt library, and the vlm_vqa and vlm_caption tools. Add a zero-shot VRSBench evaluation script to establish the baseline."*

---

### Phase 5 — Bi-Temporal Change Detection · 6–7 days
**Objective:** close **mandatory requirement 3** — change VQA *and* spatial change maps.

**Build:** `training/cd/` Siamese model (SSL4EO-initialised encoder + diff decoder, pure PyTorch/Lightning) · `torchgeo` datamodules for LEVIR-CD and OSCD · ROCm training run · `tools/change_detect.py` (tiled inference with overlap blending, threshold calibration) · upgraded `change_statistics` (components, area in m² via the affine transform, per-class change when semantic) · `vlm_change_vqa` tool · change-mask overlay artifacts.

**Verify:** F1 ≥ 0.88 on the LEVIR-CD test split (published range for this class of model); inference on a 2048² pair under 3 s on ROCm; masks export as georeferenced GeoTIFFs that overlay correctly in QGIS; CDVQA-style questions answered from real mask statistics; **train once at 0.5 m and once at 10 m and record cross-resolution degradation** — this is our Cartosat-transfer evidence.

▶ **Next prompt:** *"Execute Phase 5. Implement and train the Siamese change detection model on LEVIR-CD and OSCD using torchgeo datamodules and PyTorch Lightning on ROCm, then wire it into the registry as a tiled-inference tool with change statistics, georeferenced mask artifacts, and a vlm_change_vqa tool."*

---

### Phase 6 — Cross-Modal Optical–SAR Reasoning + Grounding · 5–6 days
**Objective:** close **mandatory requirements 4 and 2's grounding option**.

**Build:** `tools/crossmodal_dofa.py` (DOFA via torchgeo — wavelength-conditioned encoding of arbitrary band sets; cosine-similarity agreement map between optical and SAR embeddings) · `tools/physics_agreement.py` (the deterministic, sensor-independent NDVI/NDBI vs σ⁰/VV-VH rule layer producing explainable joint statements) · `tools/text_grounding.py` (Qwen3-VL boxes → pixel → geographic coordinates → GeoJSON) · `tools/semantic_segmenter.py` (SegFormer on LoveDA) · `tools/object_counter.py` · cross-modal policy-table entries.

**Verify:** on a BigEarthNet S1/S2 pair the system answers *"what does SAR reveal that optical does not"* citing both a backscatter statistic and a spectral index; grounding boxes round-trip pixel→WGS84→pixel within 1 px; [email protected] measured on a DIOR-RSVG subset; the physics layer correctly separates a bare-soil/built-up confusion case that optical alone gets wrong (this is the demo moment for judges).

▶ **Next prompt:** *"Execute Phase 6. Implement cross-modal optical-SAR reasoning using DOFA embeddings plus a deterministic physics-agreement layer, and add text-guided grounding (Qwen3-VL boxes to GeoJSON), semantic segmentation, and object counting tools. Extend the policy table for CROSS_MODAL pair types."*

---

### Phase 7 — Domain Adaptation: Instruction Corpus + QLoRA on ROCm · 6–8 days ⭐
**Objective:** satisfy **mandatory requirement 1** and produce the headline result. This is the novelty the judges score.

**Build:** `training/data/builders/` converting each source into one unified instruction JSONL where every sample carries its *rendered named views* (reusing Phase 2's renderer — no duplicate code):
| Source | Purpose | Target size |
|---|---|---|
| BigEarthNet-v2 (S1+S2 subset) | **mandated**; SAR + spectral physics grounding | 15–20 k |
| VRSBench | VQA + caption + grounding at VHR | 20 k |
| RSVQA-HR | counting / presence / comparison | 10 k |
| CDVQA | bi-temporal change VQA | 8 k |
| DIOR-RSVG | referring-expression grounding | 6 k |

Plus `build_corpus.py` (dedupe, stratify, mixing ratios, train/val split) · `train_lora.py` (Profile A/B configs, resumable, TensorBoard) · `merge_export.py` (merged weights + GGUF quantization for the offline path) · adapter versioning in the registry so the trace records exactly which adapter answered.

**Disk plan:** stream/subset BigEarthNet — a stratified 20 k-patch S1+S2 subset is ~15 GB, not the full ~400 GB. Write the subset selector *before* downloading.

**Verify:** the headline ablation on BigEarthNet-v2 validation — zero-shot micro-F1 vs adapted micro-F1 (target: ≈0.59 → ≈0.83, matching the published recipe); VRSBench VQA accuracy improves over the Phase-4 baseline; training completes in one overnight run within 24 GB; the adapter is ~200 MB and loads in the API without a restart; qualitative check that answers now cite sensor-specific evidence rather than generic scene description.

▶ **Next prompt:** *"Execute Phase 7. Build the unified instruction corpus from BigEarthNet-v2, VRSBench, RSVQA-HR, CDVQA, and DIOR-RSVG using the Phase 2 renderer for named multi-view inputs, then implement train_lora.py with the 24GB ROCm QLoRA profile and run the adaptation. Produce the zero-shot vs adapted ablation table."*

---

### Phase 8 — Benchmark Evaluation & Ablation · 4–5 days
**Objective:** the numbers that go in the submission deck.

**Build:** `eval/` runners for VRSBench (caption BLEU/ROUGE/CIDEr, VQA accuracy, grounding [email protected]), RSVQA (per-type accuracy + aggregate), CDVQA (change VQA accuracy) · `eval/metrics.py` · `eval/report.py` (markdown + CSV) · an **ISRO-set dry-run harness**: a synthetic Cartosat/RISAT proxy built by resampling VHR imagery to 0.65 m and simulating single-pol SAR, to smoke-test the hidden-set path before submission.

**Verify:** a four-row ablation table — generic VLM zero-shot · EarthDial-4B · our zero-shot · **our adapted + agentic tools** — across all three benchmarks; every reported number reproducible from a single `make eval` command.

▶ **Next prompt:** *"Execute Phase 8. Build the evaluation harness for VRSBench, RSVQA, and CDVQA with proper metrics, add the synthetic Cartosat-2S/RISAT proxy dry-run harness, and generate the four-row ablation report."*

---

### Phase 9 — Hardening, Demo & Submission · 4–5 days
**Objective:** it does not break in front of judges.

**Build:** error taxonomy with actionable messages · request/artifact caching · warm-start model preloading · concurrency limits and VRAM guards · a **degraded mode** where every tool has a deterministic fallback so the API never 500s · `docker-compose.yml` (rocm/pytorch base) · one-command `make demo` · a scripted 5-minute demo path exercising all four input types · architecture diagram + trace walkthrough doc for the deck.

**Verify:** kill the network mid-demo — still works; feed a corrupt GeoTIFF, a 1-band PNG, and mismatched CRS pairs — clean 4xx with useful messages, no crash; 10 concurrent requests without VRAM exhaustion; cold start to first answer under 60 s.

▶ **Next prompt:** *"Execute Phase 9. Harden the service: error taxonomy, caching, VRAM guards, degraded-mode fallbacks for every tool, docker-compose with the ROCm base image, a one-command demo script covering all four input types, and the architecture documentation."*

---

## 9. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ROCm userspace fights Fedora | **High** | `rocm/pytorch` Docker image as the escape hatch, scripted in Phase −1. Timebox native install to one day. |
| iGPU breaks ROCm enumeration | High | `HIP_VISIBLE_DEVICES=0` in every entrypoint script, asserted in `/v1/health`. |
| bitsandbytes NF4 unstable on gfx1100 | Medium | Training Profile B (Qwen3-VL-4B bf16 LoRA) is a config switch, not a rewrite. |
| 24 GB OOM on 8B multi-image training | Medium | Knobs in order: images/sample → `max_pixels` → seq length → Profile B. |
| BigEarthNet download exceeds disk | Medium | Subset selector written *before* download; target ~15 GB, not 400 GB. |
| Cartosat/RISAT domain gap at judging | **High** | Two-track corpus (§1), DOFA sensor-agnostic encoder, deterministic physics layer that needs no training, resolution-diverse CD training, and the Phase 8 proxy dry-run. |
| VLM hallucinates numbers | Medium | `CitationValidator` + `uncited_numeric_spans`; deterministic tools own all quantities. |
| Frontend blocked on backend | Medium | Contract frozen and mocked in Phase 0, day 2. |

---

## 10. End-to-End Verification

The system is done when this passes:

```bash
make demo   # runs all four scenarios against a live server
```
1. **Single optical** (12-band S2 GeoTIFF) — *"Describe this scene and identify water bodies."* → caption + NDWI-derived mask + trace showing renderer → index analyzer → VLM.
2. **Single SAR** (VV/VH GeoTIFF) — *"What kind of terrain is this?"* → answer citing σ⁰ dB and VV/VH ratio.
3. **Cross-modal pair** — *"What does the SAR show that the optical misses?"* → joint answer citing both a spectral index and a backscatter statistic, plus an agreement map.
4. **Bi-temporal pair** — *"How much built-up area appeared?"* → percentage + km² from the mask, georeferenced change-mask GeoTIFF, overlay PNG, full trace.

Plus: `pytest` green (unit + integration + contract); `make eval` reproduces the ablation table; `/v1/traces/{id}` returns a schema-valid trace for every scenario; the network can be unplugged at any point in the demo without failure.

---

## 11. Immediate Next Step

Complete **Phase −1** (Fedora + ROCm bring-up, §2) and confirm `rocminfo` reports `gfx1100` and `torch.cuda.is_available()` is `True`. Then issue the Phase 0 prompt.
