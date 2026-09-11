# SatQuery AI

**From Space to Answers — geospatial intelligence that shows its work.**

SatQuery AI is a vision-language pipeline for satellite imagery. You upload one
or two scenes — optical, SAR, single-date or bi-temporal — ask a question in
plain language, and get an answer in which **every number is traced back to the
tool that measured it**, and every located object is **drawn on the image** as a
bounding box. It runs end-to-end on a single consumer GPU, offline.

Built for the Smart India Hackathon.

---

## The problem

Large vision-language models can *describe* a satellite scene fluently. They
cannot be trusted to *measure* one. Ask a general-purpose VLM "how much
built-up area appeared between these two dates?" and you get a confident
paragraph with a confident percentage — and no way to know whether that
percentage came from the pixels or from the model's prior about what such
answers usually look like.

For an analyst, a planner or a disaster-response cell, a number with no
provenance is worse than no number: it will be quoted, and it may be wrong.

SatQuery AI is built on one refusal: **the model never gets to invent a
measurement.**

## How it answers

1. **Pre-flight before the GPU.** Both uploads are ingested, their sensor,
   CRS, resolution and footprint read from metadata, and a compatibility
   report says what questions this pair can honestly support — before anyone
   types.
2. **Deterministic tools measure.** A planner turns the question into a DAG of
   registered tools — spectral indices (NDVI/NDBI/NDWI), SAR backscatter, a
   Siamese change detector, a semantic segmenter, an object counter, raster
   statistics — and every scalar they produce lands in a **fact sheet**.
3. **The VLM writes, the validator checks.** A Qwen3-VL-8B model, fine-tuned on
   65,000 remote-sensing samples, is shown the rendered evidence views and the
   fact sheet and asked to write the answer. A **CitationValidator** then
   resolves every numeric span in that answer against the fact sheet — units
   must match, not just magnitude. Numbers that resolve become citations.
   Numbers that don't are **flagged, never hidden**, and cap the confidence
   score.
4. **The interface makes the audit visible.** Citations are clickable pills
   that light the evidence view and the KPI card they came from. Ungrounded
   numbers get an amber wavy underline and a disclaimer. Bounding boxes the
   model emits in Qwen's native `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`
   syntax are parsed and drawn over the raster, scaling with zoom.

## Key features

- **Bi-temporal change analysis** with an A/B swipe viewer, T1/T2 date pills
  and change-mask overlays.
- **Multi-spectral evidence strip** — true colour, false-colour IR, SWIR,
  NDVI/NDWI/NDBI heatmaps, SAR composites — each one the exact view the model
  was shown, labelled with the exact string it saw.
- **Spatial grounding.** The answer doesn't just talk; it points. Boxes are
  parsed with a TypeScript port of the backend's own serialiser so what is
  drawn is byte-for-byte what was parsed.
- **Confidence with reasons.** A weighted tool-agreement score with a ladder of
  named caps (`uncited_claims → 0.60`, `template_fallback → 0.65`, …) — the
  trace says *which* rule clamped it.
- **Live pipeline.** Jobs stream over Server-Sent Events; evidence appears in
  the tray the moment the tool that produced it finishes. A lazily-loaded DAG
  view shows every step, its status and its outputs.
- **Honest degradation.** If the VLM is unavailable, a deterministic template
  writes the answer from the measured evidence and says so. If a tool fails,
  the step is marked degraded and the confidence is capped — nothing is
  silently dropped.
- **Runs on 24 GB of consumer VRAM**, with no network, on ROCm or CUDA.

## Tech stack

| Layer | What |
|---|---|
| Base model | Qwen3-VL-8B-Instruct, QLoRA fine-tune (r=16, α=32, NF4 training, 19-hour single-epoch run) |
| Training corpus | 65k samples: BigEarthNet-v2 (Sentinel-1/2), VRSBench, RSVQA-HR, CDVQA, DIOR-RSVG, synthetic evidence-QA — SHA-256 + pHash deduplicated with hard test-split quarantine |
| Serving | Python 3.11, FastAPI, `transformers` in **bf16** with the LoRA adapter applied at load; llama.cpp GGUF as the lightweight fallback |
| Geospatial | rasterio, rioxarray, pyproj, shapely, scikit-image, OpenCV |
| Deterministic tools | Siamese change detector (PyTorch), SegFormer segmenter, DOFA cross-modal fusion, spectral/SAR analysers |
| API | REST + SSE (`/v1/validate`, `/v1/jobs`, `/v1/jobs/{id}/events`, `/v1/traces`, `/v1/artifacts`), frozen OpenAPI 1.0 contract |
| Frontend | React 19, Vite 6, TypeScript 5.9, Tailwind v4, Zustand 5, react-compare-slider, react-zoom-pan-pinch, @xyflow/react |
| Testing | 450+ pytest cases (contract, unit, integration); 194 vitest cases including a WCAG contrast contract enforced against `theme.css` |

## Repository map

```
src/satquery/
  ingest/      upload → InputManifest (sensor, CRS, GSD, bands, acquisition)
  render/      the frozen view catalogue and the labels the model sees
  agent/       task classifier, planner, DAG executor, aggregator, SSE events
  tools/       every registered measurement tool
  evidence/    fact sheet, CitationValidator, confidence
  models/      VLM backends (transformers bf16, llama.cpp) and prompt builder
  api/         FastAPI app, routers, in-process job store
  training/    corpus builder, local dataset readers, QLoRA + CD trainers
frontend/      the mission-control console
configs/       registry, policy table, band aliases, training profiles
DOCS/          API_CONTRACT, AGENT_POLICY_DAG, DATA_ADAPTATION_PLAN, Master
scripts/       fetch_sources, build_corpus, render_views, train_vlm, serve_vlm
```

## Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — how the system works end-to-end
- [`SETUP_GUIDE.md`](./SETUP_GUIDE.md) — run it locally
- [`HACKATHON_PITCH.md`](./HACKATHON_PITCH.md) — the SIH presentation narrative
- `DOCS/API_CONTRACT.md` — the frozen wire format the frontend is generated from

## Status

Core engineering complete: data pipeline, fine-tuned adapter, bf16 serving,
async streaming API, and the dark-mode mission-control frontend. Phase 8
(benchmark evaluation on the quarantined test splits) is the next milestone.
