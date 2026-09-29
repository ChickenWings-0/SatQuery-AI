# SatQuery AI — Smart India Hackathon Pitch

> **One line:** Ask a satellite image a question. Get an answer where every
> number was measured, every object is pointed at, and every gap is admitted —
> on a single consumer GPU, offline.

---

## The problem statement, as we read it

Earth observation produces more imagery than there are analysts to read it.
The promise of vision-language models is to close that gap: a planner, a
district collector, a disaster cell should be able to ask "how much of this
floodplain was built on since 2019?" and get an answer in seconds.

The reason that promise has not been kept is not fluency. Today's VLMs are
fluent. The reason is **trust**: a general model asked that question returns a
confident percentage with no way to tell whether it came from the pixels or
from the model's prior about what such answers usually look like. A number
without provenance will be quoted, acted on, and may be wrong — and in
geospatial intelligence, wrong has a coordinate.

SatQuery AI is our answer to one question: *what would it take for an AI
system's answer about a satellite scene to be safe to put in a report?*

---

## Three value propositions

### 1. Deterministic facts vs. model estimates — and the UI never confuses them

The model does not get to measure anything. Every quantity in a SatQuery
answer comes from one of two places, and the interface tells you which:

| | Where it came from | How it looks |
|---|---|---|
| **Measured fact** | A registered deterministic tool wrote it to the fact sheet — a spectral index over the scene, a Siamese change detector's mask area, an object counter's tally | A **sand citation pill** you can click. It lights the evidence view it was measured on and the KPI card carrying the same number. |
| **Model estimate** | The VLM wrote a number the CitationValidator could not resolve to any measurement — units and value both have to match | An **amber wavy underline**, a disclaimer box, and a confidence cap at 0.60. Never hidden, never stripped. |

This is enforced, not styled. The validator runs after every generation; the
confidence score is a weighted tool-agreement formula with a ladder of *named*
caps (`uncited_claims`, `degraded_execution`, `template_fallback`, …) and the
audit trace records which one clamped it. If the VLM is unavailable, a
template writes the answer from the measured evidence and says so. A judge can
click any number in the demo and see the tool, the step and the raw scalar it
came from.

The corpus enforces it too. Of the 53,098 training samples, 2,700 are
synthetic **evidence-QA** samples whose sole purpose is to teach the model to
quote the fact sheet rather than improvise. And the corpus is deduplicated by
SHA-256 and perceptual hash across every source with a hard assertion that
no training image appears in any benchmark's test split — a leaked benchmark
is worse than no benchmark, so it is a build error.

It holds up when measured: on 500 held-out samples the fine-tuned model's
citations bind to the right measurement **100 %** of the time, and **0 %** of the
numbers it writes are unresolved (`runs/eval/sq-lora-v2-full/results.md`).

### 2. Spatial grounding — it doesn't just talk; it points

"There is new construction in the north-east" is a sentence. A box drawn on
the image is a claim you can check.

Qwen3-VL has a native normalised box syntax —
`<|object_ref_start|>warehouse<|object_ref_end|><|box_start|>(560,180),(800,420)<|box_end|>`
on a 0–1000 frame. We fine-tuned on it, we parse it on the server with one
serialiser shared by training and inference, and the frontend ports that
parser to TypeScript and draws the result as an SVG whose `viewBox` *is* the
model's coordinate frame. No pixel arithmetic, no drift between what was
parsed and what is drawn. The boxes sit inside the zoom transform, follow the
A/B swipe, and are aligned to the raster's letterboxed rectangle rather than
its container, so they are still right on a wide screen.

The same principle runs through the evidence strip: every thumbnail is the
exact view the model was shown, labelled with the exact string it read —
`Image 3 (optical false colour infrared, pre-change, Sentinel-2, 2019-04-12)`.
When the answer cites NDBI, the viewer switches to the NDBI heatmap. The
model's evidence is the user's evidence.

### 3. Hardware efficiency — a sovereign, offline deployment on one 24 GB card

Everything in the demo runs on a single consumer GPU with the network cable
unplugged: the 8B model, the change detector, the segmenter, the API and the
console. No cloud inference, no imagery leaves the machine.

That took engineering, and some of it is worth telling:

- **Fine-tuning fit in 24 GB** via QLoRA — NF4 base weights, r=16 LoRA, batch 1
  with 16-step accumulation, one epoch over 53k samples from five sources in
  41 hours at a 13.4 GiB peak. Answer-token accuracy on held-out data went from
  26 % (base model) to 82 %.
- **Serving in bf16, not NF4, on purpose.** We found that 4-bit *generation*
  is broken on this ROCm/bitsandbytes combination — the base model answers
  nonsense in NF4 and correctly in bf16 — while 4-bit *training* is fine,
  which is why a training run can finish without revealing it. The serving path
  therefore loads bf16 base weights (~16.4 GB) with the adapter over them,
  ~18.7 GiB peak with six evidence views, and puts a VRAM guard in front of
  the load so a warm llama.cpp server on the same card produces a clear
  refusal rather than a half-loaded model.
- **A lightweight path for the worst day.** The fine-tuned model merged and
  quantised to a 5 GB Q4_K_M GGUF runs under llama.cpp on an 8 GB laptop GPU with
  no network and no Python ML stack. Same API, same UI, honestly reported.
- **The GPU is never touched until it has to be.** Pre-flight — sensor, CRS,
  resolution, footprint, ten compatibility checks — runs on CPU before a
  question is typed, so an incompatible pair is refused for free.

For SIH this means the system is deployable in a district office or a field
unit on hardware that already exists, and the data stays where it was
collected.

---

## What the demo shows, in order

1. Drop two Sentinel-2 scenes. Pre-flight fills in sensor, GSD, CRS,
   acquisition dates and passes ten checks — before any model loads.
2. Ask "How much built-up area appeared?" Watch the pipeline: the plan appears,
   then evidence views land in the strip one by one as each tool finishes,
   streamed over SSE.
3. The answer arrives. Click **5.34 %** — the viewer jumps to the change
   overlay, the *Scene changed* KPI lights, the pipeline dialog is armed on
   the `change_statistics` step.
4. Ask "Where are the new buildings?" — boxes appear on the raster, labelled,
   following the zoom and the swipe.
5. Point at the one amber-underlined number. Explain that it is the system
   telling you it could not verify that value, and that this is a feature.
6. Open the processing pipeline. Every step, its status, its outputs, the
   confidence formula and which cap applied.
7. Pull the network cable. Do it again.

---

## Why this is more than a wrapper around a model

- A **frozen API contract** (OpenAPI 1.0) with 620+ backend tests, 350+
  frontend tests and offline Playwright runs in CI, including a WCAG contrast
  contract the design tokens must pass to build.
- **Measured, not claimed:** a 500-sample held-out benchmark across five sources —
  80.7 % accuracy, 48.9 % grounding recall at IoU 0.5, 100 % citation precision —
  with the worst cases rendered for an honest-failure slide.
- A **data engine** that reads three public benchmarks correctly where the
  standard loader silently does not (RSVQA-HR's `active` flag alone would have
  leaked 330,324 test rows into training).
- A **policy-driven planner**: tasks map to DAGs of registered tools with
  declared capabilities, so adding a sensor or a tool is a registry entry, not
  a rewrite.
- An **audit trace** for every run — inputs, plan, executions, fact sheet,
  answer, citations, confidence — stored and retrievable by id.

## What is next

- Cartosat-2S / RISAT adaptation through the augmentation track already in
  the data plan — the sensors of the hidden evaluation set.
- Multi-label scene classification, the benchmark's weakest task (50 % label
  F1), and more referring-expression grounding data (DIOR-RSVG, once access is
  granted).
- A 10 m change-detection model (OSCD) beside the 0.5 m LEVIR-CD one, to
  measure how change maps transfer across resolutions.

---

*SatQuery AI — From Space to Answers.*
