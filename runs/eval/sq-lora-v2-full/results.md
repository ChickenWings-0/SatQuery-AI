# SatQuery AI — validation benchmark · sq-lora-v2-full

*2026-09-16T17:48:26+00:00 · 500 held-out samples · backend `hf` · adapter `runs/sq-lora-v2-full/adapter` · git `b2b87e95867d` · seed 42*

## Per source (slide table)

| Source | n | Accuracy | Grounding R@0.5 | Mean IoU | Citation precision | Uncited-number rate |
|---|---|---|---|---|---|---|
| BigEarthNet-v2 | 100 | 53.3 % | 40.0 % | 41.6 % | 100.0 % | 0.0 % |
| CDVQA | 100 | 77.0 % | — | — | — | — |
| Evidence QA | 100 | 87.0 % | — | — | 100.0 % | 0.0 % |
| RSVQA-HR | 100 | 91.0 % | — | — | 100.0 % | 0.0 % |
| VRSBench | 100 | 88.0 % | 56.0 % | 52.6 % | 100.0 % | 0.0 % |
| **All** | 500 | 80.7 % | 48.9 % | 47.7 % | 100.0 % | 0.0 % |

Accuracy is exact match on closed answers (VQA, change VQA, count, scene labels); grounding recall is at IoU ≥ 0.5 with mean IoU over reference boxes; citation precision is the share of `[tool.scalar]` markers the validator bound to that measurement; the uncited-number rate is the share of numbers in the answer that resolved to nothing. `—` means the metric does not apply to that group.

## Per task

| Task | n | Accuracy | Label F1 | Grounding R@0.5 | Mean IoU | Box format OK | Citation precision | Uncited-number rate | Fact recall | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CAPTION | 25 | — | — | — | — | — | — | — | — | 13.7 % | 36.7 % |
| CHANGE_VQA | 100 | 77.0 % | — | — | — | — | — | — | — | — | — |
| COUNT | 75 | 100.0 % | — | — | — | — | 100.0 % | 0.0 % | — | — | — |
| CROSS_MODAL_COMPARE | 20 | — | — | — | — | — | 100.0 % | 0.0 % | 100.0 % | 80.5 % | 90.0 % |
| CROSS_MODAL_VQA | 70 | 81.4 % | — | — | — | — | 100.0 % | 0.0 % | 100.0 % | — | — |
| GROUNDING | 45 | — | — | 48.9 % | 47.7 % | 100.0 % | — | — | — | — | — |
| SCENE_CLASSIFY | 20 | 5.0 % | 50.2 % | — | — | — | — | — | — | — | — |
| VQA | 145 | 83.4 % | — | — | — | — | 100.0 % | 0.0 % | 100.0 % | — | — |

## Run

- mean latency 5834 ms/sample · 0 generation errors · 0 truncated decodes
- corpus `data/processed/corpus/v2-full/val.jsonl` · 100 per source · wall 2953 s
- predictions `runs/eval/sq-lora-v2-full/predictions.jsonl` · sample ids `runs/eval/sq-lora-v2-full/sample_ids.json`
