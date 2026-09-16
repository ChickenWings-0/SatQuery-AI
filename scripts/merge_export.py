#!/usr/bin/env python
"""Merge the Phase 7 LoRA adapter into the base model and export it for GGUF.

    uv run python scripts/merge_export.py --adapter runs/sq-lora-v1/adapter \
        --out models/sq-lora-v1-merged
    uv run python scripts/merge_export.py --adapter runs/sq-lora-v1/adapter \
        --out models/sq-lora-v1-merged --gguf --llama-cpp ../llama.cpp

Two things about the merge are easy to get wrong and expensive to get wrong:

* **Merge in bf16 on the CPU, never onto the quantised weights.** The training
  run held the backbone in NF4; folding ``B @ A`` into a 4-bit tensor and
  de-quantising back would bake the quantisation error into the exported
  weights, and the resulting model is quietly worse than the adapter it came
  from. So the base is re-loaded at full precision, on the host, where 16 GB of
  system RAM is cheap and 16 GB of VRAM is not.
* **The processor travels with the weights.** ``max_pixels`` and the chat
  template are part of the contract the adapter was trained under; a merged
  directory without them serves the same weights against a different input
  distribution.

The GGUF half is the offline judging path (Master.md §8 Phase 7; ROADMAP Track
2). It shells out to llama.cpp's own converter rather than reimplementing it —
the conversion is architecture-specific and llama.cpp is where that knowledge
is maintained. Three invocations, in order:

1. ``convert_hf_to_gguf.py --outtype bf16`` — the language model, ~16 GB;
2. ``convert_hf_to_gguf.py --mmproj --outtype f16`` — the vision tower and
   merger as a separate ``mmproj`` file (~1.1 GB). **Never quantised**: the
   vision side is what breaks grounding first, and it is small;
3. ``llama-quantize <bf16> <Q4_K_M> Q4_K_M`` — the ~5 GB file the demo laptop
   loads. ``--quant`` chooses the type; ``--mmproj-outtype`` the projector's.

The commands are printed first, and ``--dry-run`` stops there. Every file the
run writes is listed in ``SHA256SUMS`` next to it, so the copy on the USB
stick can be verified on the demo machine with one command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satquery.training.vlm.qlora import (  # noqa: E402
    DEFAULT_PROFILE_PATH,
    ProfileError,
    load_profile,
    write_run_manifest,
)

DEFAULT_QUANT: str = "Q4_K_M"
"""The offline path's quantisation. Q4_K_M is the smallest quantisation that has
not measurably moved answers in our own spot checks; anything below it starts
costing citation fidelity, which is the one thing this system cannot trade."""

DEFAULT_MMPROJ_OUTTYPE: str = "f16"
"""The multimodal projector's precision. It is ~1.1 GB at f16 and is the part
of the model whose quantisation error shows up as boxes on the wrong object;
the language model, not the vision tower, is where the bytes are saved."""

DEFAULT_LLM_OUTTYPE: str = "bf16"
"""The intermediate the quantiser reads. bf16 is what the merge produced, so
converting to it loses nothing; f16 would clip Qwen's activations."""

DEFAULT_MAX_VIEWS: int = 3
"""``SATQUERY_VLM_MAX_VIEWS`` for the quantised deployment. Six 448 px views are
~6,000 image tokens; the 8 GB laptop serves the GGUF at 8192 context, so it
gets three. Recorded in the manifest so the serving side and the merge agree
on what this export was sized for."""

CONVERTER: str = "convert_hf_to_gguf.py"
QUANTISER: str = "llama-quantize"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", type=Path, required=True, help="LoRA adapter directory.")
    parser.add_argument("--out", type=Path, required=True, help="Merged model directory.")
    parser.add_argument(
        "--base",
        default=None,
        help="Base model id. Defaults to the run manifest's, then the profile's.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--gguf", action="store_true", help="Also convert and quantise to GGUF.")
    parser.add_argument("--llama-cpp", type=Path, default=None, help="llama.cpp checkout.")
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        help=f"llama-quantize type for the language model (default {DEFAULT_QUANT}).",
    )
    parser.add_argument(
        "--mmproj-outtype",
        default=DEFAULT_MMPROJ_OUTTYPE,
        choices=["f16", "bf16", "f32"],
        help=f"Precision of the vision projector GGUF (default {DEFAULT_MMPROJ_OUTTYPE}).",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=None,
        help=(
            "SATQUERY_VLM_MAX_VIEWS the export is sized for; recorded in the manifest. "
            f"Defaults to the environment variable, else {DEFAULT_MAX_VIEWS} with --gguf "
            "and 6 without."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be merged and print the GGUF commands, then stop.",
    )
    return parser.parse_args(argv)


def resolve_base_model(adapter: Path, config: Path, override: str | None) -> str:
    """Decide which base model the adapter belongs to.

    Precedence is explicit flag, then the run manifest written beside the
    adapter, then the training profile. Merging an adapter into the wrong base
    produces a model that loads, runs, and is nonsense.

    Raises:
        ProfileError: No source names a base model.
    """
    if override:
        return override
    manifest = adapter.parent / "run_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        named = payload.get("profile", {}).get("base_model")
        if named:
            return str(named)
    try:
        return load_profile(config).base_model
    except ProfileError as error:
        raise ProfileError(
            f"cannot tell which base model {adapter} belongs to; pass --base"
        ) from error


def gguf_outputs(
    merged: Path, quant: str = DEFAULT_QUANT, mmproj_outtype: str = DEFAULT_MMPROJ_OUTTYPE
) -> dict[str, Path]:
    """Where the three GGUF steps write, keyed ``llm``, ``mmproj``, ``quantised``."""
    name = merged.name
    return {
        "llm": merged.parent / f"{name}-{DEFAULT_LLM_OUTTYPE}.gguf",
        "mmproj": merged.parent / f"{name}-mmproj-{mmproj_outtype}.gguf",
        "quantised": merged.parent / f"{name}-{quant}.gguf",
    }


def gguf_commands(
    merged: Path,
    llama_cpp: Path,
    quant: str = DEFAULT_QUANT,
    mmproj_outtype: str = DEFAULT_MMPROJ_OUTTYPE,
) -> list[list[str]]:
    """The three llama.cpp invocations that produce the deployable GGUF pair.

    Returned rather than run so ``--dry-run`` can print exactly what would
    execute, and so the composition is testable without a llama.cpp checkout.
    """
    outputs = gguf_outputs(merged, quant, mmproj_outtype)
    converter = [sys.executable, str(llama_cpp / CONVERTER), str(merged)]
    return [
        [*converter, "--outfile", str(outputs["llm"]), "--outtype", DEFAULT_LLM_OUTTYPE],
        [*converter, "--mmproj", "--outfile", str(outputs["mmproj"]), "--outtype", mmproj_outtype],
        [
            str(llama_cpp / "build" / "bin" / QUANTISER),
            str(outputs["llm"]),
            str(outputs["quantised"]),
            quant,
        ],
    ]


def sha256_of(path: Path) -> str:
    """Streamed SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(paths: Sequence[Path], out: Path) -> Path:
    """Write ``SHA256SUMS`` in ``sha256sum`` format for every existing *path*.

    Paths are recorded relative to *out*'s parent, which is where the file
    lives, so ``sha256sum -c SHA256SUMS`` (or ``Get-FileHash`` on Windows, see
    ``scripts/demo_laptop/README.md``) works from that directory.
    """
    lines = []
    for path in paths:
        if path.is_file():
            lines.append(f"{sha256_of(path)}  {os.path.relpath(path, out.parent)}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def check_llama_cpp(llama_cpp: Path) -> None:
    """Verify the checkout has the two tools this needs.

    Raises:
        ProfileError: The converter or the quantiser is missing.
    """
    converter = llama_cpp / CONVERTER
    quantiser = llama_cpp / "build" / "bin" / QUANTISER
    missing = [str(path) for path in (converter, quantiser) if not path.exists()]
    if missing:
        raise ProfileError(
            f"llama.cpp checkout at {llama_cpp} is missing {missing}; build it with "
            "`cmake -B build && cmake --build build --config Release -j`"
        )


def merge_adapter(base_model: str, adapter: Path, out: Path) -> dict[str, Any]:
    """Load the base in bf16 on the CPU, fold the adapter in, and save it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")

    processor_source = adapter if (adapter / "preprocessor_config.json").is_file() else base_model
    auto_processor: Any = AutoProcessor  # optional extra; typed here, Any elsewhere
    auto_processor.from_pretrained(str(processor_source)).save_pretrained(out)
    return {
        "parameters": sum(parameter.numel() for parameter in merged.parameters()),
        "processor_source": str(processor_source),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Merge, export, and optionally quantise. Returns a process exit code."""
    args = parse_args(argv)
    if not args.adapter.is_dir():
        print(f"adapter directory {args.adapter} does not exist", file=sys.stderr)
        return 2

    base_model = resolve_base_model(args.adapter, args.config, args.base)
    print(f"base model : {base_model}")
    print(f"adapter    : {args.adapter}")
    print(f"merged out : {args.out}")

    max_views = resolve_max_views(args.max_views, gguf=args.gguf)
    print(f"max views  : {max_views}  (SATQUERY_VLM_MAX_VIEWS for this export)")

    commands: list[list[str]] = []
    outputs: dict[str, Path] = {}
    if args.gguf:
        if args.llama_cpp is None:
            print("--gguf requires --llama-cpp <checkout>", file=sys.stderr)
            return 2
        outputs = gguf_outputs(args.out, args.quant, args.mmproj_outtype)
        commands = gguf_commands(args.out, args.llama_cpp, args.quant, args.mmproj_outtype)
        for command in commands:
            print("gguf       : " + " ".join(command))

    if args.dry_run:
        print("dry run: nothing merged.")
        return 0

    details = merge_adapter(base_model, args.adapter, args.out)
    manifest = args.out / "merge_manifest.json"
    payload: dict[str, Any] = {
        "base_model": base_model,
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_digest(args.adapter),
        "merged_at": datetime.now(UTC).isoformat(),
        "serving": {
            "SATQUERY_VLM_MAX_VIEWS": max_views,
            "SATQUERY_VLM_IMAGE_MIN_TOKENS": 1024,
            "context": 8192 if args.gguf else 16384,
        },
        "gguf": (
            {
                "quantisation": args.quant,
                "llm_outtype": DEFAULT_LLM_OUTTYPE,
                "mmproj_outtype": args.mmproj_outtype,
                "files": {key: str(path) for key, path in outputs.items()},
            }
            if args.gguf
            else None
        ),
        **details,
    }
    write_run_manifest(manifest, payload)
    print(f"merged     : {details['parameters'] / 1e9:.2f} B parameters -> {args.out}")

    if commands:
        check_llama_cpp(args.llama_cpp)
        for command in commands:
            print("running    : " + " ".join(command))
            completed = subprocess.run(command, check=False)  # noqa: S603
            if completed.returncode != 0:
                print(f"gguf step failed with {completed.returncode}", file=sys.stderr)
                return completed.returncode
        # The bf16 intermediate is 16 GB the laptop never loads; the quantised
        # file and the projector are what ship.
        if outputs["llm"].is_file() and outputs["quantised"].is_file():
            outputs["llm"].unlink()
            print(f"removed    : {outputs['llm']} (intermediate)")

    checksums = write_checksums(
        [manifest, *(path for key, path in outputs.items() if key != "llm")],
        args.out.parent / "SHA256SUMS",
    )
    print(f"checksums  : {checksums}")
    return 0


def resolve_max_views(flag: int | None, gguf: bool) -> int:
    """The view ceiling this export is sized for.

    Explicit flag, then ``SATQUERY_VLM_MAX_VIEWS`` from the environment, then
    the deployment default: three for a quantised export (the 8 GB laptop),
    six for a bf16 merge (the 24 GB box).
    """
    if flag is not None:
        return flag
    from_env = os.environ.get("SATQUERY_VLM_MAX_VIEWS")
    if from_env and from_env.strip().isdigit():
        return int(from_env)
    return DEFAULT_MAX_VIEWS if gguf else 6


def adapter_digest(adapter: Path) -> str | None:
    """SHA-256 of ``adapter_model.safetensors`` — which adapter this merge came from."""
    weights = adapter / "adapter_model.safetensors"
    return sha256_of(weights) if weights.is_file() else None


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
