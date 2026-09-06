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

The GGUF half is the offline judging path (Master.md §8 Phase 7). It shells out
to llama.cpp's own converter rather than reimplementing it — the conversion is
architecture-specific and llama.cpp is where that knowledge is maintained. The
commands it will run are printed first, and ``--dry-run`` stops there.
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--quant", default=DEFAULT_QUANT)
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


def gguf_commands(
    merged: Path, llama_cpp: Path, quant: str = DEFAULT_QUANT
) -> list[list[str]]:
    """The two llama.cpp invocations that produce the quantised GGUF.

    Returned rather than run so ``--dry-run`` can print exactly what would
    execute, and so the composition is testable without a llama.cpp checkout.
    """
    name = merged.name
    f16 = merged.parent / f"{name}-f16.gguf"
    quantised = merged.parent / f"{name}-{quant}.gguf"
    return [
        [
            sys.executable,
            str(llama_cpp / CONVERTER),
            str(merged),
            "--outfile",
            str(f16),
            "--outtype",
            "f16",
        ],
        [str(llama_cpp / "build" / "bin" / QUANTISER), str(f16), str(quantised), quant],
    ]


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

    commands: list[list[str]] = []
    if args.gguf:
        if args.llama_cpp is None:
            print("--gguf requires --llama-cpp <checkout>", file=sys.stderr)
            return 2
        commands = gguf_commands(args.out, args.llama_cpp, args.quant)
        for command in commands:
            print("gguf       : " + " ".join(command))

    if args.dry_run:
        print("dry run: nothing merged.")
        return 0

    details = merge_adapter(base_model, args.adapter, args.out)
    write_run_manifest(
        args.out / "merge_manifest.json",
        {
            "base_model": base_model,
            "adapter": str(args.adapter),
            "merged_at": datetime.now(UTC).isoformat(),
            "gguf_quantisation": args.quant if args.gguf else None,
            **details,
        },
    )
    print(f"merged     : {details['parameters'] / 1e9:.2f} B parameters -> {args.out}")

    if commands:
        check_llama_cpp(args.llama_cpp)
        for command in commands:
            print("running    : " + " ".join(command))
            completed = subprocess.run(command, check=False)  # noqa: S603
            if completed.returncode != 0:
                print(f"gguf step failed with {completed.returncode}", file=sys.stderr)
                return completed.returncode
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
