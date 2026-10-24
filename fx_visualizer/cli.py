"""Export interactive FX graphs from BB repros or saved compiler captures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .api import visualize_repro
from .rendering import render_capture, render_matmul_comparison, write_index


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repro", type=Path, help="Any BB directory containing repro.py, or its repro.py path")
    source.add_argument("--capture", type=Path, action="append", help="Saved capture JSON; repeat to render several graphs without PyTorch")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("fx", "scheduler", "matmul_comparison"), default="fx")
    parser.add_argument("--configuration", default="0", help="Saved configuration index or key for --repro")
    parser.add_argument("--title", help="Display title for one graph")
    parser.add_argument("--full-compile", action="store_true", help="Compile and execute instead of stopping before codegen and autotuning")
    parser.add_argument("--triton-matmul", action="store_true", help="Force Triton GEMM templates in scheduler mode")
    parser.add_argument("--inductor-config", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args(argv)
    if args.mode == "fx" and (args.full_compile or args.triton_matmul or args.inductor_config):
        parser.error("Compiler options require --mode scheduler or --mode matmul_comparison")
    if args.mode == "matmul_comparison" and (args.triton_matmul):
        parser.error("matmul_comparison controls its template and native matmul lowering modes")
    if args.capture:
        if args.configuration != "0" or args.full_compile or args.triton_matmul or args.inductor_config:
            parser.error("Compiler and input configuration options require --repro")
        if args.mode == "matmul_comparison":
            if len(args.capture) != 2:
                parser.error("matmul_comparison requires exactly two captures: template/native")
            artifacts = [render_matmul_comparison(args.capture[0], args.capture[1], args.output_dir, title=args.title)]
        else:
            if args.title and len(args.capture) != 1:
                parser.error("--title requires exactly one --capture")
            artifacts, names = [], set()
            for path in args.capture:
                capture = json.loads(path.read_text())
                name = Path(capture.get("name") or path.parent.name).name
                if name in ("", ".", "..") or name in names:
                    parser.error(f"Invalid or duplicate graph name {name!r}; use separate output directories")
                names.add(name)
                destination = args.output_dir / name if len(args.capture) > 1 else args.output_dir
                artifacts.append(render_capture(capture, destination, title=args.title))
    else:
        config = {}
        for option in args.inductor_config:
            key, separator, value = option.partition("=")
            if not key or not separator:
                parser.error(f"Expected KEY=JSON, got {option!r}")
            try:
                config[key.replace("-", "_")] = json.loads(value)
            except json.JSONDecodeError as error:
                parser.error(f"Invalid JSON for {key}: {error.msg}")
        try:
            configuration = int(args.configuration) if args.configuration.lstrip("-").isdigit() else args.configuration
            artifacts = [visualize_repro(
                args.repro, args.output_dir, configuration=configuration, mode=args.mode,
                title=args.title, config=config, schedule_only=not args.full_compile,
                triton_matmul=args.triton_matmul,
            )]
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    index = write_index(artifacts, args.output_dir)
    print(json.dumps({"index": str(index), "graphs": artifacts}, indent=2))
