"""Run a saved backend artifact and its winning configurations without retuning."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

from benchmark_mm_backends import environment, load_module, make_benchmark_inputs, PYTHON, ROOT
from render_mm_backend_report import reassess_accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro/benchmark")
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warm-cache", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not os.environ.get("MM_REPLAY_ENV_READY"):
        env = environment(args.gpu, args.output)
        env["MM_REPLAY_ENV_READY"] = "1"
        os.execvpe(PYTHON, [PYTHON, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)
    import fcntl
    import subprocess
    import time
    lock = (args.output / f"gpu_{args.gpu}.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    while subprocess.run(["nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10).stdout.strip():
        time.sleep(3)
    import statistics
    import torch
    import triton
    from torch._inductor import config
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    if torch.version.cuda is None:
        version = torch._C._cuda_getCompiledVersion()
        torch.version.cuda = f"{version // 1000}.{version % 1000 // 10}"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    case_dir = args.output / "results" / args.case
    backend_dir = case_dir / args.backend
    row = json.loads((backend_dir / "result.json").read_text())
    case = json.loads((case_dir / "case.json").read_text())
    reassess_accuracy(row, case["first_shape"])
    if row["status"] != "ok":
        raise RuntimeError(f"Saved backend is not accuracy-valid: {row['status']}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(backend_dir / "cache/inductor")
    os.environ["TRITON_CACHE_DIR"] = str(backend_dir / "cache/triton")
    os.environ["HELION_CACHE_DIR"] = str(backend_dir / "cache/helion")
    torch.manual_seed(case["seed"])
    torch.cuda.manual_seed_all(case["seed"])
    inputs = make_benchmark_inputs(case["first_shape"])
    snapshots = []
    if case["mutating_storages"]:
        seen = set()
        for value in inputs:
            if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() not in seen:
                storage = value.untyped_storage()
                view = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
                snapshots.append((view, view.clone()))
                seen.add(storage.data_ptr())
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    source_path = backend_dir / "compiled.py"
    if args.backend == "native_helion":
        source_path = backend_dir / "standalone_tuned.py"
        if not source_path.exists():
            tree = ast.parse((backend_dir / "compiled.py").read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in row["winning_configs"]["helion"]:
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            chosen = row["winning_configs"]["helion"][node.name]["config"]
                            decorator.keywords = [kw for kw in decorator.keywords if kw.arg not in {"config", "configs"}]
                            decorator.keywords.append(ast.keyword(arg="config", value=ast.parse(f"helion.Config(**{chosen!r})", mode="eval").body))
            source_path.write_text(ast.unparse(ast.fix_missing_locations(tree)).replace("parents[4]", "parents[5]") + "\n")
    original = CachingAutotuner.autotune_to_one_config

    def pin_triton(autotuner, *a, **kw):
        saved = row["winning_configs"]["triton"].get(autotuner.fn.__name__)
        if saved:
            winner = saved[0]
            matching = [launcher for launcher in autotuner.launchers if launcher.config.kwargs == winner["kwargs"] and launcher.config.num_warps == winner["num_warps"] and launcher.config.num_stages == winner["num_stages"]]
            if matching:
                autotuner.launchers = matching[:1]
                return None
        raise RuntimeError(f"No saved winner for {autotuner.fn.__name__}; refusing to retune")

    with config.patch(row["compile_options"]), patch.object(CachingAutotuner, "autotune_to_one_config", pin_triton), patch.object(torch.cuda.CUDAGraph, "capture_begin", side_effect=RuntimeError("CUDA Graph disabled")):
        module = load_module("replay_" + args.case + args.backend, source_path)
        order = module.COMPILED_INPUT_ORDER if args.backend == "native_helion" else row["compiled_input_order"]
        fn = lambda: module.call([inputs[i] for i in order])
        fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.calls):
            for view, snapshot in snapshots:
                view.copy_(snapshot)
            if not args.warm_cache:
                flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000)
    print(json.dumps({"case": args.case, "backend": args.backend, "median_event_us": statistics.median(samples), "retuning": False, "timing_regime": "warm cache, no CUDA graph" if args.warm_cache else "64 MiB L2 flush outside every CUDA-event sample, no CUDA graph"}, indent=2))


if __name__ == "__main__":
    main()
