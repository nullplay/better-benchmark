"""Benchmark the first shape of each MM repro, without CUDA graphs."""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("aten", "triton", "nvgemm", "native_triton", "native_helion")
PYTHON = "/home/jaeyeonw/pytorch-nullplay/venv/bin/python"
TORCH_ROOT = "/home/jaeyeonw/pytorch-nullplay"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")
    tmp.replace(path)


def environment(gpu, output):
    env = os.environ.copy()
    vendor = Path(PYTHON).parent.parent / "lib/python3.12/site-packages/nvidia"
    env.update(
        PYTHONPATH=TORCH_ROOT + ":" + str(ROOT),
        LD_LIBRARY_PATH=":".join(str(vendor / name / "lib") for name in ("cu13", "cudnn", "nccl", "cusparselt"))
        + ":/usr/local/cuda-13/targets/x86_64-linux/lib",
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        TORCHINDUCTOR_CUDAGRAPHS="0",
        TORCHINDUCTOR_AUTOTUNE_CUDAGRAPH_BENCHMARKING="0",
        HELION_BENCHMARK_CUDAGRAPH="0",
        HELION_AUTOTUNE_EFFORT="full",
        HELION_AUTOTUNE_PRECOMPILE_JOBS="4",
        HELION_AUTOTUNE_LOG=str(output / "helion_autotune.jsonl"),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_CACHE_PATH=str(output / "cache/cuda_driver"),
    )
    return env


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_benchmark_inputs(shape_config):
    import torch
    from repro_harness import make_inputs_from_config

    inputs = make_inputs_from_config(shape_config)
    for position, (value, spec) in enumerate(zip(inputs, shape_config["inputs"])):
        offset = spec.get("storage_offset", 0) or 0
        if isinstance(value, torch.Tensor) and offset and value.storage_offset() == 0 and spec.get("alias_group") is None:
            storage = value.untyped_storage()
            original = torch.empty(0, dtype=value.dtype, device=value.device).set_(storage, 0, (storage.nbytes() // value.element_size(),), (1,))
            padded = torch.empty(original.numel() + offset, dtype=value.dtype, device=value.device)
            padded[:offset].zero_()
            padded[offset:].copy_(original)
            inputs[position] = padded.as_strided(value.shape, value.stride(), offset)
    return inputs


def gpu_state(gpu):
    fields = "index,uuid,name,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used"
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=10,
    )
    return dict(zip(fields.split(","), (part.strip() for part in result.stdout.strip().split(","))))


def finished(path):
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("status") not in {"pending", "compiled"}


def selected_kernel_names(source):
    names = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(("triton_", "helion_")):
            names.append(node.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "async_compile" and func.attr in {"triton", "cuda", "nv_universal_gemm"}:
                names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
    return names


def vendor_gemm_calls(source):
    return [
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "extern_kernels"
        and node.func.attr in {"mm", "addmm", "bmm", "baddbmm", "mv", "addmv", "_scaled_mm", "_int_mm"}
    ]


def worker(args):
    import torch
    from torch._inductor import config, metrics
    from torch._inductor.graph import GraphLowering
    from torch._inductor.compile_fx import compile_fx
    from torch._dynamo.source import LocalSource
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.select_algorithm import add_feedback_saver, clear_feedback_savers
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten, tree_map
    import helion
    from helion.runtime.kernel import BoundKernel
    from repro_harness import load_shape_configs

    path = ROOT / "mm_repro/canonical" / args.case / "repro.py"
    case_dir = args.output / "results" / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    original_cuda_version = torch.version.cuda
    if original_cuda_version is None:
        compiled_cuda = torch._C._cuda_getCompiledVersion()
        torch.version.cuda = f"{compiled_cuda // 1000}.{compiled_cuda % 1000 // 10}"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    seed = int(hashlib.sha256(args.case.encode()).hexdigest()[:8], 16)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shape_config = next(iter(load_shape_configs(str(path)).values()))
    case_bf16 = any(spec.get("dtype", "").removeprefix("torch.") == "bfloat16" for spec in shape_config["inputs"])
    case_fp16 = any(spec.get("dtype", "").removeprefix("torch.") == "float16" for spec in shape_config["inputs"])
    first_point = json.loads(path.with_name("shapes.json").read_text())["points"][0]
    inputs = make_benchmark_inputs(shape_config)
    for position, (value, spec) in enumerate(zip(inputs, shape_config["inputs"])):
        if isinstance(value, torch.Tensor):
            expected = (tuple(spec["shape"]), spec.get("dtype", "").removeprefix("torch."), tuple(spec["stride"]) if spec.get("stride") is not None else value.stride(), spec.get("storage_offset", 0) or 0)
            actual = (tuple(value.shape), str(value.dtype).removeprefix("torch."), value.stride(), value.storage_offset())
            if actual != expected:
                raise ValueError(f"First-shape input {position} does not match captured geometry: {actual} != {expected}")
    repro = load_module("benchmark_repro_" + args.case, path).Repro().eval()
    snapshots = {}
    for tensor in inputs:
        if isinstance(tensor, torch.Tensor):
            key = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if key not in snapshots:
                storage = tensor.untyped_storage()
                view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage, 0, (storage.nbytes(),), (1,))
                snapshots[key] = (view, view.clone())
    versions = [tensor._version if isinstance(tensor, torch.Tensor) else None for tensor in inputs]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()

    def restore(all_inputs=False):
        for key, (view, snapshot) in snapshots.items():
            if all_inputs or key in mutations:
                view.copy_(snapshot)
        if stochastic:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)

    source_text = path.read_text()
    stochastic = "inductor_random" in source_text
    mutations = set()
    eager = repro(*inputs)
    torch.cuda.synchronize()
    eager_cpu = tree_map(lambda value: value.detach().cpu() if isinstance(value, torch.Tensor) else value, eager)
    for tensor, version in zip(inputs, versions):
        if isinstance(tensor, torch.Tensor) and tensor._version != version:
            mutations.add((str(tensor.device), tensor.untyped_storage().data_ptr()))
    del eager
    restore(all_inputs=True)
    torch.cuda.empty_cache()

    class HighPrecision(TorchDispatchMode):
        def __init__(self, dtype):
            self.dtype = dtype

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            def promote(value):
                if isinstance(value, torch.dtype) and value.is_floating_point:
                    return self.dtype
                return value
            return func(*tree_map(promote, args), **tree_map(promote, kwargs or {}))

    reference_error = None
    reference_kind = "eager_fp64"
    def precision_inputs(dtype):
        bases = {}
        result = []
        for value in inputs:
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                result.append(value)
                continue
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), value.dtype)
            if key not in bases:
                original = torch.empty(0, dtype=value.dtype, device=value.device).set_(storage, 0, (storage.nbytes() // value.element_size(),), (1,))
                bases[key] = original.to(dtype)
            result.append(bases[key].as_strided(value.shape, value.stride(), value.storage_offset()))
        return result
    try:
        hp_inputs = precision_inputs(torch.float64)
        with HighPrecision(torch.float64):
            reference = repro(*hp_inputs)
        torch.cuda.synchronize()
        reference_cpu = tree_map(lambda value: value.detach().cpu() if isinstance(value, torch.Tensor) else value, reference)
        del reference, hp_inputs
    except Exception as error:
        reference_error = type(error).__name__ + ": " + str(error)
        reference_kind = "eager_fp32"
        hp_inputs = precision_inputs(torch.float32)
        restore(all_inputs=True)
        with HighPrecision(torch.float32):
            reference = repro(*hp_inputs)
        torch.cuda.synchronize()
        reference_cpu = tree_map(lambda value: value.detach().cpu() if isinstance(value, torch.Tensor) else value, reference)
        del reference, hp_inputs
    restore(all_inputs=True)
    torch.cuda.empty_cache()

    def accuracy(actual, baseline=False):
        actual_values, actual_spec = tree_flatten(actual)
        reference_values, reference_spec = tree_flatten(reference_cpu)
        checks = []
        if len(actual_values) != len(reference_values):
            return {"passed": False, "error": "output leaf count mismatch"}
        for actual_value, ref in zip(actual_values, reference_values):
            if not isinstance(ref, torch.Tensor):
                checks.append({"passed": actual_value == ref, "kind": "scalar"})
                continue
            value = actual_value.detach().cpu()
            if value.shape != ref.shape:
                checks.append({"passed": False, "kind": "shape"})
                continue
            if not ref.is_floating_point():
                checks.append({"passed": torch.equal(value, ref), "kind": "exact"})
                continue
            flat_value, flat_ref = value.reshape(-1), ref.reshape(-1)
            squared_error = squared_reference = 0.0
            normal_squared_error = normal_squared_reference = 0.0
            normal_count = extreme_count = 0
            finite_count = mismatch_nonfinite = 0
            max_abs = 0.0
            for offset in range(0, flat_ref.numel(), 1_000_000):
                chunk = flat_value[offset:offset + 1_000_000].to(torch.float64)
                target = flat_ref[offset:offset + 1_000_000].to(torch.float64)
                finite = torch.isfinite(target)
                both_finite = finite & torch.isfinite(chunk)
                same_nonfinite = (torch.isnan(chunk) & torch.isnan(target)) | (chunk == target)
                mismatch_nonfinite += int(((~both_finite) & (~same_nonfinite)).sum())
                difference = chunk[both_finite] - target[both_finite]
                squared_error += float(difference.square().sum())
                squared_reference += float(target[both_finite].square().sum())
                finite_count += int(both_finite.sum())
                normal = both_finite & (target.abs() < torch.finfo(value.dtype).max / 16)
                normal_difference = chunk[normal] - target[normal]
                normal_squared_error += float(normal_difference.square().sum())
                normal_squared_reference += float(target[normal].square().sum())
                normal_count += int(normal.sum())
                extreme_count += int((both_finite & ~normal).sum())
                if difference.numel():
                    max_abs = max(max_abs, float(difference.abs().max()))
            rmse = math.sqrt(squared_error / max(1, finite_count))
            reference_rms = math.sqrt(squared_reference / max(1, finite_count))
            nrmse = rmse / max(reference_rms, 1e-12)
            normal_rmse = math.sqrt(normal_squared_error / max(1, normal_count))
            normal_rms = math.sqrt(normal_squared_reference / max(1, normal_count))
            tolerance = 0.02 if case_bf16 or value.dtype == torch.bfloat16 else 0.002 if case_fp16 or value.dtype == torch.float16 else 0.0002
            floor = 1e-5 if case_bf16 or value.dtype == torch.bfloat16 else 1e-6
            max_scaled_error = 0.0
            for offset in range(0, flat_ref.numel(), 1_000_000):
                chunk = flat_value[offset:offset + 1_000_000].to(torch.float64)
                target = flat_ref[offset:offset + 1_000_000].to(torch.float64)
                finite = torch.isfinite(chunk) & torch.isfinite(target)
                scaled = (chunk[finite] - target[finite]).abs() / (floor + tolerance * (target[finite].abs() + normal_rms))
                if scaled.numel():
                    max_scaled_error = max(max_scaled_error, float(scaled.max()))
            passed = mismatch_nonfinite == 0 and (finite_count > 0 or value.numel() == 0) and normal_rmse <= floor + tolerance * normal_rms and max_scaled_error <= 2
            checks.append({"passed": passed, "dtype": str(value.dtype), "nrmse": nrmse, "rmse": rmse, "reference_rms": reference_rms, "normal_rmse": normal_rmse, "normal_reference_rms": normal_rms, "normal_count": normal_count, "extreme_count": extreme_count, "max_scaled_error": max_scaled_error, "max_scaled_error_limit": 2, "element_count": value.numel(), "max_abs": max_abs, "finite_count": finite_count, "nonfinite_mismatch": mismatch_nonfinite, "rtol_rms": tolerance, "atol_rms": floor})
        return {"passed": all(check["passed"] for check in checks), "reference": reference_kind, "outputs": checks, "stochastic": stochastic, "protocol": 2}

    eager_check = accuracy(eager_cpu, baseline=True)
    del eager_cpu
    metadata = {
        "repro": args.case, "shape_hash": first_point["shape_hash"], "first_shape": shape_config,
        "seed": seed, "gpu": args.gpu, "reference": reference_kind, "fp64_reference_error": reference_error,
        "eager_accuracy": eager_check, "mutating_storages": len(mutations), "stochastic": stochastic,
        "torch": torch.__version__, "torch_source": torch.__file__, "helion_source": helion.__file__,
        "original_torch_version_cuda": original_cuda_version,
        "effective_torch_version_cuda": torch.version.cuda,
        "gpu_initial": gpu_state(args.gpu),
    }
    if args.repair_accuracy or args.recheck_accuracy:
        write_json(case_dir / "validation_metadata.json", metadata)
        for backend in BACKENDS:
            output = case_dir / backend
            result_path = output / "result.json"
            if not result_path.exists():
                continue
            row = json.loads(result_path.read_text())
            if row.get("accuracy", {}).get("error") != "output structure mismatch" and not (args.recheck_accuracy and row.get("status") in {"ok", "accuracy_failed"}):
                continue
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "cache/inductor")
            os.environ["TRITON_CACHE_DIR"] = str(output / "cache/triton")
            os.environ["HELION_CACHE_DIR"] = str(output / "cache/helion")
            source = (output / "compiled.py").read_text()
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in row["winning_configs"].get("helion", {}):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call):
                            chosen = row["winning_configs"]["helion"][node.name]["config"]
                            decorator.keywords = [kw for kw in decorator.keywords if kw.arg not in {"config", "configs"}]
                            decorator.keywords.append(ast.keyword(arg="config", value=ast.parse(f"helion.Config(**{chosen!r})", mode="eval").body))
            pinned = output / "standalone_tuned.py"
            pinned.write_text(ast.unparse(ast.fix_missing_locations(tree)).replace("parents[4]", "parents[5]") + "\n")
            with config.patch(row["compile_options"]):
                module = load_module("validate_pinned_" + args.case + backend, pinned)
                for name, tuner in vars(module).items():
                    if isinstance(tuner, CachingAutotuner):
                        saved = row["winning_configs"].get("triton", {}).get(name)
                        if saved:
                            chosen = saved[0]
                            matching = [launcher for launcher in tuner.launchers if launcher.config.kwargs == chosen["kwargs"] and launcher.config.num_warps == chosen["num_warps"] and launcher.config.num_stages == chosen["num_stages"]]
                            if matching:
                                tuner.launchers = matching[:1]
                restore(all_inputs=True)
                order = module.COMPILED_INPUT_ORDER if backend == "native_helion" else row["compiled_input_order"]
                actual = module.call([inputs[i] for i in order])
                torch.cuda.synchronize()
            row["accuracy"] = accuracy(actual)
            row["status"] = "ok" if row["accuracy"]["passed"] else "accuracy_failed"
            if any(check.get("element_count", 0) > 0 and check.get("finite_count") == 0 for check in row["accuracy"].get("outputs", [])):
                row["status"] = "reference_nonfinite"
            row["accuracy_container_mapping"] = "Compare the same ordered tensor leaves; Inductor boxed tuple may wrap an eager tensor"
            write_json(result_path, row)
            print(json.dumps({"repaired_accuracy": args.case, "backend": backend, "status": row["status"]}), flush=True)
        return 0
    write_json(case_dir / "case.json", metadata)
    functions = {}
    rows = {}
    offset = seed % len(BACKENDS)
    order = BACKENDS[offset:] + BACKENDS[:offset]
    common_options = {
        "triton.cudagraphs": False,
        "autotune_cudagraph_benchmarking": False,
        "compile_threads": 4,
        "fx_graph_cache": False,
        "autotune_remote_cache": False,
        "bundled_autotune_remote_cache": False,
        "autotune_fallback_to_aten": False,
        "trace.helion_code": False,
        "graph_partition": False,
        "triton.multi_kernel": 0,
        "trace.enabled": True,
        "trace.log_autotuning_results": True,
    }

    def stage(name, backend):
        write_json(case_dir / "stage.json", {"stage": name, "backend": backend, "time": time.time(), "pid": os.getpid()})
        print(json.dumps({"repro": args.case, "backend": backend, "stage": name}), flush=True)

    def result_path(backend):
        return case_dir / backend / "result.json"

    for backend in order:
        if finished(result_path(backend)) and args.resume:
            continue
        output = case_dir / backend
        output.mkdir(exist_ok=True)
        previous = json.loads(result_path(backend).read_text()) if args.resume and result_path(backend).exists() else None
        cache = output / "cache"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
        os.environ["HELION_CACHE_DIR"] = str(cache / "helion")
        os.environ["HELION_AUTOTUNE_LOG"] = str(output / "helion_autotune.jsonl")
        options = dict(common_options, **{"trace.debug_dir": str(output / "debug")})
        options.update({
            "max_autotune": backend in {"aten", "triton", "nvgemm"},
            "max_autotune_gemm_backends": backend.upper() if backend in {"aten", "triton", "nvgemm"} else "ATEN,TRITON",
            "triton.native_matmul": backend == "native_triton",
        })
        row = {"backend": backend, "status": "pending", "compile_options": options, "cudagraph": False, "compile_order": list(order), "gpu": args.gpu, "shape_hash": first_point["shape_hash"]}
        rows[backend] = row
        selected, triton_configs, helion_configs, sources, source_modules = [], {}, {}, [], []

        def feedback(timings, name, input_nodes, choices, profiled, precompile_times):
            ordered = sorted(((choice, elapsed) for choice, elapsed in timings.items() if math.isfinite(elapsed)), key=lambda item: item[1])
            selected.append({"op": name, "stage": "GEMM candidate benchmark before optional template fusion", "candidate_best": str(ordered[0][0]) if ordered else None, "choices": [{"choice": str(choice), "time_ms": elapsed, "hash_key": choice.hash_key(), "info": choice.info_dict()} for choice, elapsed in ordered]})

        original_compile = GraphLowering._compile_to_module
        def capture_compile(graph):
            module = original_compile(graph)
            if module.__file__:
                sources.append(Path(module.__file__).read_text())
                source_modules.append(module)
            return module

        original_autotune = CachingAutotuner.autotune_to_one_config
        def capture_triton(autotuner, *a, **kw):
            value = original_autotune(autotuner, *a, **kw)
            name = autotuner.fn.__name__
            triton_configs[name] = [{"kwargs": launcher.config.kwargs, "num_warps": launcher.config.num_warps, "num_stages": launcher.config.num_stages} for launcher in autotuner.launchers]
            return value

        original_set_config = BoundKernel.set_config
        def capture_helion(bound, chosen):
            value = original_set_config(bound, chosen)
            name = bound.kernel.fn.__name__
            helion_configs[name] = {"config": dict(bound._config.config), "decorator": bound.format_kernel_decorator(bound._config, bound.settings)}
            try:
                (output / (name + "_generated.py")).write_text(inspect.getsource(bound._run))
            except (OSError, TypeError):
                pass
            return value

        deadline_fired = False
        def timeout_handler(signum, frame):
            nonlocal deadline_fired
            deadline_fired = True
            raise TimeoutError(f"backend compilation exceeded {args.timeout}s")

        signal.signal(signal.SIGALRM, timeout_handler)
        torch._dynamo.reset()
        metrics.reset()
        clear_feedback_savers()
        add_feedback_saver(feedback)
        restore(all_inputs=True)
        phase = "compile"
        started = time.perf_counter()
        stage(phase, backend)
        try:
            signal.alarm(args.timeout)
            with (output / "compile.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), config.patch(options), patch.object(GraphLowering, "_compile_to_module", capture_compile), patch.object(CachingAutotuner, "autotune_to_one_config", capture_triton), patch.object(BoundKernel, "set_config", capture_helion):
                if backend == "native_helion":
                    wrapper_path = path.parent / "helion/standalone.py"
                    if previous and previous.get("status") == "compiled":
                        tree = ast.parse(wrapper_path.read_text())
                        for node in tree.body:
                            if isinstance(node, ast.FunctionDef) and node.name in previous["winning_configs"]["helion"]:
                                for decorator in node.decorator_list:
                                    if isinstance(decorator, ast.Call):
                                        chosen = previous["winning_configs"]["helion"][node.name]["config"]
                                        decorator.keywords = [kw for kw in decorator.keywords if kw.arg not in {"config", "configs"}]
                                        decorator.keywords.append(ast.keyword(arg="config", value=ast.parse(f"helion.Config(**{chosen!r})", mode="eval").body))
                        wrapper_path = output / "recovered_helion.py"
                        wrapper_path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
                    wrapper = load_module("benchmark_wrapper_" + args.case, wrapper_path)
                    source_modules.append(wrapper)
                    if "COMPILE_ERROR" in vars(wrapper):
                        raise RuntimeError(wrapper.COMPILE_ERROR)
                    sources.append(wrapper_path.read_text())
                    fn = lambda wrapper=wrapper: wrapper.call([inputs[i] for i in wrapper.COMPILED_INPUT_ORDER])
                    result = fn()
                else:
                    artifacts = []
                    parameter_names = list(inspect.signature(repro.forward).parameters)
                    def backend_compile(gm, example_inputs):
                        input_order = []
                        for node in gm.graph.nodes:
                            if node.op != "placeholder":
                                continue
                            source = node.meta["grapharg"].source
                            if not isinstance(source, LocalSource) or source.local_name not in parameter_names:
                                raise RuntimeError(f"Unsupported compiled input source: {source}")
                            input_order.append(parameter_names.index(source.local_name))
                        artifact = compile_fx(gm, example_inputs, config_patches=options)
                        artifacts.append((artifact, input_order))
                        return artifact
                    compiled = torch.compile(repro, backend=backend_compile, fullgraph=True, dynamic=False)
                    result = compiled(*inputs)
                    artifact, input_order = artifacts[0]
                    row["compiled_input_order"] = input_order
                    fn = lambda artifact=artifact, order=input_order: artifact(*[inputs[i] for i in order])
                torch.cuda.synchronize()
            signal.alarm(0)
            row["compile_s"] = time.perf_counter() - started
            if previous and previous.get("status") == "compiled":
                row["recovery_compile_s"] = row["compile_s"]
                row["compile_s"] = previous["compile_s"]
                row["compile_cache_state"] = "Original cold compilation time retained; interrupted measurement recovered from saved caches/configs"
            row["compile_time_kind"] = "wrapper_import_compile_and_autotune_first_call" if backend == "native_helion" else "torch_compile_autotune_first_call"
            row["generated_kernel_count"] = metrics.generated_kernel_count
            for source_module in source_modules:
                for name, value in vars(source_module).items():
                    if isinstance(value, CachingAutotuner) and value.launchers:
                        triton_configs[name] = [{"kwargs": launcher.config.kwargs, "num_warps": launcher.config.num_warps, "num_stages": launcher.config.num_stages, "num_ctas": launcher.config.num_ctas, "maxnreg": launcher.config.maxnreg} for launcher in value.launchers]
            phase = "accuracy"
            stage(phase, backend)
            row["accuracy"] = accuracy(result)
            del result
            row["status"] = "compiled" if row["accuracy"]["passed"] else "accuracy_failed"
            if any(check.get("element_count", 0) > 0 and check.get("finite_count") == 0 for check in row["accuracy"].get("outputs", [])):
                row["status"] = "reference_nonfinite"
            if stochastic:
                row["status"] = "stochastic_accuracy_unverified"
            restore(all_inputs=True)
            for _ in range(5):
                restore()
                fn()
            torch.cuda.synchronize()
            functions[backend] = fn
        except Exception as error:
            signal.alarm(0)
            row.update(status="error", phase=phase, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), compile_s=time.perf_counter() - started)
            if deadline_fired and not isinstance(error, TimeoutError):
                row["cleanup_error_type"] = row["error_type"]
                row["cleanup_error"] = row["error"]
                row["error_type"] = "TimeoutError"
                row["error"] = f"backend compilation exceeded {args.timeout}s; cleanup raised {type(error).__name__}: {error}"
            if previous and previous.get("status") == "compiled":
                row["recovery_compile_s"] = row["compile_s"]
                row["compile_s"] = previous["compile_s"]
            (output / "error.txt").write_text(row["traceback"])
            if "illegal memory access" in str(error).lower() or "device-side assert" in str(error).lower():
                write_json(result_path(backend), row)
                return 3
        finally:
            signal.alarm(0)
            clear_feedback_savers()
            if sources:
                (output / "compiled.py").write_text("\n\n".join(sources))
                row["selected_kernel_names"] = list(dict.fromkeys(name for source in sources for name in selected_kernel_names(source)))
                row["vendor_gemm_calls"] = [name for source in sources for name in vendor_gemm_calls(source)]
                row["native_matmul_fallback"] = backend == "native_triton" and bool(row["vendor_gemm_calls"])
                if backend == "native_helion" and helion_configs:
                    tree = ast.parse(sources[0])
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name in helion_configs:
                            for decorator in node.decorator_list:
                                if isinstance(decorator, ast.Call):
                                    chosen = helion_configs[node.name]["config"]
                                    decorator.keywords = [kw for kw in decorator.keywords if kw.arg not in {"config", "configs"}]
                                    decorator.keywords.append(ast.keyword(arg="config", value=ast.parse(f"helion.Config(**{chosen!r})", mode="eval").body))
                    tuned_source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
                    (output / "standalone_tuned.py").write_text(tuned_source.replace("parents[4]", "parents[5]"))
            row["winning_configs"] = {"gemm_selection": selected, "triton": triton_configs, "helion": helion_configs}
            row["inductor_generated_kernel_count"] = row.get("generated_kernel_count")
            if backend == "native_helion" and sources:
                tree = ast.parse(sources[0])
                row["generated_kernel_count"] = len(helion_configs) + sum(isinstance(node, ast.FunctionDef) and node.name.startswith("triton_") for node in tree.body)
            write_json(output / "winning_configs.json", row["winning_configs"])
            write_json(result_path(backend), row)

    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    timings = {backend: [] for backend in functions}
    wall_timings = {backend: [] for backend in functions}
    timed_backends = list(functions)
    for round_index in range(5):
        round_order = timed_backends[round_index % max(1, len(timed_backends)):] + timed_backends[:round_index % max(1, len(timed_backends))]
        for backend in round_order:
            fn = functions[backend]
            stage("timing", backend)
            samples = []
            for _ in range(20):
                restore()
                flush.zero_()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            timings[backend].extend(samples)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            for _ in range(20):
                restore()
                fn()
            torch.cuda.synchronize()
            wall_timings[backend].append((time.perf_counter() - wall_start) * 1e6 / 20)
    for backend, fn in functions.items():
        stage("profiling", backend)
        row = rows[backend]
        restore()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            fn()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(case_dir / backend / "kernel_trace.json"))
        events = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA and not event.name.startswith(("Memcpy", "Memset"))]
        kernels = [{"name": event.name, "duration_us": event.time_range.elapsed_us(), "device_index": event.device_index} for event in events]
        row.update(
            status="ok" if row["status"] == "compiled" else row["status"],
            latency_us=statistics.median(timings[backend]),
            latency_samples_us=timings[backend],
            wall_latency_us=statistics.median(wall_timings[backend]),
            wall_latency_rounds_us=wall_timings[backend],
            kernel_launch_count=len(kernels),
            distinct_kernel_count=len({kernel["name"] for kernel in kernels}),
            profiled_device_us=sum(kernel["duration_us"] for kernel in kernels),
            kernels=kernels,
            gpu_final=gpu_state(args.gpu),
        )
        row["generated_kernel_count"] = len(row.get("selected_kernel_names", []))
        row["generated_kernel_count_kind"] = "Selected generated program definitions; excludes prebuilt vendor GEMMs and autotune candidates"
        write_json(result_path(backend), row)
        print(json.dumps({"repro": args.case, "backend": backend, "status": row["status"], "compile_s": row["compile_s"], "latency_us": row["latency_us"], "kernels": row["kernel_launch_count"]}), flush=True)
    return 0


def supervise(args):
    paths = sorted((ROOT / "mm_repro/canonical").glob("*/repro.py"))
    if args.cases:
        selected = set(args.cases.split(","))
        paths = [path for path in paths if path.parent.name in selected]
    if args.limit:
        paths = paths[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cases": [path.parent.name for path in paths], "backends": BACKENDS, "gpus": args.gpus,
        "cudagraph": False, "autotune_cudagraph": False, "precision": "highest, TF32 disabled, reduced FP16/BF16 accumulation disabled",
        "timing": "100 individual CUDA-event samples, 64 MiB L2 flush outside events; 5 rotated rounds; median microseconds",
        "wall_timing": "5 rounds of 20 warm-cache calls with final synchronization; includes input restoration for mutations",
        "accuracy": "FP64 eager reference with float dtype arguments promoted; FP32 fallback is recorded; RMS tolerance follows weakest input/output floating dtype: BF16=0.02, FP16=0.002, FP32=0.0002. A final FP32 cast does not erase preceding BF16 quantization.",
        "helion_autotune_effort": "full", "compile_threads_per_gpu": 4, "helion_precompile_jobs_per_gpu": 4,
        "timeout_s": args.timeout, "python": PYTHON, "torch_root": TORCH_ROOT,
        "started": time.time(),
        "torch_git": subprocess.check_output(["git", "-C", TORCH_ROOT, "rev-parse", "HEAD"], text=True).strip(),
        "helion_git": subprocess.check_output(["git", "-C", "/home/jaeyeonw/helion", "rev-parse", "HEAD"], text=True).strip(),
    }
    manifest_path = args.output / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not set(manifest["cases"]).issubset(previous["cases"]):
            raise ValueError("Resume cases must belong to the saved benchmark manifest")
        previous.setdefault("resume_events", []).append({"time": time.time(), "cases": manifest["cases"], "gpus": args.gpus})
        manifest = previous
    write_json(manifest_path, manifest)
    gpu_list = args.gpus.split(",")

    def run_gpu(gpu, gpu_paths):
        for path in gpu_paths:
            case_dir = args.output / "results" / path.parent.name
            if args.resume and all(finished(case_dir / backend / "result.json") for backend in BACKENDS):
                continue
            case_dir.mkdir(parents=True, exist_ok=True)
            while True:
                occupants = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
                if not occupants.stdout.strip():
                    break
                time.sleep(3)
            if args.resume and all(finished(case_dir / backend / "result.json") for backend in BACKENDS):
                continue
            for attempt in range(6):
                cmd = [PYTHON, "-B", str(Path(__file__).resolve()), "--worker", "--case", path.parent.name, "--gpu", str(gpu), "--output", str(args.output), "--timeout", str(args.timeout)]
                if args.resume or attempt:
                    cmd.append("--resume")
                with (case_dir / f"worker_{attempt}.log").open("w") as log:
                    process = subprocess.Popen(cmd, env=environment(gpu, args.output), stdout=log, stderr=log, start_new_session=True)
                    while process.poll() is None:
                        time.sleep(3)
                        stage_file = case_dir / "stage.json"
                        if stage_file.exists():
                            state = json.loads(stage_file.read_text())
                            if state.get("pid") == process.pid and time.time() - state["time"] > args.timeout + 30:
                                os.killpg(process.pid, signal.SIGKILL)
                                backend = state["backend"]
                                elapsed = time.time() - state["time"]
                                write_json(case_dir / backend / "result.json", {"backend": backend, "status": "error", "phase": state["stage"], "error_type": "TimeoutError", "error": "supervisor timeout", "gpu": gpu, "compile_s": elapsed if state["stage"] == "compile" else None, "stage_elapsed_s": elapsed})
                                break
                    process.wait()
                if process.returncode == 0:
                    break
                if all(finished(case_dir / backend / "result.json") for backend in BACKENDS):
                    break
                state_path = case_dir / "stage.json"
                if not state_path.exists() or json.loads(state_path.read_text()).get("pid") != process.pid:
                    for backend in BACKENDS:
                        result = case_dir / backend / "result.json"
                        if not finished(result):
                            write_json(result, {"backend": backend, "status": "error", "phase": "inputs_or_reference", "error_type": "WorkerError", "error": (case_dir / f"worker_{attempt}.log").read_text()[-12000:], "gpu": gpu})
                    break
                if attempt == 5:
                    for backend in BACKENDS:
                        result = case_dir / backend / "result.json"
                        if not finished(result):
                            write_json(result, {"backend": backend, "status": "error", "phase": "worker", "error_type": "WorkerError", "error": (case_dir / f"worker_{attempt}.log").read_text()[-12000:], "gpu": gpu})
            print(json.dumps({"completed_case": path.parent.name, "gpu": gpu}), flush=True)

    with ThreadPoolExecutor(max_workers=len(gpu_list)) as pool:
        futures = [pool.submit(run_gpu, gpu, paths[index::len(gpu_list)]) for index, gpu in enumerate(gpu_list)]
        for future in futures:
            future.result()
    print("Benchmark workers finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro/benchmark")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repair-accuracy", action="store_true")
    parser.add_argument("--recheck-accuracy", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        import fcntl
        lock = (args.output / f"gpu_{args.gpu}.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX)
        while True:
            occupants = subprocess.run(["nvidia-smi", f"--id={args.gpu}", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
            if not occupants.stdout.strip():
                break
            time.sleep(3)
        import torch
        with patch.object(
            torch.cuda.CUDAGraph,
            "capture_begin",
            side_effect=RuntimeError("CUDA Graph capture disabled by benchmark protocol"),
        ):
            sys.exit(worker(args))
    supervise(args)


if __name__ == "__main__":
    main()
