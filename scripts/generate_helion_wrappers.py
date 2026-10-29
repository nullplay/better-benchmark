"""Generate first-shape Helion/Triton boxed wrappers without executing them."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CodegenComplete(Exception):
    pass


class WrapperGenerationError(Exception):
    pass


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def compile_inputs(config):
    import torch

    storages, inputs = {}, []
    for spec in config["inputs"]:
        kind = spec.get("kind", "tensor")
        if kind == "shape":
            inputs.append(spec["dims"])
            continue
        if kind in {"scalar", "symint"}:
            inputs.append(spec["value"])
            continue
        dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
        device = spec.get("device", "cuda")
        shape = spec["shape"]
        stride = spec.get("stride")
        group = spec.get("alias_group")
        if group is not None:
            if group not in storages:
                storages[group] = torch.empty(
                    config["alias_group_nbytes"][group],
                    dtype=torch.uint8,
                    device=device,
                )
            typed = storages[group].view(dtype)
            if not stride:
                stride = torch.empty(shape, dtype=dtype, device=device).stride()
            inputs.append(
                typed.as_strided(shape, stride, spec.get("storage_offset", 0))
            )
        elif stride:
            offset = spec.get("storage_offset", 0)
            size = offset + 1 + sum((s - 1) * st for s, st in zip(shape, stride))
            inputs.append(
                torch.empty(max(size, 0), dtype=dtype, device=device).as_strided(
                    shape, stride, offset
                )
            )
        else:
            inputs.append(torch.empty(shape, dtype=dtype, device=device))
    return inputs


def assemble_wrapper(source, previews, bindings, config, input_order):
    queues = defaultdict(deque)
    for binding in bindings:
        queues[binding["kernel"]].append(binding["preview"])
    native_names = set(queues)
    helper_names, helper_args = {}, {}
    fragments = []
    for i, preview in enumerate(previews):
        fragment = ast.parse(preview)
        helper = next(
            n
            for n in fragment.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("call_helion_")
        )
        helper_names[i] = helper.name
        helper_args[i] = [a.arg for a in helper.args.args]
        fragments.extend(fragment.body)
    tree = ast.parse(source)
    replacements = []

    class ReplaceNative(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in native_names
            ):
                name = node.func.value.id
                if not queues[name]:
                    raise WrapperGenerationError(f"Missing Helion binding for {name}")
                preview = queues[name].popleft()
                replacements.append(
                    {
                        "triton_kernel": name,
                        "helion_call": helper_names[preview],
                        "arguments": helper_args[preview],
                    }
                )
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(helper_names[preview], ast.Load()),
                        args=[ast.Name(a, ast.Load()) for a in helper_args[preview]],
                        keywords=[],
                    ),
                    node,
                )
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "extern_kernels"
                and node.func.attr
                in {
                    "mm",
                    "addmm",
                    "bmm",
                    "baddbmm",
                    "mv",
                    "addmv",
                    "_scaled_mm",
                    "_int_mm",
                }
            ):
                raise WrapperGenerationError(
                    f"MM fell back to extern_kernels.{node.func.attr}"
                )
            return node

    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (
            node.name in native_names or node.name == "get_args"
        ):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id in native_names for n in node.targets
        ):
            continue
        if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
            continue
        body.append(node)
    tree.body = body
    tree = ReplaceNative().visit(tree)
    boxed = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "call"
    )
    live = {a.arg for a in boxed.args.args}
    deleted = {}

    def retain_alias_bases(statements):
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    live.update(
                        n.id
                        for n in ast.walk(target)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    )
            if isinstance(statement, ast.Delete):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        live.discard(target.id)
                        deleted[target.id] = statement
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id.startswith("call_helion_")
            ):
                for arg in statement.value.args:
                    if isinstance(arg, ast.Name) and arg.id not in live:
                        if arg.id not in deleted:
                            raise WrapperGenerationError(
                                f"Unbound Helion argument: {arg.id}"
                            )
                        deletion = deleted[arg.id]
                        deletion.targets = [
                            t
                            for t in deletion.targets
                            if not isinstance(t, ast.Name) or t.id != arg.id
                        ]
                        live.add(arg.id)
            if isinstance(statement, (ast.With, ast.If, ast.For)):
                retain_alias_bases(statement.body)

    retain_alias_bases(boxed.body)

    class RemoveEmptyDeletes(ast.NodeTransformer):
        def visit_Delete(self, node):
            return node if node.targets else None

    tree = RemoveEmptyDeletes().visit(tree)
    if any(queues.values()) or not replacements:
        raise WrapperGenerationError(f"Unmatched native kernels: {dict(queues)}")
    tree.body = fragments + tree.body
    footer = ast.parse(f"""
FIRST_SHAPE_CONFIG = {config!r}
COMPILED_INPUT_ORDER = {input_order!r}

def get_args():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from repro_harness import make_inputs_from_config
    original = make_inputs_from_config(FIRST_SHAPE_CONFIG)
    return [original[i] for i in COMPILED_INPUT_ORDER]

make_inputs = get_args

if __name__ == "__main__":
    benchmark_compiled_module(get_args())
""")
    tree.body.extend(footer.body)
    ast.fix_missing_locations(tree)
    compile(tree, "standalone.py", "exec")
    return ast.unparse(tree) + "\n", replacements


def generate_one(repro_path):
    import contextlib
    from unittest.mock import patch

    import torch
    from torch._dynamo.exc import BackendCompilerFailed
    from torch._dynamo.source import LocalSource
    from torch._inductor import config as inductor_config
    from torch._inductor.codegen.triton import TritonScheduling
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.debug import DebugFormatter
    from torch._inductor.graph import GraphLowering
    from torch._subclasses.fake_tensor import FakeTensorMode

    sys.path.insert(0, str(ROOT))
    from repro_harness import load_shape_configs

    path = Path(repro_path)
    output = path.parent / "helion"
    output.mkdir(exist_ok=True)
    first_point = json.loads(path.with_name("shapes.json").read_text())["points"][0]
    result = {
        "repro": str(path.relative_to(ROOT)),
        "shape_hash": first_point["shape_hash"],
        "wrapper": str((output / "standalone.py").relative_to(ROOT)),
        "executed": False,
        "gpu_kernels_executed": False,
    }
    start = time.perf_counter()
    previews, bindings, input_order = [], [], []
    wrapper_source = None
    phase = "load"
    with (
        tempfile.TemporaryDirectory(prefix="helion_codegen_") as scratch,
        (output / "codegen.log").open("w") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("helion_export_repro", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            shape_config = next(iter(load_shape_configs(str(path)).values()))
            parameters = list(inspect.signature(module.Repro().forward).parameters)
            original_define = TritonScheduling.define_kernel

            def capture_preview(self, source):
                previews.append(source)

            def capture_kernel(self, source, schedule, kernel):
                name = original_define(self, source, schedule, kernel)
                if kernel.is_native_matmul:
                    bindings.append({"kernel": name, "preview": len(previews) - 1})
                return name

            def capture_source(self):
                nonlocal wrapper_source
                code, _ = self.codegen()
                wrapper_source = code.value
                raise CodegenComplete()

            def backend(gm, args):
                for node in gm.graph.nodes:
                    if node.op != "placeholder":
                        continue
                    source = node.meta["grapharg"].source
                    if (
                        not isinstance(source, LocalSource)
                        or source.local_name not in parameters
                    ):
                        raise WrapperGenerationError(
                            f"Unsupported compiled input source: {source!r}"
                        )
                    input_order.append(parameters.index(source.local_name))
                return compile_fx(gm, args)

            options = {
                "triton.native_matmul": True,
                "trace.helion_code": True,
                "trace.enabled": True,
                "trace.debug_dir": scratch,
                "benchmark_fusion": False,
                "benchmark_template_fusion": False,
                "max_autotune": False,
                "max_autotune_gemm": False,
                "triton.autotune_at_compile_time": False,
                "compile_threads": 1,
                "fx_graph_cache": False,
                "graph_partition": False,
                "triton.multi_kernel": 0,
            }
            torch._dynamo.reset()
            torch.set_num_threads(1)
            phase = "torch_compile_codegen"
            with FakeTensorMode(
                allow_non_fake_inputs=True, allow_fallback_kernels=False
            ):
                inputs = compile_inputs(shape_config)
                with (
                    inductor_config.patch(options),
                    patch.object(DebugFormatter, "helion_code", capture_preview),
                    patch.object(TritonScheduling, "define_kernel", capture_kernel),
                    patch.object(GraphLowering, "_compile_to_module", capture_source),
                ):
                    try:
                        torch.compile(
                            module.Repro(),
                            backend=backend,
                            fullgraph=True,
                            dynamic=False,
                        )(*inputs)
                    except BackendCompilerFailed as error:
                        if not isinstance(error.inner_exception, CodegenComplete):
                            raise
            if wrapper_source is None:
                raise WrapperGenerationError(
                    "Compiler did not produce a source wrapper"
                )
            phase = "wrapper_conversion"
            wrapper, replacements = assemble_wrapper(
                wrapper_source, previews, bindings, shape_config, input_order
            )
            header = f'"""First-shape Helion native MM wrapper; generated without execution.\nRepro: {path.parent.name}\nShape hash: {first_point["shape_hash"]}\ncall(args) follows the Inductor boxed calling convention.\n"""\n'
            (output / "standalone.py").write_text(header + wrapper)
            result.update(
                status="ok",
                helion_kernels=len(previews),
                native_calls=len(replacements),
                replacements=replacements,
                compiled_input_order=input_order,
                torch_source=str(Path(torch.__file__).resolve()),
            )
        except Exception as error:  # noqa: BLE001 -- preserve each compiler failure
            root_error = (
                error.inner_exception
                if isinstance(error, BackendCompilerFailed)
                else error
            )
            category = (
                "wrapper_generation"
                if isinstance(root_error, WrapperGenerationError)
                else "helion_unsupported"
                if isinstance(root_error, NotImplementedError)
                and "Helion" in str(root_error)
                else "torch_compile"
            )
            result.update(
                status="failed",
                phase=phase,
                category=category,
                error_type=type(root_error).__name__,
                error=str(root_error),
                traceback=traceback.format_exc(),
                helion_previews=len(previews),
            )
            print(result["traceback"], flush=True)
            failure = f"{result['error_type']}: {result['error']}"
            (output / "standalone.py").write_text(
                '"""Code generation failed. See codegen_status.json; do not benchmark this file."""\n'
                f"COMPILE_ERROR = {failure!r}\n\ndef call(args):\n    raise RuntimeError(COMPILE_ERROR)\n"
                '\nif __name__ == "__main__":\n    raise RuntimeError(COMPILE_ERROR)\n'
            )
            if previews:
                (output / "partial_helion.py").write_text("\n".join(previews))
    result["seconds"] = round(time.perf_counter() - start, 3)
    write_json(output / "codegen_status.json", result)
    return result


def init_worker(torch_root):
    sys.path.insert(0, str(torch_root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "mm_repro" / "canonical")
    parser.add_argument(
        "--torch-root", type=Path, default=Path("/home/jaeyeonw/pytorch-nullplay")
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-error",
        action="append",
        help="Retry failures containing this text; may be supplied multiple times",
    )
    args = parser.parse_args()
    vendor = Path(sys.prefix) / "lib/python3.12/site-packages/nvidia"
    libraries = [
        vendor / "cu13/lib",
        vendor / "cudnn/lib",
        vendor / "nccl/lib",
        vendor / "cusparselt/lib",
        Path("/usr/local/cuda-13/targets/x86_64-linux/lib"),
    ]
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join(str(p) for p in libraries)
        + ":"
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    paths = sorted(args.corpus.glob("*/repro.py"))
    if args.limit:
        paths = paths[: args.limit]
    rows, pending = [], []
    for path in paths:
        status = path.parent / "helion/codegen_status.json"
        if status.exists() and (args.resume or args.retry_error):
            previous = json.loads(status.read_text())
            if (
                args.retry_error
                and previous["status"] == "failed"
                and any(text in previous["error"] for text in args.retry_error)
            ):
                pending.append(str(path.resolve()))
            else:
                rows.append(previous)
        else:
            pending.append(str(path.resolve()))
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="helion_inductor_cache_") as cache:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
        os.environ["TRITON_CACHE_DIR"] = str(Path(cache) / "triton")
        print(
            f"Generating {len(pending)} first-shape wrappers ({len(paths)} corpus entries); "
            "no generated kernels will execute",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,
            initargs=(args.torch_root,),
        ) as pool:
            futures = {pool.submit(generate_one, p): p for p in pending}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                if len(rows) % 10 == 0 or row["status"] == "failed":
                    print(
                        json.dumps(
                            {
                                "completed": len(rows),
                                "total": len(paths),
                                "ok": sum(r["status"] == "ok" for r in rows),
                                "failed": sum(r["status"] == "failed" for r in rows),
                                "last": row["repro"],
                                "seconds": round(time.perf_counter() - start, 1),
                            }
                        ),
                        flush=True,
                    )
    failures = [r for r in rows if r["status"] == "failed"]
    report = {
        "repros": len(paths),
        "ok": len(rows) - len(failures),
        "failed": len(failures),
        "seconds": round(time.perf_counter() - start, 3),
        "executed": False,
        "gpu_kernels_executed": False,
        "torch_source": str(args.torch_root),
        "error_categories": dict(Counter(r["category"] for r in failures)),
        "error_types": dict(Counter(r["error_type"] for r in failures)),
        "error_messages": dict(
            Counter(
                r["error_type"] + ": " + r["error"].splitlines()[0] for r in failures
            )
        ),
        "results": sorted(rows, key=lambda r: r["repro"]),
    }
    write_json(args.corpus.parent / "helion_codegen_report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
