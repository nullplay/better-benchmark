"""Render the live or completed first-shape five-backend benchmark report."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from benchmark_mm_backends import selected_kernel_names, vendor_gemm_calls, write_json

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["aten", "triton", "nvgemm", "native_triton", "native_helion"]


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def stats(values):
    return {
        "n": len(values), "min": min(values) if values else None,
        "p25": percentile(values, 0.25), "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75), "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "sum": sum(values),
    }


def reassess_accuracy(result, shape):
    if result.get("accuracy", {}).get("error") == "output structure mismatch":
        result["status"] = "validation_pending"
        return
    if "accuracy" not in result or "outputs" not in result["accuracy"]:
        return
    dtypes = {spec.get("dtype", "").removeprefix("torch.") for spec in shape["inputs"] if isinstance(spec, dict)}
    for check in result["accuracy"]["outputs"]:
        if "rmse" not in check:
            continue
        bf16 = "bfloat16" in dtypes or check["dtype"] == "torch.bfloat16"
        fp16 = "float16" in dtypes or check["dtype"] == "torch.float16"
        tolerance, floor = (0.02, 1e-5) if bf16 else (0.002, 1e-6) if fp16 else (0.0002, 1e-6)
        check.update(rtol_rms=tolerance, atol_rms=floor)
        if "max_scaled_error" in check:
            check["max_scaled_error_limit"] = 2
            check["passed"] = check["nonfinite_mismatch"] == 0 and (check["finite_count"] > 0 or check.get("element_count") == 0) and check["normal_rmse"] <= floor + tolerance * check["normal_reference_rms"] and check["max_scaled_error"] <= 2
        else:
            check["passed"] = check["nonfinite_mismatch"] == 0 and check["finite_count"] > 0 and check["rmse"] <= floor + tolerance * check["reference_rms"]
    result["accuracy"]["passed"] = all(check["passed"] for check in result["accuracy"]["outputs"])
    if any(check.get("element_count", 0) > 0 and check.get("finite_count") == 0 for check in result["accuracy"]["outputs"]):
        result["status"] = "reference_nonfinite"
        return
    if result["accuracy"].get("protocol", 1) < 2 and result["status"] in {"ok", "accuracy_failed"}:
        result["status"] = "validation_pending"
        return
    if result["status"] == "accuracy_failed" and result["accuracy"]["passed"]:
        result["status"] = "ok" if "latency_us" in result else "compiled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro/benchmark")
    args = parser.parse_args()
    output = args.output.resolve()
    import fcntl
    lock = (output / "report.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest = json.loads((output / "manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        case_dir = output / "results" / case
        metadata_path = case_dir / "case.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        validation_path = case_dir / "validation_metadata.json"
        if validation_path.exists():
            validation = json.loads(validation_path.read_text())
            metadata["eager_accuracy"] = validation["eager_accuracy"]
            metadata["reference"] = validation["reference"]
        if "eager_accuracy" in metadata:
            baseline = {"status": "pending", "accuracy": metadata["eager_accuracy"]}
            reassess_accuracy(baseline, metadata["first_shape"])
            metadata["eager_accuracy"] = baseline["accuracy"]
        repro_path = ROOT / "mm_repro/canonical" / case / "repro.py"
        point = json.loads(repro_path.with_name("shapes.json").read_text())["points"][0]
        shape = metadata.get("first_shape", {"inputs": point.get("inputs", [])})
        row = {"id": case, "metadata": metadata, "shape": shape, "shape_hash": point["shape_hash"], "mm_count": len(re.findall(r"torch\.ops\.aten\.(?:mm|addmm|bmm|_scaled_mm)\.", repro_path.read_text())), "backends": {}}
        for backend in BACKENDS:
            result = case_dir / backend / "result.json"
            row["backends"][backend] = json.loads(result.read_text()) if result.exists() else {"backend": backend, "status": "pending"}
            reassess_accuracy(row["backends"][backend], shape)
            value = row["backends"][backend]
            if value.get("error_type") == "AttributeError" and "TimeoutError: backend compilation exceeded" in value.get("traceback", ""):
                value["cleanup_error_type"] = value["error_type"]
                value["cleanup_error"] = value["error"]
                value["error_type"] = "TimeoutError"
                value["error"] = "backend compilation exceeded 900s; cleanup raised AttributeError: " + value["cleanup_error"]
            compiled_source = case_dir / backend / "compiled.py"
            if "latency_us" in value and compiled_source.exists() and "selected_kernel_names" not in value:
                value["selected_kernel_names"] = selected_kernel_names(compiled_source.read_text())
                value["generated_kernel_count"] = len(value["selected_kernel_names"])
                value["generated_kernel_count_kind"] = "Selected generated program definitions; excludes prebuilt vendor GEMMs and autotune candidates"
            if compiled_source.exists() and "vendor_gemm_calls" not in value:
                value["vendor_gemm_calls"] = vendor_gemm_calls(compiled_source.read_text())
                value["native_matmul_fallback"] = backend == "native_triton" and bool(value["vendor_gemm_calls"])
        rows.append(row)
    summary = {}
    for backend in BACKENDS:
        values = [row["backends"][backend] for row in rows]
        valid = [value for value in values if value["status"] == "ok"]
        failures = [value for value in values if value["status"] not in {"ok", "pending", "compiled", "validation_pending"}]
        reasons = Counter(
            "accuracy_failed" if value["status"] == "accuracy_failed" else
            "stochastic_accuracy_unverified" if value["status"] == "stochastic_accuracy_unverified" else
            "reference contains no finite values in an output" if value["status"] == "reference_nonfinite" else
            value.get("error_type", "error") + ": " + (value.get("error", "").splitlines() or [""])[0][:220]
            for value in failures
        )
        paired = [row for row in rows if row["backends"][backend]["status"] == "ok" and row["backends"]["aten"]["status"] == "ok"]
        ratios = [row["backends"]["aten"]["latency_us"] / row["backends"][backend]["latency_us"] for row in paired]
        summary[backend] = {
            "status_counts": dict(Counter(value["status"] for value in values)),
            "runtime_us": stats([value["latency_us"] for value in valid]),
            "compile_s": stats([value["compile_s"] for value in valid if "compile_s" in value]),
            "all_attempt_compile_s": stats([value["compile_s"] for value in values if "compile_s" in value]),
            "kernel_launch_count": stats([value["kernel_launch_count"] for value in valid]),
            "generated_kernel_count": stats([value["generated_kernel_count"] for value in valid if value.get("generated_kernel_count") is not None]),
            "paired_n": len(paired),
            "native_matmul_fallback_cases": sum(value.get("native_matmul_fallback", False) for value in valid),
            "paired_speedup_geomean": math.exp(statistics.mean(math.log(ratio) for ratio in ratios)) if ratios else None,
            "errors": dict(reasons),
        }
    common = [row for row in rows if all(row["backends"][backend]["status"] == "ok" for backend in BACKENDS)]
    data = {"manifest": manifest, "updated": time.time(), "rows": rows, "backends": BACKENDS, "summary": summary, "common_n": len(common)}
    write_json(output / "results.json", data)
    write_json(output / "summary.json", {key: value for key, value in data.items() if key != "rows"})
    with (output / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["repro", "shape_hash", "gpu", "backend", "status", "compile_s", "latency_us", "wall_latency_us", "kernel_launch_count", "distinct_kernel_count", "generated_kernel_count", "profiled_device_us", "error"])
        writer.writeheader()
        for row in rows:
            for backend, result in row["backends"].items():
                entry = {key: result.get(key) for key in writer.fieldnames}
                entry.update(repro=row["id"], shape_hash=row["shape_hash"], backend=backend)
                writer.writerow(entry)
    template = Path(__file__).with_name("mm_backend_report.html").read_text()
    visual_data = copy.deepcopy(data)
    for row in visual_data["rows"]:
        for value in row["backends"].values():
            value.pop("latency_samples_us", None)
            value.pop("wall_latency_rounds_us", None)
            value.pop("traceback", None)
            if "error" in value:
                value["error"] = value["error"][:4000]
            for selection in value.get("winning_configs", {}).get("gemm_selection", []):
                selection["candidate_count"] = len(selection["choices"])
                selection["choices"] = selection["choices"][:3]
    embedded = json.dumps(visual_data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (output / "report.html").write_text(template.replace("__BENCHMARK_DATA__", embedded))
    print(json.dumps({"cases": len(rows), "common_all_five": len(common), "counts": {backend: value["status_counts"] for backend, value in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
