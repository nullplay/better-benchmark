"""Upgrade early benchmark accuracy checks using saved kernels and configs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark_mm_backends import BACKENDS, environment, PYTHON, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro/benchmark")
    parser.add_argument("--follow-pid", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()

    def recheck_gpu(gpu):
        while True:
            pending = []
            for case_dir in sorted((args.output / "results").glob("*")):
                metadata_path = case_dir / "case.json"
                if not metadata_path.exists():
                    continue
                metadata = json.loads(metadata_path.read_text())
                if str(metadata["gpu"]) != gpu or (case_dir / "strong_recheck_status.json").exists():
                    continue
                stage_path = case_dir / "stage.json"
                if stage_path.exists() and Path("/proc/" + str(json.loads(stage_path.read_text())["pid"])).exists():
                    continue
                paths = [case_dir / backend / "result.json" for backend in BACKENDS]
                if not all(path.exists() for path in paths):
                    continue
                rows = [json.loads(path.read_text()) for path in paths]
                if any(row["status"] == "compiled" for row in rows):
                    continue
                if any(row["status"] in {"ok", "accuracy_failed"} and row.get("accuracy", {}).get("protocol", 1) < 2 for row in rows):
                    pending.append(case_dir)
            for case_dir in pending:
                command = [PYTHON, "-B", str(ROOT / "scripts/benchmark_mm_backends.py"), "--worker", "--recheck-accuracy", "--case", case_dir.name, "--gpu", gpu, "--output", str(args.output)]
                with (case_dir / "strong_accuracy_recheck.log").open("w") as log:
                    result = subprocess.run(command, env=environment(gpu, args.output), stdout=log, stderr=log)
                marker = {"exit_code": result.returncode, "time": time.time()}
                if result.returncode:
                    error = (case_dir / "strong_accuracy_recheck.log").read_text()[-12000:]
                    for backend in BACKENDS:
                        path = case_dir / backend / "result.json"
                        row = json.loads(path.read_text())
                        if row["status"] in {"ok", "accuracy_failed"} and row.get("accuracy", {}).get("protocol", 1) < 2:
                            row.update(status="accuracy_unverified", phase="accuracy_recheck", error_type="SavedKernelValidationError", error=error)
                            path.write_text(json.dumps(row, indent=2) + "\n")
                (case_dir / "strong_recheck_status.json").write_text(json.dumps(marker, indent=2) + "\n")
                print(json.dumps({"accuracy_rechecked": case_dir.name, "gpu": gpu, "exit_code": result.returncode}), flush=True)
            if not args.follow_pid or not Path("/proc/" + str(args.follow_pid)).exists():
                if not pending:
                    break
            time.sleep(10)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(recheck_gpu, ["0", "1", "2", "3"]))


if __name__ == "__main__":
    main()
