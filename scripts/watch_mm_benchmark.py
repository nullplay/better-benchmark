"""Refresh local and published benchmark reports while the sweep is running."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "mm_repro/benchmark")
    parser.add_argument("--pids", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    pids = [int(value) for value in args.pids.split(",")]
    last_publish = 0.0
    while True:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/render_mm_backend_report.py"), "--output", str(args.output)], check=True, stdout=subprocess.DEVNULL)
        summary = json.loads((args.output / "summary.json").read_text())
        print(json.dumps({"time": time.time(), "counts": {backend: values["status_counts"] for backend, values in summary["summary"].items()}, "all_five_valid": summary["common_n"]}), flush=True)
        running = any(Path(f"/proc/{pid}").exists() for pid in pids)
        if time.time() - last_publish >= 300 or not running:
            command = ["meta", "artifacts.artifact", "update", "--id=" + args.artifact, "--output=json"]
            command.extend("--file=file://" + str(args.output / name) for name in ("report.html", "results.json", "results.csv", "README.md"))
            with (args.output / "report.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(command, text=True, capture_output=True)
            (args.output / "last_artifact_update.json").write_text(result.stdout or result.stderr)
            print(json.dumps({"artifact_updated": result.returncode == 0, "url": "https://artifacts.internalmeta.com/a/" + args.artifact}), flush=True)
            last_publish = time.time()
        if not running:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
