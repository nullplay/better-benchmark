"""PyTorch eager baseline. Optimized oracle pending."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from oracle_harness import get_repro_instance, oracle_impl

REPRO_DIR = Path(__file__).resolve().parent
_REPRO = get_repro_instance(REPRO_DIR)


@oracle_impl(description="PyTorch eager baseline; optimization pending")
def oracle_forward(inputs):
    return _REPRO(*inputs)


if __name__ == "__main__":
    from oracle_harness import _runner_main
    raise SystemExit(_runner_main([str(REPRO_DIR), *sys.argv[1:]]))
