#!/usr/bin/env python3
"""Command-line entry point for the fx_visualizer package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fx_visualizer.cli import main


if __name__ == "__main__":
    main()
