"""Formal command-line entry point for FGAT single-label training."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "src" / "train_single_label.py"
    runpy.run_path(str(target), run_name="__main__")
