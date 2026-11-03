"""Replay the photographic compression experiment from lesson 30 on a CPU."""
import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compression_math import benchmark_photo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT / "data/codec-study/settings.json",
                        help="Settings JSON saved by the lesson 30 experiment")
    parser.add_argument("--output", type=Path, help="Write the report here (default: standard output)")
    args = parser.parse_args()
    try:
        settings = json.loads(args.settings.read_text(encoding="utf-8"))
        report = benchmark_photo(settings)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
    report["runtime"] = {
        "renderer": "NumPy CPU",
        "os": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    result = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(result)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
