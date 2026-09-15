from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAGIC = re.compile(r"^\s*[%!]")


class Display:
    def __init__(self, verbose: bool):
        self.verbose = verbose

    def __call__(self, *objects) -> None:
        if not self.verbose:
            return
        for item in objects:
            print(item)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute the notebook's code cells in order, outside Jupyter"
    )
    parser.add_argument(
        "--notebook", type=str,
        default="notebooks/Hierarchical_Agentic_AI_Risk_Assessment.ipynb",
    )
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--full", dest="quick", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()

    notebook = json.loads(Path(arguments.notebook).read_text())
    cells = [
        "".join(cell["source"]) if isinstance(cell["source"], str) else "\n".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]

    namespace = {
        "__name__": "__notebook__",
        "display": Display(arguments.verbose),
        "get_ipython": lambda: None,
    }
    failures = []
    for index, source in enumerate(cells):
        lines = []
        for line in source.split("\n"):
            if MAGIC.match(line):
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}pass")
            else:
                lines.append(line)
        stripped = "\n".join(lines)
        if arguments.quick:
            stripped = stripped.replace("QUICK = False", "QUICK = True")
        if not stripped.strip():
            continue
        try:
            exec(compile(stripped, f"<cell {index}>", "exec"), namespace)
            plt.close("all")
            print(f"cell {index:3d}  OK")
        except Exception as exc:
            failures.append((index, stripped[:400], traceback.format_exc()))
            print(f"cell {index:3d}  FAIL  {type(exc).__name__}: {exc}")

    print("=" * 78)
    if failures:
        print(f"{len(failures)} of {len(cells)} cells failed")
        for index, source, trace in failures:
            print("-" * 78)
            print(f"cell {index}\n{source}\n{trace}")
        return 1
    print(f"all {len(cells)} cells executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
