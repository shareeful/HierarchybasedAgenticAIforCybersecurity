from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from haai.data.registry import data_root, inventory
from haai.data.spec import SPEC_INDEX


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report which required dataset files are present and which are missing"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    arguments = parser.parse_args()

    base = data_root(arguments.data_dir)
    entries = inventory(base)
    present = [entry for entry in entries if entry.present]
    missing = [entry for entry in entries if not entry.present]

    print(f"dataset root: {base}")
    print(f"{len(present)} of {len(entries)} required datasets present\n")
    for entry in entries:
        marker = "OK     " if entry.present else "MISSING"
        print(f"  {marker}  {entry.key:<32} {entry.message}")
    if missing:
        print("\nmissing datasets and where to obtain them:\n")
        for entry in missing:
            key = entry.key.split("/")[-1]
            spec = SPEC_INDEX[key]
            print(f"  {entry.key}")
            print(f"    expects : {', '.join(spec.patterns)}")
            print(f"    contains: {spec.description}")
            print(f"    columns : {', '.join(spec.columns) if spec.columns else 'n/a'}")
            print(f"    source  : {spec.source}\n")
        print("this pipeline reads recorded data only; it does not generate substitutes")
        return 1
    print("\nall required datasets are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
