"""Prepare the local Kaggle modules dataset payload for ARC-AGI-2.

The notebook looks for these files under `/kaggle/input/modules` or a dataset
whose final path contains the same module files. This script assembles a local
folder that can be uploaded as the Kaggle dataset backing that mount.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent

CORE_FILES = [
    "dsl.py",
    "submission_helper.py",
    "arc2_candidate_solver.py",
    "sia_operators.py",
    "arc2_ranker_model.json",
    "arc2_program_cache.json",
    "execution_tracer.py",
    "generate_arc2_program_proposals.py",
    "mine_arc2_nearmiss_clusters.py",
    "perception_search.py",
    "mdl_compose.py",
    "mcts_compose.py",
]


def prepare_modules(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    for filename in CORE_FILES:
        src = WORKSPACE / filename
        if not src.exists():
            if filename.endswith(".json"):
                continue
            missing.append(filename)
            continue
        shutil.copy2(src, out_dir / filename)
        copied.append(filename)

    for src in sorted(WORKSPACE.glob("eval_holdout_primitives_batch*.py")):
        shutil.copy2(src, out_dir / src.name)
        copied.append(src.name)

    metadata = {
        "title": "arc2-workspace-modules",
        "id": "aurascoper/arc2-workspace-modules",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    copied.append("dataset-metadata.json")

    return {
        "out_dir": str(out_dir),
        "copied": copied,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORKSPACE / "kaggle_modules")
    args = parser.parse_args()
    print(json.dumps(prepare_modules(args.out), indent=2))


if __name__ == "__main__":
    main()
