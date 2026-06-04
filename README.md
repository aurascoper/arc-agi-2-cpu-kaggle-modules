# ARC-AGI-2 CPU Kaggle Modules Baseline

This repository contains the CPU-only symbolic ARC-AGI-2 Kaggle modules payload and a local submission driver.

Important result: this CPU engine is **not** the current leaderboard path. It produced real hidden-test predictions but scored **0.00** on Kaggle hidden. That makes it a useful historical baseline and packaging reference, not evidence that more CPU/GPU compute will transfer.

## What Is Included

- `kaggle_modules/` — the Python module payload used as a Kaggle dataset.
- `cpu_only_submission.py` — local public/evaluation scoring driver for the CPU-only engine.
- `kaggle_cpu_only_notebook_cell.py` — paste-in Kaggle notebook cell that loads the modules dataset and writes `submission.json`.
- `prepare_arc2_kaggle_modules.py` — helper for rebuilding the modules dataset folder.
- `CLAUDE_DEEP_RESEARCH_PROMPT.md` — prompt for researching a stronger L4x4 GPU / test-time-training path.

## Hidden-LB Diagnosis

The CPU notebook got 0.00 because the symbolic engine did not generalize:

- It loaded the official hidden challenges.
- It emitted 259 non-default predictions.
- It had 0 solve errors.
- It solved 0 hidden tasks.

The likely failure mode is public-eval selection: many “general” operators and solvers were developed while looking at the public evaluation corpus. Runtime scrubbing removes direct `solve_<taskid>` calls, but it cannot remove development-time selection baked into the general engine.

Hardware does not fix this. Running the same symbolic search on L4x4 GPUs would mostly make the same wrong search faster.

## Current Direction

The working transfer path is the ARC_2026D-style Qwen test-time-training notebook:

1. Train/adapt on each task’s own demonstrations at inference time.
2. Decode with grid-token constrained search.
3. Select attempts via augmentation consistency and voting.

The CPU modules here should be treated as fallback-only tail coverage or historical reference. The real score-push work belongs in stronger base models, synthetic-data pretraining/SFT, TTT scale, ensembling, and selection.

## Local Use

From a checkout containing ARC-AGI-2 public/evaluation JSON files:

```bash
python3 -m py_compile cpu_only_submission.py kaggle_modules/*.py
ARC_EVAL_DIR=/path/to/arc-agi-2-data/data/evaluation python3 cpu_only_submission.py
```

The script writes:

```text
tmp/cpu_only_submission.json
```

## Kaggle Dataset Use

Upload `kaggle_modules/` as a Kaggle dataset, attach it to a notebook, then use `kaggle_cpu_only_notebook_cell.py` as the single notebook cell if you want to reproduce the CPU-only baseline.

Again: this is expected to score around 0 on hidden. It is preserved so future work can avoid mistaking local/pseudo-private corpus fit for hidden transfer.

