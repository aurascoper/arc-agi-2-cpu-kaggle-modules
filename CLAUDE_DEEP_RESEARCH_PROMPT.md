# Claude.ai Deep Research Prompt: ARC-AGI-2 L4x4 Push Past 49

Use this prompt in Claude.ai with Deep Research enabled.

```text
You are helping me improve an ARC-AGI-2 Kaggle submission. I want an evidence-grounded plan to push hidden leaderboard score past 49 using Kaggle L4x4 GPUs.

Context and constraints:
- Current known working transfer baseline: ARC_2026D-style Qwen test-time training (TTT), hidden LB 30.14.
- Hardware target: Kaggle L4x4, 12-hour notebook limit, no internet during rerun except attached datasets/models.
- CPU symbolic engine is a dead end for leading score. It emitted 259 non-default hidden predictions with 0 errors but scored 0.00 hidden. Diagnose it as public-eval overfit / non-transfer, not a compute problem.
- Do not propose hidden-label leakage, public-eval task memorization, task-ID lookup, or competition-rule violations.
- The useful path is GPU/LLM/TTT: stronger base model, synthetic ARC pretraining/SFT, per-task LoRA/TTT scale, augmentation, constrained decoding, ensembling, and selection/voting.

Known ARC_2026D/Qwen-TTT mechanism:
- Base: Qwen3-4B grid SFT style model (`qwen3_4b_grids15_sft139` in the current notebook).
- For each puzzle: reset LoRA weights, augment that puzzle's own train demonstrations, fine-tune LoRA, then decode test grids with constrained grid-token search.
- Default-ish recipe observed locally: train augmentation n=16, LoRA r=256, lr=5e-5, 1 epoch, eval augmentation n=2, beam cap 5, DFS cutoff around `max_score=-log(0.2)`, dual selection by probability and augmentation-consistency voting.
- Our first small A/B is AB1: widen decode cutoff to `-log(0.15)`.
- Beyond-AB1 draft exposes knobs: dynamic base model path, cutoff 0.15, beam 8, train aug 24, eval aug 3, optional short-task epochs, LoRA rank/alpha/dropout hooks, deterministic seeds, optional multi-seed TTT ensembling.

Your task:
1. Deep-research the current public ARC-AGI-2 / ARC Prize 2025-2026 state of the art, especially any public notebook, writeup, discussion, or leaderboard clue that plausibly exceeds 40 or approaches 49.
2. Identify whether "49+" corresponds to a real public entry/notebook/recipe. If yes, reverse-engineer the mechanism: base model, synthetic data, TTT recipe, decoding, ensembling, runtime budget, and packaging.
3. If no credible 49+ public recipe exists, say so plainly and give the most realistic path to incremental gains over 30.14.
4. Prioritize interventions by expected hidden-LB gain per L4x4 runtime risk:
   - base model swap or synthetic-data SFT
   - TTT augmentation counts
   - epochs / LR / LoRA rank and dropout
   - multi-seed TTT ensembling
   - eval augmentation and scoring views
   - constrained decoding/beam/cutoff
   - final attempt selection/voting
5. Produce a concrete Kaggle implementation plan:
   - exact model/dataset artifacts to attach or create
   - exact notebook code edits
   - env knobs/defaults
   - expected runtime on L4x4
   - failure modes and how to detect them in logs
   - A/B ladder with stop/go gates
6. Be adversarial about claims. Distinguish:
   - hidden-LB evidence
   - public-eval/local evidence
   - speculation
   - contamination/leakage risk
7. Final output format:
   - Executive verdict: is 49+ realistic with L4x4 only?
   - If yes: the shortest credible recipe.
   - If no: top 3 score-push experiments likely to beat 30.14.
   - A table of knobs with expected gain, runtime cost, implementation complexity, and leakage risk.
   - A concrete next notebook patch plan.

Links/artifacts I can provide:
- Public CPU baseline repo: https://github.com/aurascoper/arc-agi-2-cpu-kaggle-modules
- Current TTT notebooks:
  - ARC_2026D baseline/dynamic-safe notebook
  - AB1 max_score=0.15 notebook
  - beyond-AB1 scale-up notebook

Do not ask me to upload my private full workspace. Ask only for the specific notebook cells or public artifacts you need.
```
