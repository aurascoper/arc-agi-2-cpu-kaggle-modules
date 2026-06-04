# Claude.ai Deep Research Prompt: ARC-AGI-2 L4x4 Private-LB Push

Use this prompt in Claude.ai with Deep Research enabled.

```text
You are helping me improve an ARC-AGI-2 Kaggle submission. I want an evidence-grounded plan for private-leaderboard progress under Kaggle L4x4 GPU constraints. Start by falsifying or confirming whether "49+" is a real Kaggle-legal ARC-AGI-2 private target; do not assume it is.

Context and constraints:
- Current known working transfer baseline: ARC_2026D-style Qwen test-time training (TTT), reported score 30.14.
- Crucial uncertainty: 30.14 is almost certainly a 2026 public/semi-private live-LB score, not a final private score, because the 2025 ARC-AGI-2 private record was about 24% (NVARC). Treat score identity as Stage 0 to disambiguate before optimizing to that number.
- Known hard fact to verify from primary sources: the ARC Prize 2025 technical report says the top ARC-AGI-2 private score reached 24%, and the official ARC Prize 2025 page lists NVARC at 24.0% on ARC-AGI-2 Private Evaluation.
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
- Current caution: multi-seed TTT ensembling is probably low ROI under 12h; do not recommend it ahead of data/SFT, selection/voting, decode speedups, or LoRA right-sizing unless you have concrete evidence.

Your task:
1. Stage 0: determine what the reported 30.14 likely is: public/semi-private live LB vs private/final score. Explain how Kaggle/ARC Prize reporting splits public, semi-private, and private scores in 2025-2026.
2. Deep-research the current public ARC-AGI-2 / ARC Prize 2025-2026 state of the art, especially Kaggle-legal notebooks/writeups. Separate private-LB evidence from public/semi-private or API/unconstrained results.
3. Identify whether "49+" corresponds to a real ARC-AGI-2 Kaggle-private entry/notebook/recipe. If yes, reverse-engineer the mechanism: base model, synthetic data, TTT recipe, decoding, ensembling, runtime budget, and packaging. If no, say so plainly and name the category error (ARC-AGI-1, semi-private, API/unconstrained, etc.).
4. If no credible 49+ private recipe exists, give the most realistic path to private-LB gains: likely high-20s/low-30s frontier first, not "49 by knob tuning."
5. Prioritize interventions by expected private-LB gain per L4x4 runtime risk:
   - NVARC/open synthetic-data SFT and base checkpoint
   - product-of-experts / AIRV-style selection and eval augmentation
   - speculative decoding, prefix/KV caching, and only then lower DFS cutoff
   - right-sized per-task TTT rank/steps
   - epochs / LR / LoRA rank and dropout
   - constrained decoding/beam/cutoff
   - final attempt selection/voting
   - multi-seed TTT ensembling only if runtime headroom remains
6. Produce a concrete Kaggle implementation plan:
   - exact model/dataset artifacts to attach or create
   - exact notebook code edits
   - env knobs/defaults
   - expected runtime on L4x4
   - failure modes and how to detect them in logs
   - A/B ladder with stop/go gates
7. Be adversarial about claims. Distinguish:
   - hidden-LB evidence
   - public-eval/local evidence
   - speculation
   - contamination/leakage risk
8. Final output format:
   - Executive verdict: is 49+ private realistic with Kaggle L4x4 only?
   - Score identity verdict: what is 30.14 most likely measuring, and what evidence would prove it?
   - If 49+ private is real: the shortest credible Kaggle-legal recipe.
   - If no: top 3 private-LB score-push experiments, ordered by expected hidden/private gain.
   - A table of knobs with expected gain, runtime cost, implementation complexity, and leakage risk.
   - A concrete next notebook/data patch plan.
   - A stop/go ladder: what to run first, what to bank as null, and when to stop local knob-spinning.

Links/artifacts I can provide:
- Public CPU baseline repo: https://github.com/aurascoper/arc-agi-2-cpu-kaggle-modules
- Current TTT notebooks:
  - ARC_2026D baseline/dynamic-safe notebook
  - AB1 max_score=0.15 notebook
  - beyond-AB1 scale-up notebook

Do not ask me to upload my private full workspace. Ask only for the specific notebook cells or public artifacts you need.
```
