# Claude.ai Deep Research Prompt: ARC-AGI-2 L4x4 Private-LB Push

Use this prompt in Claude.ai with Deep Research enabled.

```text
You are helping me improve an ARC-AGI-2 Kaggle submission. I want an evidence-grounded plan for leaderboard progress under Kaggle L4x4 GPU constraints. A current live Kaggle leaderboard row reports first place at 49.17 (`nvbanana`, 2026-06-04), so the key question is no longer whether a 49+ live score exists. The key question is what mechanism produced it, whether it is public/reproducible/Kaggle-legal, and whether it is likely to transfer to final private scoring.

Context and constraints:
- User-attested live-LB baseline: `/Users/aurascoper/Downloads/arc-2026d(1).ipynb`, an ARC_2026D-style Qwen test-time-training (TTT) notebook, scored 30.12/30.14 live LB. Treat this as a real reported result, but do not treat it as an owned reproducible baseline until it is bound to a Kaggle row/screenshot/export or re-submitted cleanly.
- Reload-score landmine: the notebook PRINTS a local reload/eval score (~37.67, `*** Reload score: 37.667`). That is a public-eval reload, NOT a Kaggle LB row — the same inflation class as the CPU 0.72-local -> 0.00-hidden collapse. Do not treat 37.67 as the baseline or as evidence of transfer; only a submitted Kaggle leaderboard row counts.
- Crucial score-identity distinction: 30.14/30.12 and 49.17 are live Kaggle leaderboard scores unless proven otherwise. The 2025 ARC-AGI-2 final private record was about 24% (NVARC), while the 2026 live leaderboard can be much higher. Do not conflate live public/semi-private LB with final private score.
- Known hard facts to verify from primary sources:
  - the ARC Prize 2025 technical report says the top ARC-AGI-2 private score reached 24%;
  - the official ARC Prize 2025 page lists NVARC at 24.0% on ARC-AGI-2 Private Evaluation;
  - current Kaggle live leaderboard shows `nvbanana` at 49.17 as of 2026-06-04.
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
1. Gate -1 / Stage 0: explain how to bind the user-attested 30.12/30.14 ARC_2026D notebook score to a reproducible owned baseline, then explain how Kaggle/ARC Prize reporting splits live public/semi-private and final private scores in 2025-2026. State what can and cannot be inferred from a live 49.17 score.
2. Deep-research the current public ARC-AGI-2 / ARC Prize 2025-2026 state of the art, especially Kaggle-legal notebooks/writeups and any discussion, code, dataset, or kernel tied to `nvbanana` or the 49.17 live score.
3. Identify whether the 49.17 live score has a public/reproducible recipe. If yes, reverse-engineer the mechanism: base model, synthetic data, TTT recipe, decoding, ensembling, runtime budget, and packaging. If no, say what evidence is missing and what can still be inferred from neighboring public solutions.
4. Distinguish live-LB gain from final-private gain. If 49.17 is only live/semi-private evidence, estimate private-transfer risk and the likely failure modes.
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
   - Executive verdict: what is the likely mechanism class behind the live 49.17?
   - Score identity verdict: what do 30.12/30.14 and 49.17 measure, and what evidence would prove final-private transfer?
   - If the 49.17 recipe is public/reproducible: the shortest credible Kaggle-legal implementation path.
   - If no public recipe exists: top 3 reverse-engineering hypotheses and top 3 score-push experiments, ordered by expected live-LB and private-transfer gain.
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
