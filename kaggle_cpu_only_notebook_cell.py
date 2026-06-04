# CPU-only ARC-AGI-2 Kaggle submission cell.
#
# Attach the `kaggle_modules` dataset containing submission_helper.py and
# arc2_candidate_solver.py. This cell writes /kaggle/working/submission.json.
#
# Historical note: this CPU symbolic engine emitted real hidden predictions but
# scored 0.00 hidden, so it is a baseline/fallback reference, not the leading path.

import os, sys, json, glob, time, re, hashlib
from pathlib import Path


# 1) Put the modules dataset directly on sys.path.
DATASET = None
for root, _dirs, files in os.walk("/kaggle/input"):
    if "submission_helper.py" in files and "arc2_candidate_solver.py" in files:
        DATASET = root
        break

assert DATASET, "modules dataset not attached (no submission_helper.py under /kaggle/input)"

if DATASET not in sys.path:
    sys.path.insert(0, DATASET)
print("modules:", DATASET)


# 2) CPU engine config. No LLM/vLLM/LoRA/GPU path.
os.environ.update({
    "ARC_SUBMISSION_MODE": "hidden",
    "ARC_HIDDEN_FACING": "1",
    "ARC_DISABLE_DIRECT_SOLVERS": "1",
    "SUBMISSION_ENABLE_CANDIDATE_SOLVER": "1",
    "ARC_ENABLE_RANKER_MODEL": "1",
    "ARC_ENABLE_DSL_SOLVE_CANDIDATES": "0",
    "ARC_DISABLE_DSL_SOLVE_CANDIDATES": "1",
    "ARC_ENABLE_PROGRAM_CACHE": "0",
    "ARC_ENABLE_NARROW_TTA": "1",
})

for k, v in {
    "ARC_CANDIDATE_POOL_LIMIT": "48",
    "ARC_NARROW_TTA_MAX_SOURCES": "4",
    "ARC_NARROW_TTA_MAX_CANDIDATES": "40",
}.items():
    os.environ.setdefault(k, v)


from submission_helper import solve_task


_TASK_SOLVER_RE = re.compile(r"^solve_[0-9a-f]{8}$")


def build_namespace(modules_dir, scrub_task_solvers=True):
    ns = {}
    dsl = os.path.join(modules_dir, "dsl.py")
    if os.path.exists(dsl):
        exec(open(dsl).read(), ns)

    for bf in sorted(glob.glob(os.path.join(modules_dir, "eval_holdout_primitives_batch*.py"))):
        bns = {}
        try:
            exec(open(bf).read(), bns)
        except Exception as exc:
            print("warn:", os.path.basename(bf), exc)
            continue
        for name, value in bns.items():
            if name.startswith("solve_") and callable(value):
                ns[name] = value

    if scrub_task_solvers:
        for name in [x for x in ns if _TASK_SOLVER_RE.match(x)]:
            del ns[name]
    return ns


def pseudo_task_id(tid, i):
    digest = hashlib.sha256(("arc2-pseudo-private:" + tid).encode()).hexdigest()[:10]
    return f"hidden_{i:03d}_{digest}"


def normgrid(g):
    if hasattr(g, "tolist"):
        g = g.tolist()
    if not isinstance(g, list) or not g:
        return [[0]]
    out = []
    for row in g:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list) or not row:
            return [[0]]
        out.append([int(x) for x in row])
    return out


ns = build_namespace(DATASET, scrub_task_solvers=True)
print("solve_* fns:", sum(1 for n, v in ns.items() if n.startswith("solve_") and callable(v)))


# 3) Load competition test challenges.
candidates = [p for p in Path("/kaggle/input").rglob("*test_challenges.json")]
assert candidates, "no *test_challenges.json under /kaggle/input"
test_path = sorted(candidates, key=lambda p: ("training" in p.name, "evaluation" in p.name, len(str(p))))[0]
test_tasks = json.loads(test_path.read_text())
print(f"test tasks: {len(test_tasks)} from {test_path}")


# 4) Generate pass@2 submission. Hidden test has no gold outputs.
submission, errs, t0 = {}, 0, time.time()
for idx, (tid, task) in enumerate(sorted(test_tasks.items())):
    pid = pseudo_task_id(tid, idx)
    entries = []
    for pair in task.get("test", []):
        test_input = pair["input"]
        try:
            a1, a2 = solve_task(pid, {"train": task.get("train", []), "test": [{"input": test_input}]}, test_input, ns)
        except Exception:
            a1, a2 = [], []
            errs += 1
        entries.append({"attempt_1": normgrid(a1), "attempt_2": normgrid(a2)})
    submission[tid] = entries

Path("/kaggle/working/submission.json").write_text(json.dumps(submission, separators=(",", ":")))
nondefault = sum(
    1
    for task_entries in submission.values()
    for entry in task_entries
    for key in ("attempt_1", "attempt_2")
    if entry[key] != [[0]]
)
print(
    f"WROTE /kaggle/working/submission.json: {len(submission)} tasks, "
    f"nondefault slots={nondefault}, errs={errs}, {time.time() - t0:.1f}s"
)

