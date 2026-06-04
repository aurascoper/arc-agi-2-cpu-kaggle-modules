"""Generate verified ARC-AGI-2 program proposals with an LLM.

This is an offline workflow: it calls an LLM locally or through Ollama, verifies
the returned `transform()` functions on training pairs, and writes a JSON cache
that the Kaggle notebook can consume without internet access.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import os
import re
import time
from collections import Counter, defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from arc2_candidate_solver import (
    PROGRAM_CACHE_PATH,
    call_with_timeout,
    grid_object_summary,
    grid_key,
    load_namespace,
    load_tasks,
    normalize_grid,
    pixel_accuracy,
    shape,
)

try:
    from execution_tracer import ExecutionTracer
except Exception:  # pragma: no cover
    ExecutionTracer = None


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("ARC_PROPOSAL_MODEL", "deepseek-v4-pro:cloud")
DEFAULT_REJECT_LOG = WORKSPACE / "arc2_rejected_programs.jsonl"


HELPER_NAMES = [
    "get_objects", "get_shapes", "detect_background_color", "crop_foreground",
    "bbox", "crop_object", "object_to_grid", "rotate_cw", "rotate_ccw",
    "rotate_180", "mirror_h", "mirror_v", "transpose", "remap_colors",
    "fill_holes", "repair_symmetry", "best_pattern_repair", "remove_noise",
    "largest_object", "smallest_object", "move_object", "scale_grid",
    "tile_grid", "extract_color", "remove_color", "keep_most_common_colors",
]


def grid_text(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(c) for c in row) for row in grid)


def code_hash(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:16]


def stringify_summary_value(value: Any) -> Any:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def object_summary(grid: list[list[int]]) -> dict[str, Any]:
    summary = grid_object_summary(normalize_grid(grid))
    return {key: stringify_summary_value(value) for key, value in summary.items()}


def diff_summary(expected: list[list[int]], predicted: list[list[int]], limit: int = 40) -> dict[str, Any]:
    expected = normalize_grid(expected)
    predicted = normalize_grid(predicted)
    er, ec = shape(expected)
    pr, pc = shape(predicted)
    rows, cols = max(er, pr), max(ec, pc)
    diffs = []
    transition_counts: Counter[str] = Counter()
    for r in range(rows):
        for c in range(cols):
            ev = expected[r][c] if r < er and c < ec else None
            pv = predicted[r][c] if r < pr and c < pc else None
            if ev != pv:
                if len(diffs) < limit:
                    diffs.append({"r": r, "c": c, "expected": ev, "got": pv})
                transition_counts[f"{ev}->{pv}"] += 1
    return {
        "expected_shape": [er, ec],
        "predicted_shape": [pr, pc],
        "mismatch_count": sum(transition_counts.values()),
        "sample_diffs": diffs,
        "transition_counts": dict(transition_counts.most_common(20)),
        "expected_summary": object_summary(expected),
        "predicted_summary": object_summary(predicted),
    }


def trace_candidate_call_tree(
    code: str,
    input_grid: list[list[int]],
    expected: list[list[int]],
    ns: dict[str, Any],
    timeout: float = 2.0,
    limit: int = 6000,
) -> str:
    if ExecutionTracer is None:
        return ""
    trace_ns = dict(ns)
    trace_ns.update({
        "deepcopy": deepcopy,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "deque": deque,
        "itertools": itertools,
        "math": math,
    })
    try:
        tracer = ExecutionTracer(max_depth=6, max_grid_repr=160, timeout=timeout)
        actual, root = tracer.trace_transform(code, deepcopy(input_grid), trace_ns)
        if root is None:
            return ""
        actual_grid = normalize_grid(actual)
        text = tracer.format_trace(root, expected_output=normalize_grid(expected), actual_output=actual_grid)
        return text[:limit]
    except Exception as exc:
        return f"trace error: {type(exc).__name__}: {exc}"


def extract_code_blocks(text: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I):
        code = match.group(1).strip()
        if "def transform" in code:
            blocks.append(trim_code(code))
    if not blocks and "def transform" in text:
        blocks.append(trim_code(text[text.find("def transform") :]))
    seen = set()
    out = []
    for code in blocks:
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def trim_code(code: str) -> str:
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.S)
    code = code.replace("\t", "    ")
    start = code.find("def transform")
    if start >= 0:
        code = code[start:]
    lines = code.splitlines()
    kept = []
    for i, line in enumerate(lines):
        if i == 0 or line.startswith((" ", "\t")) or not line.strip() or line.lstrip().startswith("#"):
            kept.append(line)
        else:
            break
    return "\n".join(kept).strip()


def helper_summary(ns: dict[str, Any]) -> str:
    lines = []
    for name in HELPER_NAMES:
        obj = ns.get(name)
        if obj is None:
            continue
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(...)"
        lines.append(f"- `{name}{sig}`")
    return "\n".join(lines)


def build_prompt(task_id: str, task: dict[str, Any], ns: dict[str, Any], attempt: int) -> str:
    examples = []
    for idx, pair in enumerate(task.get("train", []), start=1):
        examples.append(
            f"Example {idx}\nInput:\n{grid_text(pair['input'])}\nOutput:\n{grid_text(pair['output'])}"
        )
    tests = []
    for idx, pair in enumerate(task.get("test", []), start=1):
        tests.append(f"Test {idx} input:\n{grid_text(pair['input'])}")
    helper_text = helper_summary(ns)
    return f"""You are writing an ARC-AGI-2 grid transformation program.

Task id: {task_id}
Attempt: {attempt}

Training pairs:
{chr(10).join(examples)}

{chr(10).join(tests)}

Available helper functions include:
{helper_text}

Write one concise Python function:

def transform(input_grid):
    ...

Rules:
- Return a rectangular list[list[int]] using colors 0-9.
- Use only Python standard library and the helper functions listed above.
- If a helper signature is unfamiliar, write raw Python loops instead.
- Do not hard-code the test output.
- Generalize from the training pairs.
- Prefer object/shape reasoning, symmetry, separators, color roles, and small explicit procedures.
- Internally self-check against every training example before finalizing.

Return ONLY code in a Python fenced block. No explanation."""


def ollama_generate(prompt: str, model: str, url: str, temperature: float, timeout: int) -> str:
    if requests is None:
        raise RuntimeError("requests is required for Ollama generation")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2200,
        },
    }
    response = requests.post(f"{url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def verify_code(code: str, task: dict[str, Any], ns: dict[str, Any]) -> tuple[bool, float, str | None, str, dict[str, Any]]:
    exec_ns = dict(ns)
    exec_ns.update({
        "deepcopy": deepcopy,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "deque": deque,
        "itertools": itertools,
        "math": math,
    })
    try:
        exec(code, exec_ns)
        transform = exec_ns.get("transform")
        if not callable(transform):
            return False, 0.0, "no transform", "", {"error": "no transform", "examples": []}
    except Exception as exc:
        return False, 0.0, f"exec: {exc}", "", {"error": f"exec: {exc}", "examples": []}

    scores = []
    trace = {"examples": []}
    try:
        for idx, pair in enumerate(task.get("train", []), start=1):
            pred = normalize_grid(call_with_timeout(lambda p=pair: transform(deepcopy(p["input"])), 2.0))
            expected = normalize_grid(pair["output"])
            score = pixel_accuracy(expected, pred)
            scores.append(score)
            example_trace = {
                "index": idx,
                "score": score,
                "input_shape": list(shape(normalize_grid(pair["input"]))),
                "input_summary": object_summary(pair["input"]),
                "expected": expected,
                "predicted": pred,
                "diff": diff_summary(expected, pred),
            }
            trace["examples"].append(example_trace)
            if pred != expected:
                example_trace["call_trace"] = trace_candidate_call_tree(code, pair["input"], expected, ns)
                detail = (
                    f"Failed training example {idx}.\n"
                    f"Input:\n{grid_text(pair['input'])}\n"
                    f"Expected:\n{grid_text(expected)}\n"
                    f"Got:\n{grid_text(pred)}\n"
                )
                return False, sum(scores) / len(scores), "train mismatch", detail, trace
    except Exception as exc:
        trace["error"] = f"run: {exc}"
        return False, sum(scores) / len(scores) if scores else 0.0, f"run: {exc}", "", trace
    return True, 1.0, None, "", trace


def format_trace_for_prompt(trace: dict[str, Any], max_examples: int = 3) -> str:
    chunks = []
    for item in trace.get("examples", [])[:max_examples]:
        diff = item.get("diff", {})
        chunks.append(
            "Example {index}: score={score:.3f}\n"
            "Input summary: {input_summary}\n"
            "Expected summary: {expected_summary}\n"
            "Predicted summary: {predicted_summary}\n"
            "Mismatch count: {mismatch_count}\n"
            "Common expected->got transitions: {transitions}\n"
            "Sample diffs: {sample_diffs}\n"
            "Call trace:\n{call_trace}\n"
            "Expected grid:\n{expected}\n"
            "Predicted grid:\n{predicted}".format(
                index=item.get("index"),
                score=item.get("score", 0.0),
                input_summary=json.dumps(item.get("input_summary", {}), sort_keys=True),
                expected_summary=json.dumps(diff.get("expected_summary", {}), sort_keys=True),
                predicted_summary=json.dumps(diff.get("predicted_summary", {}), sort_keys=True),
                mismatch_count=diff.get("mismatch_count"),
                transitions=json.dumps(diff.get("transition_counts", {}), sort_keys=True),
                sample_diffs=json.dumps(diff.get("sample_diffs", []), sort_keys=True),
                call_trace=item.get("call_trace") or "(not available)",
                expected=grid_text(item.get("expected", [[0]])),
                predicted=grid_text(item.get("predicted", [[0]])),
            )
        )
    if trace.get("error"):
        chunks.append(f"Runtime error: {trace['error']}")
    return "\n\n".join(chunks)


def build_repair_prompt(
    task_id: str,
    task: dict[str, Any],
    ns: dict[str, Any],
    code: str,
    failure: str,
    trace: dict[str, Any],
    depth: int,
) -> str:
    base = build_prompt(task_id, task, ns, attempt=999)
    trace_text = format_trace_for_prompt(trace)
    return f"""{base}

The previous code failed verification:
```python
{code}
```

Failure:
{failure}

Execution trace and diffs:
{trace_text}

Patch depth: {depth}

Repair the code by making the smallest useful change that explains the diffs.
Keep the same `def transform(input_grid):` interface.
Do not hard-code individual example grids or task ids.
Return ONLY one corrected Python fenced block."""


def build_mutation_prompt(
    task_id: str,
    task: dict[str, Any],
    ns: dict[str, Any],
    code: str,
    trace: dict[str, Any],
    depth: int,
) -> str:
    base = build_prompt(task_id, task, ns, attempt=998)
    trace_text = format_trace_for_prompt(trace)
    return f"""{base}

The following strategy is close but wrong:
```python
{code}
```

Execution trace:
{trace_text}

Mutation depth: {depth}

Write a NEW variant using a different object/geometry/color-role hypothesis.
Use the trace to avoid the same mistake. Return ONLY one Python fenced block."""


def append_reject(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def load_reject_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def base_transform_code(code: str) -> str:
    return re.sub(r"def\s+transform\s*\(", "def _base_transform(", trim_code(code), count=1)


def compile_transform(code: str, ns: dict[str, Any]) -> tuple[Any | None, str | None]:
    exec_ns = dict(ns)
    exec_ns.update({
        "deepcopy": deepcopy,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "deque": deque,
        "itertools": itertools,
        "math": math,
    })
    try:
        exec(code, exec_ns)
    except Exception as exc:
        return None, f"exec: {exc}"
    transform = exec_ns.get("transform")
    if not callable(transform):
        return None, "no transform"
    return transform, None


def collect_prediction_examples(code: str, task: dict[str, Any], ns: dict[str, Any]) -> tuple[list[dict[str, Any]], float, str | None]:
    transform, err = compile_transform(code, ns)
    if transform is None:
        return [], 0.0, err
    examples = []
    scores = []
    for idx, pair in enumerate(task.get("train", []), start=1):
        try:
            predicted = normalize_grid(call_with_timeout(lambda p=pair: transform(deepcopy(p["input"])), 2.0))
        except Exception as exc:
            return examples, sum(scores) / len(scores) if scores else 0.0, f"run: {exc}"
        expected = normalize_grid(pair["output"])
        inp = normalize_grid(pair["input"])
        score = pixel_accuracy(expected, predicted)
        scores.append(score)
        diffs = []
        er, ec = shape(expected)
        pr, pc = shape(predicted)
        for r in range(max(er, pr)):
            for c in range(max(ec, pc)):
                ev = expected[r][c] if r < er and c < ec else None
                pv = predicted[r][c] if r < pr and c < pc else None
                if ev != pv:
                    diffs.append({"r": r, "c": c, "expected": ev, "got": pv})
        examples.append({
            "index": idx,
            "input": inp,
            "expected": expected,
            "predicted": predicted,
            "score": score,
            "diffs": diffs,
        })
    return examples, sum(scores) / len(scores) if scores else 0.0, None


def window_signature(grid: list[list[int]], r: int, c: int, radius: int) -> tuple[Any, ...]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    values = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            values.append(grid[rr][cc] if 0 <= rr < rows and 0 <= cc < cols else None)
    return tuple(values)


def cross_signature(grid: list[list[int]], r: int, c: int) -> tuple[Any, ...]:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    values = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        rr, cc = r + dr, c + dc
        values.append(grid[rr][cc] if 0 <= rr < rows and 0 <= cc < cols else None)
    return tuple(values)


def build_pattern_rules(
    examples: list[dict[str, Any]],
    *,
    source: str,
    radius: int | None = None,
) -> list[tuple[Any, int]] | None:
    rule_map: dict[Any, int] = {}
    for example in examples:
        inp = example["input"]
        pred = example["predicted"]
        for diff in example["diffs"]:
            r, c = int(diff["r"]), int(diff["c"])
            target = diff["expected"]
            if not isinstance(target, int):
                return None
            if source in {"input_window", "input_cross", "combined_window", "combined_cross"} and (
                r >= len(inp) or c >= len(inp[0])
            ):
                return None
            if source == "pred_window":
                pattern = window_signature(pred, r, c, int(radius or 1))
            elif source == "input_window":
                pattern = window_signature(inp, r, c, int(radius or 1))
            elif source == "combined_window":
                pattern = (
                    window_signature(pred, r, c, int(radius or 1)),
                    window_signature(inp, r, c, int(radius or 1)),
                )
            elif source == "pred_cross":
                pattern = cross_signature(pred, r, c)
            elif source == "input_cross":
                pattern = cross_signature(inp, r, c)
            elif source == "combined_cross":
                pattern = (cross_signature(pred, r, c), cross_signature(inp, r, c))
            else:
                return None
            if pattern in rule_map and rule_map[pattern] != target:
                return None
            rule_map[pattern] = target
    return sorted(rule_map.items(), key=lambda item: repr(item[0]))


PATTERN_PATCH_TEMPLATE = r'''
_PATCH_SOURCE = __PATCH_SOURCE__
_PATCH_RADIUS = __PATCH_RADIUS__
_PATCH_RULES = __PATCH_RULES__

def _norm_grid(grid):
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return [[0]]
    return [list(row) for row in grid]

def _window(grid, r, c, radius):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    values = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            values.append(grid[rr][cc] if 0 <= rr < rows and 0 <= cc < cols else None)
    return tuple(values)

def _cross(grid, r, c):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    values = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
        rr, cc = r + dr, c + dc
        values.append(grid[rr][cc] if 0 <= rr < rows and 0 <= cc < cols else None)
    return tuple(values)

def _patch_key(source, base, inp, r, c, radius):
    if source == "pred_window":
        return _window(base, r, c, radius)
    if source == "input_window":
        return _window(inp, r, c, radius)
    if source == "combined_window":
        return (_window(base, r, c, radius), _window(inp, r, c, radius))
    if source == "pred_cross":
        return _cross(base, r, c)
    if source == "input_cross":
        return _cross(inp, r, c)
    if source == "combined_cross":
        return (_cross(base, r, c), _cross(inp, r, c))
    return None

def transform(input_grid):
    inp = _norm_grid(input_grid)
    base = _norm_grid(_base_transform([row[:] for row in inp]))
    out = [row[:] for row in base]
    rows = len(base)
    cols = len(base[0]) if rows else 0
    rule_map = dict(_PATCH_RULES)
    for r in range(rows):
        for c in range(cols):
            key = _patch_key(_PATCH_SOURCE, base, inp, r, c, _PATCH_RADIUS)
            if key in rule_map:
                out[r][c] = rule_map[key]
    return out
'''


PERIODIC_PATCH_TEMPLATE = r'''
def _norm_grid(grid):
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return [[0]]
    return [list(row) for row in grid]

def _full_rows(grid, color):
    return [r for r, row in enumerate(grid) if all(v == color for v in row)]

def _full_cols(grid, color):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    return [c for c in range(cols) if all(grid[r][c] == color for r in range(rows))]

def _segments(length, seps):
    points = sorted(set([0, length - 1] + [s for s in seps if 0 <= s < length]))
    return [(points[i], points[i + 1]) for i in range(len(points) - 1) if points[i + 1] - points[i] >= 2]

def _repair_rect(grid, r0, r1, c0, c1, max_frac):
    height = r1 - r0
    width = c1 - c0
    if height <= 0 or width <= 0 or height * width < 4:
        return []
    patch = [row[c0:c1] for row in grid[r0:r1]]
    best = None
    for pr in range(1, min(height, 8) + 1):
        for pc in range(1, min(width, 8) + 1):
            if pr * pc >= height * width:
                continue
            phase = {}
            for rr in range(pr):
                for cc in range(pc):
                    counts = {}
                    for r in range(rr, height, pr):
                        for c in range(cc, width, pc):
                            value = patch[r][c]
                            counts[value] = counts.get(value, 0) + 1
                    if not counts:
                        continue
                    max_count = max(counts.values())
                    tied = [value for value, count in counts.items() if count == max_count]
                    anchor = patch[rr][cc]
                    phase[(rr, cc)] = anchor if anchor in tied else min(tied)
            repaired = [[phase[(r % pr, c % pc)] for c in range(width)] for r in range(height)]
            changes = sum(repaired[r][c] != patch[r][c] for r in range(height) for c in range(width))
            if changes == 0:
                continue
            if changes > max(1, int(height * width * max_frac)):
                continue
            score = (changes, pr * pc, pr + pc)
            if best is None or score < best[0]:
                best = (score, repaired)
    if best is None:
        return []
    _, repaired = best
    edits = []
    for r in range(height):
        for c in range(width):
            if repaired[r][c] != patch[r][c]:
                edits.append((r0 + r, c0 + c, repaired[r][c]))
    return edits

def _periodic_repair(grid, max_frac=0.12):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return grid
    bg = grid[0][0]
    row_seps = _full_rows(grid, bg)
    col_seps = _full_cols(grid, bg)
    regions = [(0, rows, 0, cols)]
    if len(row_seps) >= 2 or len(col_seps) >= 2:
        if len(row_seps) < 2:
            row_seps = [0, rows - 1]
        if len(col_seps) < 2:
            col_seps = [0, cols - 1]
        for r0, r2 in _segments(rows, row_seps):
            for c0, c2 in _segments(cols, col_seps):
                regions.append((r0 + 1, r2, c0 + 1, c2))
                if r2 - r0 >= 4 and c2 - c0 >= 4:
                    regions.append((r0 + 2, r2 - 1, c0 + 2, c2 - 1))
    best = []
    for r0, r1, c0, c1 in regions:
        edits = _repair_rect(grid, r0, r1, c0, c1, max_frac)
        if edits and (not best or len(edits) < len(best)):
            best = edits
    out = [row[:] for row in grid]
    for r, c, value in best:
        if 0 <= r < rows and 0 <= c < cols:
            out[r][c] = value
    return out

def transform(input_grid):
    base = _norm_grid(_base_transform([row[:] for row in _norm_grid(input_grid)]))
    return _periodic_repair(base, max_frac=__MAX_FRAC__)
'''


SYMMETRY_PATCH_TEMPLATE = r'''
_SYMMETRY_MODE = __SYMMETRY_MODE__

def _norm_grid(grid):
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
        return [[0]]
    return [list(row) for row in grid]

def _orbit(r, c, rows, cols, mode):
    coords = {(r, c)}
    if mode in ("h", "both"):
        coords.add((r, cols - 1 - c))
    if mode in ("v", "both"):
        coords.add((rows - 1 - r, c))
    if mode in ("rot180", "both"):
        coords.add((rows - 1 - r, cols - 1 - c))
    return sorted(coords)

def _symmetry_repair(grid, mode):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    out = [row[:] for row in grid]
    seen = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in seen:
                continue
            coords = _orbit(r, c, rows, cols, mode)
            seen.update(coords)
            counts = {}
            for rr, cc in coords:
                value = grid[rr][cc]
                counts[value] = counts.get(value, 0) + 1
            max_count = max(counts.values())
            tied = [value for value, count in counts.items() if count == max_count]
            value = grid[r][c] if grid[r][c] in tied else min(tied)
            for rr, cc in coords:
                out[rr][cc] = value
    return out

def transform(input_grid):
    base = _norm_grid(_base_transform([row[:] for row in _norm_grid(input_grid)]))
    return _symmetry_repair(base, _SYMMETRY_MODE)
'''


def deterministic_patch_codes(
    code: str,
    task: dict[str, Any],
    ns: dict[str, Any],
    *,
    max_total_diffs: int,
    max_example_diffs: int,
) -> list[tuple[str, str, float, int]]:
    examples, score, err = collect_prediction_examples(code, task, ns)
    if err or not examples:
        return []
    if any(shape(example["expected"]) != shape(example["predicted"]) for example in examples):
        return []
    if any(diff.get("expected") is None or diff.get("got") is None for example in examples for diff in example["diffs"]):
        return []
    total_diffs = sum(len(example["diffs"]) for example in examples)
    if total_diffs == 0 or total_diffs > max_total_diffs:
        return []
    if any(len(example["diffs"]) > max_example_diffs for example in examples):
        return []

    base = base_transform_code(code)
    specs: list[tuple[str, str, float, int]] = []
    for source in ("pred_window", "input_window", "combined_window"):
        for radius in (1, 2):
            rules = build_pattern_rules(examples, source=source, radius=radius)
            if not rules:
                continue
            patch = (
                PATTERN_PATCH_TEMPLATE
                .replace("__PATCH_SOURCE__", repr(source))
                .replace("__PATCH_RADIUS__", repr(radius))
                .replace("__PATCH_RULES__", repr(rules))
            )
            specs.append((f"det_patch:{source}_r{radius}", f"{base}\n\n{patch}".strip(), score, total_diffs))
    for source in ("pred_cross", "input_cross", "combined_cross"):
        rules = build_pattern_rules(examples, source=source)
        if not rules:
            continue
        patch = (
            PATTERN_PATCH_TEMPLATE
            .replace("__PATCH_SOURCE__", repr(source))
            .replace("__PATCH_RADIUS__", "1")
            .replace("__PATCH_RULES__", repr(rules))
        )
        specs.append((f"det_patch:{source}", f"{base}\n\n{patch}".strip(), score, total_diffs))
    for max_frac in (0.04, 0.08, 0.12, 0.20):
        patch = PERIODIC_PATCH_TEMPLATE.replace("__MAX_FRAC__", repr(max_frac))
        specs.append((f"det_patch:periodic_frac_{max_frac}", f"{base}\n\n{patch}".strip(), score, total_diffs))
    for mode in ("h", "v", "rot180", "both"):
        patch = SYMMETRY_PATCH_TEMPLATE.replace("__SYMMETRY_MODE__", repr(mode))
        specs.append((f"det_patch:symmetry_{mode}", f"{base}\n\n{patch}".strip(), score, total_diffs))

    seen = set()
    unique = []
    for name, patch_code, base_score, diff_count in specs:
        sig = code_hash(patch_code)
        if sig in seen:
            continue
        seen.add(sig)
        unique.append((name, patch_code, base_score, diff_count))
    return unique


def deterministic_patch_leave_one_out_ok(
    base_code: str,
    task: dict[str, Any],
    ns: dict[str, Any],
    patch_name: str,
    *,
    max_total_diffs: int,
    max_example_diffs: int,
) -> bool:
    """Validate a deterministic patch family without fitting on the held example."""
    train = task.get("train", [])
    if len(train) < 2:
        return False
    for held_index, held_pair in enumerate(train):
        reduced_task = dict(task)
        reduced_task["train"] = [deepcopy(pair) for i, pair in enumerate(train) if i != held_index]
        reduced_specs = deterministic_patch_codes(
            base_code,
            reduced_task,
            ns,
            max_total_diffs=max_total_diffs,
            max_example_diffs=max_example_diffs,
        )
        matching = [code for name, code, _score, _diff_count in reduced_specs if name == patch_name]
        if not matching:
            return False
        held_expected = normalize_grid(held_pair["output"])
        held_ok = False
        for patch_code in matching:
            transform, err = compile_transform(patch_code, ns)
            if transform is None or err:
                continue
            try:
                pred = normalize_grid(call_with_timeout(lambda p=held_pair: transform(deepcopy(p["input"])), 2.0))
            except Exception:
                continue
            if pred == held_expected:
                held_ok = True
                break
        if not held_ok:
            return False
    return True


def enqueue_generated(
    pending: deque[dict[str, Any]],
    response: str,
    *,
    kind: str,
    depth: int,
    parent_hash: str,
    limit: int,
    max_queue: int,
) -> int:
    added = 0
    for code in extract_code_blocks(response)[:limit]:
        if len(pending) >= max_queue:
            break
        pending.append({
            "code": code,
            "kind": kind,
            "depth": depth,
            "parent_hash": parent_hash,
        })
        added += 1
    return added


def enqueue_code_specs(
    pending: deque[dict[str, Any]],
    specs: list[tuple[str, str, float, int]],
    *,
    depth: int,
    parent_hash: str,
    limit: int,
    max_queue: int,
    base_code: str | None = None,
) -> int:
    added = 0
    for name, code, base_score, diff_count in specs[:limit]:
        if len(pending) >= max_queue:
            break
        pending.append({
            "code": code,
            "kind": name,
            "depth": depth,
            "parent_hash": parent_hash,
            "base_train_score": base_score,
            "base_diff_count": diff_count,
            "base_code": base_code,
        })
        added += 1
    return added


def test_hits(code: str, task: dict[str, Any], ns: dict[str, Any]) -> int:
    exec_ns = dict(ns)
    exec_ns.update({
        "deepcopy": deepcopy,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "deque": deque,
        "itertools": itertools,
        "math": math,
    })
    try:
        exec(code, exec_ns)
        transform = exec_ns.get("transform")
    except Exception:
        return 0
    hits = 0
    for pair in task.get("test", []):
        if "output" not in pair:
            continue
        try:
            pred = normalize_grid(call_with_timeout(lambda p=pair: transform(deepcopy(p["input"])), 2.0))
        except Exception:
            continue
        if pred == normalize_grid(pair["output"]):
            hits += 1
    return hits


def public_output_count(task: dict[str, Any]) -> int:
    return sum(1 for pair in task.get("test", []) if "output" in pair)


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def run_replay_rejects(
    args: argparse.Namespace,
    tasks: dict[str, Any],
    ns: dict[str, Any],
    cache: dict[str, Any],
) -> dict[str, Any]:
    rows = load_reject_rows(args.replay_rejects)
    allowed = set(args.task_id or tasks.keys())
    rows = [row for row in rows if row.get("task_id") in allowed and row.get("task_id") in tasks and row.get("code")]
    rows.sort(key=lambda row: (-float(row.get("train_score", 0.0)), row.get("task_id", ""), row.get("code_hash", "")))
    if args.max_replay_rows:
        rows = rows[: args.max_replay_rows]

    seen_parent_hashes: set[str] = set()
    seen_patch_hashes: set[str] = set()
    verified = 0
    cached = 0
    withheld_public = 0
    withheld_loo = 0
    attempted_patches = 0
    exact_public_hits = 0
    for row_index, row in enumerate(rows, start=1):
        task_id = row["task_id"]
        task = tasks[task_id]
        parent_code = row["code"]
        parent_hash = row.get("code_hash") or code_hash(parent_code)
        if parent_hash in seen_parent_hashes:
            continue
        seen_parent_hashes.add(parent_hash)
        specs = deterministic_patch_codes(
            parent_code,
            task,
            ns,
            max_total_diffs=args.patch_max_total_diffs,
            max_example_diffs=args.patch_max_example_diffs,
        )
        if not specs:
            continue
        print(f"[replay] {row_index}/{len(rows)} {task_id} parent={parent_hash} patches={len(specs)}", flush=True)
        task_rows = list(cache.get(task_id, []))
        existing_code = {item.get("code", "") for item in task_rows if isinstance(item, dict)}
        for name, patch_code, base_score, diff_count in specs[: args.det_patch_limit]:
            patch_hash = code_hash(patch_code)
            if patch_hash in seen_patch_hashes or patch_code in existing_code:
                continue
            seen_patch_hashes.add(patch_hash)
            attempted_patches += 1
            ok, train_score, err, _failure, _trace = verify_code(patch_code, task, ns)
            if not ok:
                continue
            if args.deterministic_loo and not deterministic_patch_leave_one_out_ok(
                parent_code,
                task,
                ns,
                name,
                max_total_diffs=args.patch_max_total_diffs,
                max_example_diffs=args.patch_max_example_diffs,
            ):
                withheld_loo += 1
                print(f"  verified withheld {name} leave-one-out failed", flush=True)
                continue
            verified += 1
            hits = test_hits(patch_code, task, ns)
            test_count = public_output_count(task)
            if test_count and hits == test_count:
                exact_public_hits += 1
            if args.require_public_exact_for_cache and test_count and hits != test_count:
                withheld_public += 1
                print(f"  verified withheld {name} public_hits={hits}/{test_count}", flush=True)
                continue
            row_name = f"det_replay_{parent_hash}_{name.replace(':', '_')}_{len(task_rows)}"
            task_rows.append({
                "name": row_name,
                "model": "deterministic_replay",
                "attempt": row.get("attempt", 0),
                "kind": name,
                "depth": int(row.get("depth", 0)) + 1,
                "parent_hash": parent_hash,
                "code_hash": patch_hash,
                "base_train_score": base_score,
                "base_diff_count": diff_count,
                "train_score": train_score,
                "public_test_hits": hits,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "code": patch_code,
            })
            existing_code.add(patch_code)
            cached += 1
            print(f"  verified cached {row_name} public_hits={hits}/{test_count}", flush=True)
        if task_rows:
            cache[task_id] = task_rows
            save_cache(args.out, cache)
    save_cache(args.out, cache)
    return {
        "replay_rows": len(rows),
        "parents_considered": len(seen_parent_hashes),
        "patches_attempted": attempted_patches,
        "verified_train_exact": verified,
        "cached": cached,
        "withheld_loo": withheld_loo,
        "withheld_public": withheld_public,
        "public_exact_verified": exact_public_hits,
    }


def reject_cluster_key(row: dict[str, Any]) -> str:
    examples = row.get("trace", {}).get("examples", [])
    failing = None
    for example in examples:
        diff = example.get("diff", {})
        if diff.get("mismatch_count"):
            failing = example
            break
    if failing is None and examples:
        failing = examples[-1]
    diff = (failing or {}).get("diff", {})
    expected_summary = diff.get("expected_summary", {})
    predicted_summary = diff.get("predicted_summary", {})
    transitions = tuple(sorted(diff.get("transition_counts", {}).items())[:5])
    shape_pair = (tuple(diff.get("expected_shape", [])), tuple(diff.get("predicted_shape", [])))
    object_delta = (
        expected_summary.get("object_count", 0) - predicted_summary.get("object_count", 0),
        expected_summary.get("hole_count", 0) - predicted_summary.get("hole_count", 0),
        expected_summary.get("sep_rows", 0) - predicted_summary.get("sep_rows", 0),
        expected_summary.get("sep_cols", 0) - predicted_summary.get("sep_cols", 0),
    )
    mismatch_count = int(diff.get("mismatch_count") or 0)
    mismatch_bucket = 0 if mismatch_count == 0 else 1 if mismatch_count == 1 else 2 if mismatch_count <= 4 else 3 if mismatch_count <= 16 else 4
    return json.dumps({
        "error": row.get("error"),
        "shape_pair": shape_pair,
        "mismatch_bucket": mismatch_bucket,
        "transitions": transitions,
        "object_delta": object_delta,
    }, sort_keys=True)


def cluster_rejects(path: Path, *, limit: int = 20) -> dict[str, Any]:
    rows = load_reject_rows(path)
    clusters: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = reject_cluster_key(row)
        cluster = clusters.setdefault(key, {
            "key": json.loads(key),
            "count": 0,
            "tasks": Counter(),
            "best_train_score": 0.0,
            "examples": [],
        })
        cluster["count"] += 1
        cluster["tasks"][row.get("task_id", "")] += 1
        cluster["best_train_score"] = max(cluster["best_train_score"], float(row.get("train_score", 0.0)))
        if len(cluster["examples"]) < 3:
            cluster["examples"].append({
                "task_id": row.get("task_id"),
                "kind": row.get("kind"),
                "depth": row.get("depth"),
                "code_hash": row.get("code_hash"),
                "train_score": row.get("train_score"),
            })
    ranked = sorted(clusters.values(), key=lambda item: (-item["count"], -item["best_train_score"], json.dumps(item["key"], sort_keys=True)))
    for cluster in ranked:
        cluster["tasks"] = dict(cluster["tasks"].most_common(8))
        cluster["best_train_score"] = round(cluster["best_train_score"], 6)
    return {
        "path": str(path),
        "rows": len(rows),
        "clusters": ranked[:limit],
    }


def solved_from_diagnostics(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return set()
    # Diagnostics do not include gold correctness, so only use the known helper
    # solved list from the current public eval artifacts when available.
    return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=WORKSPACE / "arc_agi_2_data" / "evaluation")
    parser.add_argument("--out", type=Path, default=PROGRAM_CACHE_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--reflections", type=int, default=1)
    parser.add_argument("--repair-depth", type=int, default=2)
    parser.add_argument("--repair-threshold", type=float, default=0.70)
    parser.add_argument("--mutations", type=int, default=1)
    parser.add_argument("--max-queued", type=int, default=12)
    parser.add_argument("--reject-log", type=Path, default=DEFAULT_REJECT_LOG)
    parser.add_argument("--replay-rejects", type=Path)
    parser.add_argument("--cluster-rejects", type=Path)
    parser.add_argument("--cluster-limit", type=int, default=20)
    parser.add_argument("--deterministic-only", action="store_true")
    parser.add_argument("--deterministic-patches", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--det-patch-limit", type=int, default=12)
    parser.add_argument("--patch-max-total-diffs", type=int, default=16)
    parser.add_argument("--patch-max-example-diffs", type=int, default=8)
    parser.add_argument("--deterministic-loo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-replay-rows", type=int, default=0)
    parser.add_argument("--require-public-exact-for-cache", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    ns = load_namespace()
    cache = load_cache(args.out) if (args.resume or args.replay_rejects) else {}

    if args.cluster_rejects:
        print(json.dumps(cluster_rejects(args.cluster_rejects, limit=args.cluster_limit), indent=2))
        return

    task_ids = sorted(args.task_id or tasks.keys())
    if args.max_tasks:
        task_ids = task_ids[: args.max_tasks]

    t0 = time.time()
    total_verified = 0
    total_test_hits = 0
    total_rejected = 0
    total_repairs_requested = 0
    total_mutations_requested = 0
    total_enqueued = 0
    total_det_patches_queued = 0
    total_withheld_public = 0
    total_withheld_loo = 0
    replay_summary = None
    if args.replay_rejects:
        replay_summary = run_replay_rejects(args, tasks, ns, cache)
        if args.deterministic_only:
            print(json.dumps({
                "out": str(args.out),
                "reject_log": str(args.replay_rejects),
                **replay_summary,
                "elapsed_s": round(time.time() - t0, 1),
            }, indent=2))
            return

    for index, task_id in enumerate(task_ids, start=1):
        task = tasks[task_id]
        existing = cache.get(task_id, [])
        if args.resume and len(existing) >= args.attempts:
            continue
        print(f"[proposal] {index}/{len(task_ids)} {task_id} existing={len(existing)}", flush=True)
        rows = list(existing)
        seen_code = {row.get("code", "") for row in rows if isinstance(row, dict)}
        for attempt in range(len(existing) + 1, args.attempts + 1):
            prompt = build_prompt(task_id, task, ns, attempt)
            try:
                response = ollama_generate(prompt, args.model, args.ollama_url, args.temperature, args.timeout)
            except Exception as exc:
                print(f"  generation error: {exc}", flush=True)
                continue
            codes = extract_code_blocks(response)
            print(f"  attempt {attempt}: {len(codes)} code block(s)", flush=True)
            pending_codes: deque[dict[str, Any]] = deque(
                {"code": code, "kind": "initial", "depth": 0, "parent_hash": None}
                for code in codes[: args.max_queued]
            )
            evaluated_for_attempt = 0
            while pending_codes:
                if evaluated_for_attempt >= args.max_queued:
                    print(f"    queue cap reached ({args.max_queued})", flush=True)
                    break
                item = pending_codes.popleft()
                code = item["code"]
                if code in seen_code:
                    continue
                seen_code.add(code)
                evaluated_for_attempt += 1
                kind = item.get("kind", "initial")
                depth = int(item.get("depth", 0))
                parent_hash = item.get("parent_hash")
                code_id = code_hash(code)
                ok, train_score, err, failure, trace = verify_code(code, task, ns)
                if not ok:
                    total_rejected += 1
                    eligible = (
                        depth < args.repair_depth
                        and train_score >= args.repair_threshold
                        and bool(trace.get("examples") or trace.get("error") or failure)
                    )
                    append_reject(args.reject_log, {
                        "task_id": task_id,
                        "attempt": attempt,
                        "kind": kind,
                        "depth": depth,
                        "parent_hash": parent_hash,
                        "code_hash": code_id,
                        "train_score": train_score,
                        "error": err,
                        "failure": failure,
                        "trace": trace,
                        "eligible_for_repair": eligible,
                        "model": args.model,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "code": code,
                    })
                    print(
                        f"    reject {kind} d{depth} train={train_score:.3f} {err} hash={code_id}",
                        flush=True,
                    )
                    if eligible and args.deterministic_patches and not str(kind).startswith("det_patch"):
                        patch_specs = deterministic_patch_codes(
                            code,
                            task,
                            ns,
                            max_total_diffs=args.patch_max_total_diffs,
                            max_example_diffs=args.patch_max_example_diffs,
                        )
                        added = enqueue_code_specs(
                            pending_codes,
                            patch_specs,
                            depth=depth + 1,
                            parent_hash=code_id,
                            limit=args.det_patch_limit,
                            max_queue=args.max_queued,
                            base_code=code,
                        )
                        total_det_patches_queued += added
                        total_enqueued += added
                        if added:
                            print(f"      deterministic patches queued={added}", flush=True)
                    if eligible and args.reflections > 0:
                        try:
                            repair_response = ollama_generate(
                                build_repair_prompt(task_id, task, ns, code, failure, trace, depth + 1),
                                args.model,
                                args.ollama_url,
                                max(0.05, args.temperature - 0.15),
                                args.timeout,
                            )
                            total_repairs_requested += 1
                            added = enqueue_generated(
                                pending_codes,
                                repair_response,
                                kind="repair",
                                depth=depth + 1,
                                parent_hash=code_id,
                                limit=args.reflections,
                                max_queue=args.max_queued,
                            )
                            total_enqueued += added
                            print(f"      repair queued={added}", flush=True)
                        except Exception as exc:
                            print(f"    repair generation error: {exc}", flush=True)
                    if eligible and args.mutations > 0:
                        try:
                            mutation_response = ollama_generate(
                                build_mutation_prompt(task_id, task, ns, code, trace, depth + 1),
                                args.model,
                                args.ollama_url,
                                min(0.95, args.temperature + 0.20),
                                args.timeout,
                            )
                            total_mutations_requested += 1
                            added = enqueue_generated(
                                pending_codes,
                                mutation_response,
                                kind="mutation",
                                depth=depth + 1,
                                parent_hash=code_id,
                                limit=args.mutations,
                                max_queue=args.max_queued,
                            )
                            total_enqueued += added
                            print(f"      mutation queued={added}", flush=True)
                        except Exception as exc:
                            print(f"    mutation generation error: {exc}", flush=True)
                    continue
                hits = test_hits(code, task, ns)
                test_count = public_output_count(task)
                if args.deterministic_loo and str(kind).startswith("det_patch"):
                    base_code = item.get("base_code")
                    if not base_code or not deterministic_patch_leave_one_out_ok(
                        base_code,
                        task,
                        ns,
                        kind,
                        max_total_diffs=args.patch_max_total_diffs,
                        max_example_diffs=args.patch_max_example_diffs,
                    ):
                        total_withheld_loo += 1
                        print(
                            f"    verified withheld {kind} d{depth} leave-one-out failed hash={code_id}",
                            flush=True,
                        )
                        continue
                if args.require_public_exact_for_cache and test_count and hits != test_count:
                    total_withheld_public += 1
                    print(
                        f"    verified withheld {kind} d{depth} public_hits={hits}/{test_count} hash={code_id}",
                        flush=True,
                    )
                    continue
                total_verified += 1
                total_test_hits += hits
                name = f"{args.model.replace(':', '_')}_a{attempt}_{kind}d{depth}_{len(rows)}"
                rows.append({
                    "name": name,
                    "model": args.model,
                    "attempt": attempt,
                    "kind": kind,
                    "depth": depth,
                    "parent_hash": parent_hash,
                    "code_hash": code_id,
                    "train_score": train_score,
                    "public_test_hits": hits,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "code": code,
                })
                print(f"    verified {name} public_hits={hits}", flush=True)
        if rows:
            cache[task_id] = rows
            save_cache(args.out, cache)
    save_cache(args.out, cache)
    print(json.dumps({
        "out": str(args.out),
        "tasks": len(task_ids),
        "cached_tasks": len(cache),
        "programs": sum(len(v) for v in cache.values() if isinstance(v, list)),
        "new_verified": total_verified,
        "new_public_test_hits": total_test_hits,
        "withheld_public": total_withheld_public,
        "withheld_loo": total_withheld_loo + (replay_summary or {}).get("withheld_loo", 0),
        "rejected": total_rejected,
        "repairs_requested": total_repairs_requested,
        "mutations_requested": total_mutations_requested,
        "deterministic_patches_queued": total_det_patches_queued,
        "generated_children_queued": total_enqueued,
        "reject_log": str(args.reject_log),
        "replay": replay_summary,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
