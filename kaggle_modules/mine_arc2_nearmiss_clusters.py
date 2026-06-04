"""Mine ARC-AGI-2 near-miss candidate clusters.

This is an offline diagnostics tool. It evaluates the current candidate factory
on public evaluation tasks, finds high train-accuracy candidates that still miss
held-out outputs, and groups them into reusable operator targets.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from arc2_candidate_solver import (
    Grid,
    call_with_timeout,
    collect_operator_specs,
    generate_candidates,
    grid_key,
    load_namespace,
    load_tasks,
    normalize_grid,
    shape,
)

try:
    from generate_arc2_program_proposals import cluster_rejects
except Exception:  # pragma: no cover
    cluster_rejects = None


WORKSPACE = Path(__file__).resolve().parent


def color_counts(grid: Grid) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in normalize_grid(grid):
        counts.update(row)
    return counts


def background_color(grid: Grid) -> int:
    counts = color_counts(grid)
    return counts.most_common(1)[0][0] if counts else 0


def same_shape(a: Grid, b: Grid) -> bool:
    return shape(a) == shape(b)


def solid_frame_score(grid: Grid) -> int:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if rows == 0 or cols == 0:
        return 0
    score = 0
    if len(set(grid[0])) == 1:
        score += 1
    if rows > 1 and len(set(grid[-1])) == 1:
        score += 1
    if len({grid[r][0] for r in range(rows)}) == 1:
        score += 1
    if cols > 1 and len({grid[r][cols - 1] for r in range(rows)}) == 1:
        score += 1
    return score


def solid_line_count(grid: Grid) -> int:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if rows == 0 or cols == 0:
        return 0
    solid_rows = sum(1 for row in grid if len(set(row)) == 1)
    solid_cols = sum(1 for c in range(cols) if len({grid[r][c] for r in range(rows)}) == 1)
    return solid_rows + solid_cols


def grid_diff_stats(expected: Grid, predicted: Grid, *, bg: int | None = None) -> dict[str, Any]:
    expected = normalize_grid(expected)
    predicted = normalize_grid(predicted)
    er, ec = shape(expected)
    pr, pc = shape(predicted)
    rows, cols = max(er, pr), max(ec, pc)
    if bg is None:
        bg = background_color(expected)
    transitions: Counter[str] = Counter()
    coords: list[tuple[int, int]] = []
    extra_fg = 0
    missing_fg = 0
    fg_swap = 0
    for r in range(rows):
        for c in range(cols):
            ev = expected[r][c] if r < er and c < ec else None
            pv = predicted[r][c] if r < pr and c < pc else None
            if ev == pv:
                continue
            coords.append((r, c))
            transitions[f"{ev}->{pv}"] += 1
            if ev == bg and pv not in (bg, None):
                extra_fg += 1
            elif ev not in (bg, None) and pv == bg:
                missing_fg += 1
            elif ev not in (bg, None) and pv not in (bg, None):
                fg_swap += 1
    row_counts = Counter(r for r, _c in coords)
    col_counts = Counter(c for _r, c in coords)
    aligned = bool(coords) and (
        max(row_counts.values(), default=0) / len(coords) >= 0.50
        or max(col_counts.values(), default=0) / len(coords) >= 0.50
    )
    return {
        "shape_pair": [list((er, ec)), list((pr, pc))],
        "mismatch_count": len(coords),
        "transition_counts": dict(transitions.most_common(8)),
        "extra_fg": extra_fg,
        "missing_fg": missing_fg,
        "fg_swap": fg_swap,
        "aligned": aligned,
        "sample_coords": coords[:12],
    }


def train_task_stats(task: dict[str, Any]) -> dict[str, Any]:
    pairs = task.get("train", [])
    same = bool(pairs) and all(same_shape(pair["input"], pair["output"]) for pair in pairs)
    cells = 0
    diffs = 0
    transitions: Counter[str] = Counter()
    frameish = False
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if same_shape(inp, out):
            rows, cols = shape(inp)
            cells += rows * cols
            bg = background_color(inp)
            stats = grid_diff_stats(out, inp, bg=bg)
            diffs += stats["mismatch_count"]
            transitions.update(stats["transition_counts"])
        frameish = frameish or solid_frame_score(inp) >= 3 or solid_line_count(inp) >= 4
    diff_frac = diffs / cells if cells else None
    return {
        "same_shape": same,
        "diff_frac": diff_frac,
        "sparse": diff_frac is not None and 0 < diff_frac <= 0.08,
        "frameish": frameish,
        "train_transitions": dict(transitions.most_common(6)),
    }


def diff_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 16:
        return "5-16"
    if count <= 64:
        return "17-64"
    return "65+"


def classify_nearmiss(
    *,
    task_stats: dict[str, Any],
    family: str,
    name: str,
    public_stats: list[dict[str, Any]],
) -> str:
    total = sum(item["mismatch_count"] for item in public_stats)
    extra_fg = sum(item["extra_fg"] for item in public_stats)
    missing_fg = sum(item["missing_fg"] for item in public_stats)
    fg_swap = sum(item["fg_swap"] for item in public_stats)
    aligned = any(item["aligned"] for item in public_stats)
    if family == "periodic_repair" or "periodic" in name:
        return "periodic_or_panel_repair"
    if task_stats["frameish"] and task_stats["sparse"]:
        return "panel_grid_sparse_repair"
    if total and extra_fg / total >= 0.65:
        return "denoise_remove_extra_foreground"
    if total and missing_fg / total >= 0.65:
        return "foreground_completion_or_copy"
    if total and fg_swap / total >= 0.55:
        return "color_role_swap"
    if aligned and task_stats["same_shape"]:
        return "line_path_or_alignment_repair"
    if task_stats["same_shape"]:
        return "object_movement_contact_alignment"
    return "resize_extract_or_recompose"


def cluster_key(row: dict[str, Any]) -> str:
    transitions = tuple(sorted(row["public_transitions"].items())[:4])
    return json.dumps({
        "category": row["category"],
        "same_shape": row["task_stats"]["same_shape"],
        "train_sparse": row["task_stats"]["sparse"],
        "public_diff_bucket": row["public_diff_bucket"],
        "transitions": transitions,
    }, sort_keys=True)


def mine_candidates(
    tasks: dict[str, Any],
    ns: dict[str, Any],
    *,
    min_train_accuracy: float,
    per_task_candidates: int,
    timeout: float,
    skip_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    solved: list[dict[str, Any]] = []
    skip_task_ids = skip_task_ids or set()
    for task_index, (task_id, task) in enumerate(sorted(tasks.items()), start=1):
        if task_id in skip_task_ids:
            print(f"[mine] {task_index}/{len(tasks)} {task_id} skipped=current_submission_solved", flush=True)
            continue
        task_stats = train_task_stats(task)
        candidates = generate_candidates(task, ns=ns, task_id=task_id)
        seen_predictions: dict[tuple[str, ...], dict[str, Any]] = {}
        for candidate in candidates[:per_task_candidates]:
            if candidate.train_accuracy < min_train_accuracy:
                continue
            pred_keys: list[str] = []
            public_stats: list[dict[str, Any]] = []
            public_hits = 0
            public_total = 0
            public_transitions: Counter[str] = Counter()
            failed = False
            for pair in task.get("test", []):
                if "output" not in pair:
                    continue
                public_total += 1
                try:
                    pred = normalize_grid(call_with_timeout(
                        lambda pair=pair, candidate=candidate: candidate.transform(deepcopy(pair["input"])),
                        timeout,
                    ))
                except Exception:
                    failed = True
                    break
                pred_keys.append(grid_key(pred))
                expected = normalize_grid(pair["output"])
                if pred == expected:
                    public_hits += 1
                stats = grid_diff_stats(expected, pred, bg=background_color(pair["input"]))
                public_stats.append(stats)
                public_transitions.update(stats["transition_counts"])
            if failed or public_total == 0:
                continue
            pred_sig = tuple(pred_keys)
            previous = seen_predictions.get(pred_sig)
            if previous and previous["train_accuracy"] >= candidate.train_accuracy:
                continue
            public_mismatches = sum(item["mismatch_count"] for item in public_stats)
            row = {
                "task_id": task_id,
                "candidate": candidate.name,
                "family": candidate.family,
                "exact_train": candidate.exact_train,
                "train_accuracy": candidate.train_accuracy,
                "public_hits": public_hits,
                "public_total": public_total,
                "public_mismatches": public_mismatches,
                "public_diff_counts": [item["mismatch_count"] for item in public_stats],
                "public_diff_bucket": diff_bucket(public_mismatches),
                "public_transitions": dict(public_transitions.most_common(8)),
                "task_stats": task_stats,
                "category": classify_nearmiss(
                    task_stats=task_stats,
                    family=candidate.family,
                    name=candidate.name,
                    public_stats=public_stats,
                ),
            }
            seen_predictions[pred_sig] = row
        task_rows = list(seen_predictions.values())
        for row in task_rows:
            if row["public_hits"] == row["public_total"]:
                solved.append(row)
            else:
                rows.append(row)
        print(
            f"[mine] {task_index}/{len(tasks)} {task_id} candidates={len(candidates)} "
            f"near_misses={sum(1 for row in task_rows if row['public_hits'] < row['public_total'])}",
            flush=True,
        )
    rows.sort(key=lambda row: (
        -int(row["exact_train"]),
        -float(row["train_accuracy"]),
        int(row["public_mismatches"]),
        row["task_id"],
        row["candidate"],
    ))
    solved.sort(key=lambda row: (row["task_id"], row["candidate"]))
    return rows, solved


def mine_speculative_operators(
    tasks: dict[str, Any],
    ns: dict[str, Any],
    *,
    skip_task_ids: set[str] | None = None,
    timeout: float = 0.5,
) -> list[dict[str, Any]]:
    skip_task_ids = skip_task_ids or set()
    rows: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        if task_id in skip_task_ids:
            continue
        for spec in collect_operator_specs(task, ns=ns, task_id=task_id):
            ev = spec.evidence
            public_hits = 0
            public_total = 0
            public_diff_counts: list[int] = []
            if spec.transform is not None:
                for pair in task.get("test", []):
                    if "output" not in pair:
                        continue
                    public_total += 1
                    try:
                        pred = normalize_grid(call_with_timeout(
                            lambda pair=pair, spec=spec: spec.transform(deepcopy(pair["input"])),
                            timeout,
                        ))
                    except Exception:
                        public_diff_counts.append(-1)
                        continue
                    stats = grid_diff_stats(normalize_grid(pair["output"]), pred, bg=background_color(pair["input"]))
                    public_diff_counts.append(stats["mismatch_count"])
                    public_hits += stats["mismatch_count"] == 0
            rows.append({
                "task_id": task_id,
                "candidate": spec.name,
                "family": spec.family,
                "tier": spec.tier,
                "admission_type": ev.admission_type,
                "evidence_label": ev.evidence_label,
                "loo_informative": ev.loo_informative,
                "train_exact": ev.train_exact,
                "train_accuracy": ev.train_accuracy,
                "leave_one_out": ev.leave_one_out,
                "jackknife_stable": ev.jackknife_stable,
                "structural_confidence": ev.structural_confidence,
                "synthetic_self_consistency": ev.synthetic_self_consistency,
                "synthetic_checks": dict(ev.synthetic_checks),
                "promotion_blockers": list(ev.promotion_blockers),
                "public_hits": public_hits,
                "public_total": public_total,
                "public_diff_counts": public_diff_counts,
            })
    rows.sort(key=lambda row: (
        row["tier"] != "quarantined",
        -float(row["train_accuracy"]),
        min([d for d in row["public_diff_counts"] if d >= 0] or [999999]),
        row["task_id"],
        row["candidate"],
    ))
    return rows


def current_submission_solved_tasks(tasks: dict[str, Any], ns: dict[str, Any]) -> set[str]:
    try:
        from submission_helper import solve_task
    except Exception:
        return set()
    solved: set[str] = set()
    for task_id, task in sorted(tasks.items()):
        all_exact = bool(task.get("test"))
        for pair in task.get("test", []):
            if "output" not in pair:
                all_exact = False
                break
            try:
                attempt1, attempt2 = solve_task(task_id, {"train": task.get("train", []), "test": [pair]}, pair["input"], ns)
            except Exception:
                all_exact = False
                break
            expected = normalize_grid(pair["output"])
            if normalize_grid(attempt1) != expected and normalize_grid(attempt2) != expected:
                all_exact = False
                break
        if all_exact:
            solved.add(task_id)
    return solved


def summarize_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = cluster_key(row)
        cluster = clusters.setdefault(key, {
            "key": json.loads(key),
            "count": 0,
            "tasks": Counter(),
            "families": Counter(),
            "categories": Counter(),
            "exact_train": 0,
            "best_train_accuracy": 0.0,
            "min_public_mismatches": None,
            "examples": [],
        })
        cluster["count"] += 1
        cluster["tasks"][row["task_id"]] += 1
        cluster["families"][row["family"]] += 1
        cluster["categories"][row["category"]] += 1
        cluster["exact_train"] += int(row["exact_train"])
        cluster["best_train_accuracy"] = max(cluster["best_train_accuracy"], float(row["train_accuracy"]))
        if cluster["min_public_mismatches"] is None:
            cluster["min_public_mismatches"] = row["public_mismatches"]
        else:
            cluster["min_public_mismatches"] = min(cluster["min_public_mismatches"], row["public_mismatches"])
        if len(cluster["examples"]) < 8:
            cluster["examples"].append({
                "task_id": row["task_id"],
                "candidate": row["candidate"],
                "family": row["family"],
                "exact_train": row["exact_train"],
                "train_accuracy": round(row["train_accuracy"], 4),
                "public_diff_counts": row["public_diff_counts"],
                "public_transitions": row["public_transitions"],
            })
    ranked = sorted(
        clusters.values(),
        key=lambda item: (
            -item["exact_train"],
            -item["best_train_accuracy"],
            item["min_public_mismatches"] if item["min_public_mismatches"] is not None else 999999,
            -item["count"],
            json.dumps(item["key"], sort_keys=True),
        ),
    )
    for cluster in ranked:
        cluster["tasks"] = dict(cluster["tasks"].most_common(10))
        cluster["families"] = dict(cluster["families"].most_common(8))
        cluster["categories"] = dict(cluster["categories"].most_common(8))
        cluster["best_train_accuracy"] = round(cluster["best_train_accuracy"], 6)
    return ranked


def abstract_rule_prompt(row: dict[str, Any], task: dict[str, Any]) -> str:
    train_chunks = []
    for idx, pair in enumerate(task.get("train", []), start=1):
        train_chunks.append(
            f"Train {idx} input:\n{grid_to_text(pair['input'])}\n"
            f"Train {idx} output:\n{grid_to_text(pair['output'])}"
        )
    return (
        "This local wrapper passed train but failed held-out; infer the abstract rule.\n"
        "Do not write another task-id patch. Describe a reusable ARC operator family "
        "with preconditions, transformation steps, and leave-one-train-out checks.\n\n"
        f"Task: {row['task_id']}\n"
        f"Candidate: {row['candidate']} ({row['family']})\n"
        f"Train accuracy: {row['train_accuracy']:.4f}; exact_train={row['exact_train']}\n"
        f"Held-out public hits: {row['public_hits']}/{row['public_total']}\n"
        f"Held-out diff counts: {row['public_diff_counts']}\n"
        f"Held-out expected->predicted transitions: {json.dumps(row['public_transitions'], sort_keys=True)}\n"
        f"Cluster category: {row['category']}\n\n"
        + "\n\n".join(train_chunks)
    )


def grid_to_text(grid: Grid) -> str:
    return "\n".join("".join(str(v) for v in row) for row in normalize_grid(grid))


def write_markdown_report(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    solved: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    speculative_rows: list[dict[str, Any]],
    reject_summary: dict[str, Any] | None,
    elapsed_s: float,
    prompt_path: Path,
    skipped_solved: set[str],
) -> None:
    non_periodic = [
        cluster for cluster in clusters
        if cluster["key"].get("category") != "periodic_or_panel_repair"
    ]
    top_non_periodic = non_periodic[0] if non_periodic else None
    lines = [
        "# ARC-AGI-2 Near-Miss Cluster Report",
        "",
        f"- Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- Runtime: {elapsed_s:.1f}s",
        f"- High-train public misses: {len(rows)}",
        f"- Public-exact high-train candidate predictions: {len(solved)}",
        f"- Speculative operator diagnostics: {len(speculative_rows)}",
        f"- Current-submission solved tasks skipped: {len(skipped_solved)}",
        f"- Abstract-rule prompt file: `{prompt_path}`",
        "",
        "## Top Candidate Clusters",
        "",
        "| rank | category | count | exact train | best train | min public diffs | tasks | families |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, cluster in enumerate(clusters[:20], start=1):
        key = cluster["key"]
        tasks = ", ".join(f"{task}:{count}" for task, count in list(cluster["tasks"].items())[:5])
        families = ", ".join(f"{family}:{count}" for family, count in list(cluster["families"].items())[:4])
        lines.append(
            f"| {rank} | {key.get('category')} | {cluster['count']} | {cluster['exact_train']} | "
            f"{cluster['best_train_accuracy']:.4f} | {cluster['min_public_mismatches']} | {tasks} | {families} |"
        )
    lines.extend(["", "## Top Non-Periodic Target", ""])
    if top_non_periodic:
        lines.append(f"Category: `{top_non_periodic['key'].get('category')}`")
        lines.append("")
        for example in top_non_periodic["examples"][:6]:
            lines.append(
                f"- `{example['task_id']}` `{example['candidate']}` "
                f"train={example['train_accuracy']:.4f} diffs={example['public_diff_counts']} "
                f"transitions={example['public_transitions']}"
            )
    else:
        lines.append("No non-periodic high-train public-miss cluster found.")
    if speculative_rows:
        lines.extend(["", "## Speculative Operator Diagnostics", ""])
        for row in speculative_rows[:20]:
            lines.append(
                f"- `{row['task_id']}` `{row['candidate']}` tier={row['tier']} "
                f"type={row.get('admission_type')} evidence={row.get('evidence_label')} "
                f"train={row['train_accuracy']:.4f} loo={row['leave_one_out']} "
                f"loo_info={row.get('loo_informative')} "
                f"jackknife={row['jackknife_stable']} confidence={row['structural_confidence']:.2f} "
                f"public={row['public_hits']}/{row['public_total']} diffs={row['public_diff_counts']} "
                f"blockers={row['promotion_blockers']}"
            )
    lines.extend(["", "## Strongest Individual Near Misses", ""])
    for row in rows[:30]:
        lines.append(
            f"- `{row['task_id']}` `{row['candidate']}` family={row['family']} "
            f"exact={row['exact_train']} train={row['train_accuracy']:.4f} "
            f"public={row['public_hits']}/{row['public_total']} diffs={row['public_diff_counts']} "
            f"category={row['category']}"
        )
    if reject_summary:
        lines.extend(["", "## Reject-Trace Clusters", ""])
        for rank, cluster in enumerate(reject_summary.get("clusters", [])[:10], start=1):
            lines.append(
                f"- {rank}. count={cluster.get('count')} best_train={cluster.get('best_train_score')} "
                f"tasks={cluster.get('tasks')} key={cluster.get('key')}"
            )
    path.write_text("\n".join(lines) + "\n")


def write_prompt_file(path: Path, rows: list[dict[str, Any]], tasks: dict[str, Any], *, limit: int) -> None:
    prompt_rows = [row for row in rows if row["exact_train"]]
    if len(prompt_rows) < limit:
        seen = {id(row) for row in prompt_rows}
        prompt_rows.extend(row for row in rows if id(row) not in seen)
    chunks = []
    for row in prompt_rows[:limit]:
        chunks.append(abstract_rule_prompt(row, tasks[row["task_id"]]))
    path.write_text("\n\n---\n\n".join(chunks) + ("\n" if chunks else ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=WORKSPACE / "arc_agi_2_data" / "evaluation")
    parser.add_argument("--out", type=Path, default=WORKSPACE / "tmp" / "arc2_nearmiss_clusters.json")
    parser.add_argument("--report", type=Path, default=WORKSPACE / "ARC2_NEARMISS_CLUSTER_REPORT.md")
    parser.add_argument("--prompt-out", type=Path, default=WORKSPACE / "tmp" / "arc2_public_wrong_abstract_prompts.md")
    parser.add_argument("--reject-log", type=Path, action="append", default=[])
    parser.add_argument("--min-train-accuracy", type=float, default=0.85)
    parser.add_argument("--per-task-candidates", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--prompt-limit", type=int, default=10)
    parser.add_argument("--disable-ttt", action="store_true")
    parser.add_argument("--include-solved-tasks", action="store_true")
    args = parser.parse_args()

    if args.disable_ttt:
        os.environ["ARC_ENABLE_TTT_PIXEL"] = "0"

    t0 = time.time()
    tasks = load_tasks(args.tasks)
    ns = load_namespace()
    skipped_solved = set() if args.include_solved_tasks else current_submission_solved_tasks(tasks, ns)
    rows, solved = mine_candidates(
        tasks,
        ns,
        min_train_accuracy=args.min_train_accuracy,
        per_task_candidates=args.per_task_candidates,
        timeout=args.timeout,
        skip_task_ids=skipped_solved,
    )
    speculative_rows = mine_speculative_operators(tasks, ns, skip_task_ids=skipped_solved, timeout=args.timeout)
    clusters = summarize_clusters(rows)

    reject_clusters = []
    reject_logs = args.reject_log or sorted((WORKSPACE / "tmp").glob("arc2_rejected*.jsonl"))
    if cluster_rejects is not None:
        for reject_log in reject_logs:
            if reject_log.exists():
                reject_clusters.append(cluster_rejects(reject_log, limit=10))

    elapsed = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(elapsed, 1),
        "min_train_accuracy": args.min_train_accuracy,
        "per_task_candidates": args.per_task_candidates,
        "skipped_current_submission_solved": sorted(skipped_solved),
        "near_misses": rows,
        "public_exact": solved,
        "speculative_operators": speculative_rows,
        "clusters": clusters,
        "reject_clusters": reject_clusters,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_prompt_file(args.prompt_out, rows, tasks, limit=args.prompt_limit)
    write_markdown_report(
        args.report,
        rows=rows,
        solved=solved,
        clusters=clusters,
        speculative_rows=speculative_rows,
        reject_summary=reject_clusters[0] if reject_clusters else None,
        elapsed_s=elapsed,
        prompt_path=args.prompt_out,
        skipped_solved=skipped_solved,
    )
    print(json.dumps({
        "out": str(args.out),
        "report": str(args.report),
        "prompt_out": str(args.prompt_out),
        "near_misses": len(rows),
        "clusters": len(clusters),
        "speculative_operators": len(speculative_rows),
        "elapsed_s": round(elapsed, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
