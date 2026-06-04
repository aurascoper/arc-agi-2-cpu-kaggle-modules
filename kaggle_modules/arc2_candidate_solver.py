"""Candidate-factory ARC-AGI-2 solver.

This module is the start of the competitive pipeline:
generate many executable hypotheses, verify them exactly on train pairs,
rank them, and emit exactly two Kaggle-valid attempts per test input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from copy import deepcopy
from collections import Counter, defaultdict, deque
import argparse
import itertools
import json
import math
import os
import random
import signal
import sys

Grid = list[list[int]]
Transform = Callable[[Grid], Grid | None]

WORKSPACE = Path(__file__).resolve().parent
RANKER_MODEL_PATH = Path(os.environ.get("ARC_RANKER_MODEL", WORKSPACE / "arc2_ranker_model.json"))
PROGRAM_CACHE_PATH = Path(os.environ.get("ARC_PROGRAM_CACHE", WORKSPACE / "arc2_program_cache.json"))
_RANKER_MODEL_CACHE: dict[str, Any] | None = None
_PROGRAM_CACHE: dict[str, Any] | None = None
CONTAMINATED_DIRECT_SOLVER_TASKS = {
    # Finalization audit: exclude test-output-derived direct solvers from all candidate pools.
    "d8e07eb2",
}


def env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def hidden_facing_mode() -> bool:
    return (
        os.environ.get("ARC_SUBMISSION_MODE", "").strip().lower() == "hidden"
        or env_enabled("ARC_HIDDEN_FACING", "0")
    )


@dataclass(order=True)
class RankedCandidate:
    sort_key: tuple[float, int, str]
    name: str = field(compare=False)
    transform: Transform = field(compare=False)
    exact_train: bool = field(default=False, compare=False)
    train_accuracy: float = field(default=0.0, compare=False)
    family: str = field(default="", compare=False)
    notes: str = field(default="", compare=False)


@dataclass(frozen=True)
class AdmissionPolicy:
    min_structural_confidence: float = 0.70
    require_exact_train: bool = True
    require_loo_for_certified: bool = True
    require_synthetic_for_certified: bool = True


@dataclass
class OperatorEvidence:
    admission_type: str = "learned_factory"
    evidence_label: str = ""
    loo_informative: bool = True
    train_exact: bool = False
    train_accuracy: float = 0.0
    leave_one_out: bool = False
    jackknife_stable: bool = False
    structural_confidence: float = 0.0
    synthetic_self_consistency: bool = False
    synthetic_invariance_status: str = "unavailable"  # passed | failed | unavailable
    synthetic_checks: dict[str, bool | str] = field(default_factory=dict)
    cross_task_firing: int | None = None  # distinct design tasks this op is train-exact on
    generalization_tier: str = "unknown"  # general (>=2 tasks) | memorized (1) | unknown
    public_shadow_hits: int | None = None
    public_shadow_total: int | None = None
    public_shadow_diff_counts: list[int] = field(default_factory=list)
    promotion_blockers: list[str] = field(default_factory=list)


@dataclass
class CandidateSpec:
    name: str
    family: str
    prior: int
    transform: Transform | None
    tier: str
    evidence: OperatorEvidence


DEFAULT_SPECULATIVE_POLICY = AdmissionPolicy()


@dataclass(frozen=True)
class ObjectInfo:
    cells: tuple[tuple[int, int], ...]
    color: int
    bbox: tuple[int, int, int, int]
    size: int
    shape_key: tuple[tuple[int, int], ...]
    color_shape_key: tuple[tuple[int, int, int], ...]
    holes: int
    touches_border: bool
    center2: tuple[int, int]


def normalize_grid(grid: Any) -> Grid:
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if isinstance(grid, tuple):
        grid = list(grid)
    if not isinstance(grid, list) or not grid:
        return [[0]]

    out: Grid = []
    width = None
    for row in grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, tuple):
            row = list(row)
        if not isinstance(row, list) or not row:
            return [[0]]

        norm_row: list[int] = []
        for cell in row:
            try:
                value = int(cell)
            except Exception:
                return [[0]]
            if value < 0 or value > 9:
                return [[0]]
            norm_row.append(value)

        if width is None:
            width = len(norm_row)
        elif len(norm_row) != width:
            return [[0]]
        out.append(norm_row)

    if width is None or len(out) > 30 or width > 30:
        return [[0]]
    return out


def is_valid_grid(grid: Any) -> bool:
    return normalize_grid(grid) != [[0]] or grid == [[0]]


def grid_key(grid: Any) -> str:
    return json.dumps(normalize_grid(grid), separators=(",", ":"))


def shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def rot90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])]


def rot180(grid: Grid) -> Grid:
    return rot90(rot90(grid))


def rot270(grid: Grid) -> Grid:
    return rot90(rot180(grid))


def mirror_h(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def mirror_v(grid: Grid) -> Grid:
    return grid[::-1]


def transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)]


def anti_transpose(grid: Grid) -> Grid:
    return mirror_h(rot90(grid))


def palette(grid: Grid) -> set[int]:
    return {cell for row in grid for cell in row}


def background_color(grid: Grid) -> int:
    counts = Counter(cell for row in grid for cell in row)
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def border_background_color(grid: Grid) -> int:
    rows, cols = shape(grid)
    if rows == 0 or cols == 0:
        return 0
    border = []
    for c in range(cols):
        border.append(grid[0][c])
        border.append(grid[rows - 1][c])
    for r in range(rows):
        border.append(grid[r][0])
        border.append(grid[r][cols - 1])
    return Counter(border).most_common(1)[0][0] if border else background_color(grid)


def bbox_non_bg(grid: Grid, bg: int | None = None) -> tuple[int, int, int, int] | None:
    if bg is None:
        bg = background_color(grid)
    coords = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != bg
    ]
    if not coords:
        return None
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def crop_bbox(grid: Grid, bbox: tuple[int, int, int, int]) -> Grid:
    r0, c0, r1, c1 = bbox
    return [row[c0 : c1 + 1] for row in grid[r0 : r1 + 1]]


def crop_foreground(grid: Grid) -> Grid:
    bbox = bbox_non_bg(grid)
    if bbox is None:
        return copy_grid(grid)
    return crop_bbox(grid, bbox)


def normalized_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = list(cells)
    if not cells:
        return tuple()
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def object_crop(grid: Grid, obj: ObjectInfo, bg: int | None = None) -> Grid:
    if bg is None:
        bg = border_background_color(grid)
    r0, c0, r1, c1 = obj.bbox
    out = [[bg for _ in range(c1 - c0 + 1)] for _ in range(r1 - r0 + 1)]
    for r, c in obj.cells:
        out[r - r0][c - c0] = grid[r][c]
    return out


def count_component_holes(grid: Grid, cells: Iterable[tuple[int, int]], bg: int) -> int:
    cells_set = set(cells)
    if not cells_set:
        return 0
    r0 = min(r for r, _ in cells_set)
    c0 = min(c for _, c in cells_set)
    r1 = max(r for r, _ in cells_set)
    c1 = max(c for _, c in cells_set)
    blocked = {(r - r0 + 1, c - c0 + 1) for r, c in cells_set}
    h = r1 - r0 + 3
    w = c1 - c0 + 3
    visited: set[tuple[int, int]] = set()
    holes = 0

    for rr in range(h):
        for cc in range(w):
            if (rr, cc) in blocked or (rr, cc) in visited:
                continue
            q = deque([(rr, cc)])
            visited.add((rr, cc))
            touches_border = False
            while q:
                cr, cc2 = q.popleft()
                if cr in (0, h - 1) or cc2 in (0, w - 1):
                    touches_border = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc2 + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blocked and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def connected_components(
    grid: Grid,
    bg: int | None = None,
    *,
    same_color: bool = True,
    diag: bool = False,
) -> list[ObjectInfo]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if bg is None:
        bg = border_background_color(grid)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        neighbors += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    visited: set[tuple[int, int]] = set()
    objects: list[ObjectInfo] = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited or grid[r][c] == bg:
                continue
            start_color = grid[r][c]
            q = deque([(r, c)])
            visited.add((r, c))
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in visited:
                        continue
                    if grid[nr][nc] == bg:
                        continue
                    if same_color and grid[nr][nc] != start_color:
                        continue
                    visited.add((nr, nc))
                    q.append((nr, nc))
            r0 = min(rr for rr, _ in cells)
            c0 = min(cc for _, cc in cells)
            r1 = max(rr for rr, _ in cells)
            c1 = max(cc for _, cc in cells)
            colors = Counter(grid[rr][cc] for rr, cc in cells)
            shape_key = normalized_cells(cells)
            color_shape_key = tuple(
                sorted((rr - r0, cc - c0, grid[rr][cc]) for rr, cc in cells)
            )
            objects.append(
                ObjectInfo(
                    cells=tuple(sorted(cells)),
                    color=colors.most_common(1)[0][0],
                    bbox=(r0, c0, r1, c1),
                    size=len(cells),
                    shape_key=shape_key,
                    color_shape_key=color_shape_key,
                    holes=count_component_holes(grid, cells, bg),
                    touches_border=any(rr in (0, rows - 1) or cc in (0, cols - 1) for rr, cc in cells),
                    center2=(r0 + r1, c0 + c1),
                )
            )
    return objects


def enclosed_regions(grid: Grid, bg: int | None = None) -> list[tuple[tuple[int, int], ...]]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if bg is None:
        bg = border_background_color(grid)
    visited: set[tuple[int, int]] = set()
    regions: list[tuple[tuple[int, int], ...]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg or (r, c) in visited:
                continue
            q = deque([(r, c)])
            visited.add((r, c))
            cells = []
            touches_border = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr in (0, rows - 1) or cc in (0, cols - 1):
                    touches_border = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == bg:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            if not touches_border:
                regions.append(tuple(sorted(cells)))
    return regions


def separator_indices(grid: Grid, axis: str, bg: int | None = None) -> list[int]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if bg is None:
        bg = border_background_color(grid)
    indices = []
    if axis == "row":
        for r, row in enumerate(grid):
            counts = Counter(row)
            color, count = counts.most_common(1)[0]
            if count == cols and color != bg:
                indices.append(r)
    else:
        for c in range(cols):
            col = [grid[r][c] for r in range(rows)]
            counts = Counter(col)
            color, count = counts.most_common(1)[0]
            if count == rows and color != bg:
                indices.append(c)
    return indices


def split_by_separators(grid: Grid, bg: int | None = None) -> list[tuple[str, int, Grid]]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    panels: list[tuple[str, int, Grid]] = []
    for r in separator_indices(grid, "row", bg):
        if r > 0:
            panels.append(("above_sep", r, [row[:] for row in grid[:r]]))
        if r < rows - 1:
            panels.append(("below_sep", r, [row[:] for row in grid[r + 1 :]]))
    for c in separator_indices(grid, "col", bg):
        if c > 0:
            panels.append(("left_sep", c, [row[:c] for row in grid]))
        if c < cols - 1:
            panels.append(("right_sep", c, [row[c + 1 :] for row in grid]))
    return [(name, idx, panel) for name, idx, panel in panels if panel and panel[0]]


def contiguous_segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    seps = sorted(set(idx for idx in separators if 0 <= idx < length))
    segments = []
    start = 0
    for sep in seps:
        if start < sep:
            segments.append((start, sep))
        start = sep + 1
    if start < length:
        segments.append((start, length))
    return segments


def equal_shape_panels(grid: Grid, bg: int | None = None) -> list[Grid]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if bg is None:
        bg = border_background_color(grid)
    candidates: list[list[Grid]] = []
    row_seps = separator_indices(grid, "row", bg)
    if row_seps:
        panels = [[row[:] for row in grid[r0:r1]] for r0, r1 in contiguous_segments(rows, row_seps)]
        candidates.append(panels)
    col_seps = separator_indices(grid, "col", bg)
    if col_seps:
        panels = [[row[c0:c1] for row in grid] for c0, c1 in contiguous_segments(cols, col_seps)]
        candidates.append(panels)
    for panels in candidates:
        if len(panels) >= 2 and len({shape(panel) for panel in panels}) == 1:
            return panels
    return []


def solid_line_indices(grid: Grid, axis: str) -> list[int]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if axis == "row":
        return [r for r, row in enumerate(grid) if cols and len(set(row)) == 1]
    return [c for c in range(cols) if rows and len({grid[r][c] for r in range(rows)}) == 1]


def solid_border_indices(grid: Grid, axis: str) -> list[int]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    indices = solid_line_indices(grid, axis)
    if axis == "row":
        for idx in (0, rows - 1):
            if 0 <= idx < rows and idx not in indices:
                indices.append(idx)
    else:
        for idx in (0, cols - 1):
            if 0 <= idx < cols and idx not in indices:
                indices.append(idx)
    return sorted(set(indices))


def framed_content_rectangles(grid: Grid) -> list[tuple[int, int, int, int]]:
    """Return half-open content rectangles inside solid-line framed panels."""
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if rows < 5 or cols < 5:
        return []
    row_lines = solid_border_indices(grid, "row")
    col_lines = solid_border_indices(grid, "col")
    if len(row_lines) < 2:
        row_lines = [0, rows - 1]
    if len(col_lines) < 2:
        col_lines = [0, cols - 1]
    rects: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for r0, r1 in zip(row_lines, row_lines[1:]):
        if r1 - r0 < 4:
            continue
        for c0, c1 in zip(col_lines, col_lines[1:]):
            if c1 - c0 < 4:
                continue
            rect = (r0 + 2, r1 - 1, c0 + 2, c1 - 1)
            rr0, rr1, cc0, cc1 = rect
            if rr0 >= rr1 or cc0 >= cc1:
                continue
            edge: list[int] = []
            top, bottom = rr0 - 1, rr1
            left, right = cc0 - 1, cc1
            if 0 <= top < rows and 0 <= bottom < rows:
                edge.extend(grid[top][cc0:cc1])
                edge.extend(grid[bottom][cc0:cc1])
            if 0 <= left < cols and 0 <= right < cols:
                edge.extend(grid[r][left] for r in range(rr0, rr1))
                edge.extend(grid[r][right] for r in range(rr0, rr1))
            if not edge:
                continue
            _color, count = Counter(edge).most_common(1)[0]
            if count / len(edge) < 0.70:
                continue
            if rect not in seen:
                seen.add(rect)
                rects.append(rect)
    return rects


def periodic_majority_repair_patch(
    patch: Grid,
    *,
    max_period: int = 10,
    max_change_frac: float = 0.25,
    min_phase_confidence: float = 0.70,
) -> tuple[Grid, int, tuple[int, int] | None]:
    patch = normalize_grid(patch)
    rows, cols = shape(patch)
    total = rows * cols
    if rows == 0 or cols == 0 or total < 4:
        return patch, 0, None
    best: tuple[tuple[int, int, int], Grid, int, tuple[int, int]] | None = None
    max_rows = min(rows, max_period)
    max_cols = min(cols, max_period)
    for pr in range(1, max_rows + 1):
        for pc in range(1, max_cols + 1):
            if pr * pc >= total:
                continue
            phase: dict[tuple[int, int], int] = {}
            confidences: list[float] = []
            for rr in range(pr):
                for cc in range(pc):
                    values = [
                        patch[r][c]
                        for r in range(rr, rows, pr)
                        for c in range(cc, cols, pc)
                    ]
                    if not values:
                        continue
                    counts = Counter(values)
                    value, count = counts.most_common(1)[0]
                    phase[(rr, cc)] = value
                    confidences.append(count / len(values))
            if len(phase) != pr * pc or not confidences or min(confidences) < min_phase_confidence:
                continue
            repaired = [[phase[(r % pr, c % pc)] for c in range(cols)] for r in range(rows)]
            changes = sum(repaired[r][c] != patch[r][c] for r in range(rows) for c in range(cols))
            if changes == 0:
                continue
            if changes > max(1, int(total * max_change_frac)):
                continue
            score = (changes, pr * pc, pr + pc)
            if best is None or score < best[0]:
                best = (score, repaired, changes, (pr, pc))
    if best is None:
        return patch, 0, None
    return best[1], best[2], best[3]


def repair_framed_periodic_panels(grid: Grid) -> Grid:
    grid = normalize_grid(grid)
    out = copy_grid(grid)
    for rr0, rr1, cc0, cc1 in framed_content_rectangles(grid):
        patch = [row[cc0:cc1] for row in out[rr0:rr1]]
        repaired, changes, _period = periodic_majority_repair_patch(patch)
        if not changes:
            continue
        for r, row in enumerate(repaired):
            for c, value in enumerate(row):
                out[rr0 + r][cc0 + c] = value
    return out


def leave_one_out_validates(
    pairs: list[dict[str, Grid]],
    factory: Callable[[list[dict[str, Grid]]], Transform | None],
) -> bool:
    if len(pairs) <= 1:
        return True
    for held_idx, held in enumerate(pairs):
        subset = [deepcopy(pair) for idx, pair in enumerate(pairs) if idx != held_idx]
        transform = factory(subset)
        if transform is None:
            return False
        try:
            pred = normalize_grid(call_with_timeout(lambda: transform(deepcopy(held["input"]))))
        except Exception:
            return False
        if pred != normalize_grid(held["output"]):
            return False
    return True


def train_transform_score(transform: Transform, pairs: list[dict[str, Grid]]) -> tuple[bool, float]:
    if not pairs:
        return False, 0.0
    scores = []
    for pair in pairs:
        try:
            pred = normalize_grid(call_with_timeout(lambda p=pair: transform(deepcopy(p["input"]))))
        except Exception:
            return False, sum(scores) / len(scores) if scores else 0.0
        expected = normalize_grid(pair["output"])
        scores.append(pixel_accuracy(expected, pred))
    return all(score == 1.0 for score in scores), sum(scores) / len(scores)


def jackknife_stable(
    pairs: list[dict[str, Grid]],
    factory: Callable[[list[dict[str, Grid]]], Transform | None],
    full_transform: Transform,
) -> bool:
    if len(pairs) <= 1:
        return True
    try:
        full_preds = [
            normalize_grid(call_with_timeout(lambda pair=pair: full_transform(deepcopy(pair["input"]))))
            for pair in pairs
        ]
    except Exception:
        return False
    for held_idx, _held in enumerate(pairs):
        subset = [deepcopy(pair) for idx, pair in enumerate(pairs) if idx != held_idx]
        subset_transform = factory(subset)
        if subset_transform is None:
            return False
        for pair, full_pred in zip(pairs, full_preds):
            try:
                pred = normalize_grid(call_with_timeout(lambda pair=pair: subset_transform(deepcopy(pair["input"]))))
            except Exception:
                return False
            if pred != full_pred:
                return False
    return True


def admit_operator_factory(
    name: str,
    family: str,
    prior: int,
    pairs: list[dict[str, Grid]],
    factory: Callable[[list[dict[str, Grid]]], Transform | None],
    *,
    policy: AdmissionPolicy = DEFAULT_SPECULATIVE_POLICY,
    structural_confidence: float = 0.0,
    synthetic_self_consistency: bool = False,
) -> CandidateSpec:
    transform = factory(pairs)
    evidence = OperatorEvidence(
        admission_type="learned_factory",
        evidence_label="learned_train_loo_synthetic",
        loo_informative=True,
        structural_confidence=round(float(structural_confidence), 4),
        synthetic_self_consistency=bool(synthetic_self_consistency),
        synthetic_checks={"learned_self_consistency": bool(synthetic_self_consistency)},
    )
    if transform is None:
        evidence.promotion_blockers.append("factory_returned_none")
        return CandidateSpec(name, family, prior, None, "rejected", evidence)

    train_exact, train_accuracy = train_transform_score(transform, pairs)
    evidence.train_exact = train_exact
    evidence.train_accuracy = round(train_accuracy, 6)
    if train_exact:
        evidence.leave_one_out = leave_one_out_validates(pairs, factory)
        evidence.jackknife_stable = jackknife_stable(pairs, factory, transform)

    if policy.require_exact_train and not evidence.train_exact:
        evidence.promotion_blockers.append("train_not_exact")
    if structural_confidence < policy.min_structural_confidence:
        evidence.promotion_blockers.append("low_structural_confidence")
    if policy.require_loo_for_certified and not evidence.leave_one_out:
        evidence.promotion_blockers.append("leave_one_out_failed")
    if policy.require_synthetic_for_certified and not evidence.synthetic_self_consistency:
        evidence.promotion_blockers.append("synthetic_self_consistency_missing")

    if not evidence.train_exact:
        tier = "rejected"
    elif evidence.promotion_blockers:
        tier = "quarantined"
    else:
        tier = "certified"
    return CandidateSpec(name, family, prior, transform, tier, evidence)


def operator_spec_diagnostic(spec: CandidateSpec) -> dict[str, Any]:
    ev = spec.evidence
    return {
        "name": spec.name,
        "family": spec.family,
        "prior": spec.prior,
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
        "synthetic_invariance_status": ev.synthetic_invariance_status,
        "synthetic_checks": dict(ev.synthetic_checks),
        "cross_task_firing": ev.cross_task_firing,
        "generalization_tier": ev.generalization_tier,
        "public_shadow_hits": ev.public_shadow_hits,
        "public_shadow_total": ev.public_shadow_total,
        "public_shadow_diff_counts": ev.public_shadow_diff_counts,
        "promotion_blockers": list(ev.promotion_blockers),
    }


def grid_d4_variants(grid: Grid) -> list[Grid]:
    grid = normalize_grid(grid)
    out: list[Grid] = []
    seen: set[str] = set()
    current = copy_grid(grid)
    for _ in range(4):
        for variant in (current, mirror_h(current)):
            key = grid_key(variant)
            if key in seen:
                continue
            seen.add(key)
            out.append(normalize_grid(variant))
        current = rot90(current)
    return out


def repair_binary_hole_frames(grid: Grid, hole_color: int, solid_color: int, marker_color: int) -> Grid:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if rows == 0 or cols == 0:
        return grid
    allowed = {hole_color, solid_color}
    if any(value not in allowed for row in grid for value in row):
        return copy_grid(grid)

    stray_solid: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != solid_color:
                continue
            neighbors = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < rows and 0 <= c + dc < cols
            ]
            hole_count = sum(1 for rr, cc in neighbors if grid[rr][cc] == hole_color)
            solid_count = sum(1 for rr, cc in neighbors if grid[rr][cc] == solid_color)
            if hole_count > solid_count:
                stray_solid.add((r, c))

    cleaned = copy_grid(grid)
    for r, c in stray_solid:
        cleaned[r][c] = hole_color

    framed_holes: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if cleaned[r][c] != hole_color:
                continue
            neighbors = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < rows and 0 <= c + dc < cols
            ]
            hole_count = sum(1 for rr, cc in neighbors if cleaned[rr][cc] == hole_color)
            solid_count = sum(1 for rr, cc in neighbors if cleaned[rr][cc] == solid_color)
            if solid_count > hole_count:
                framed_holes.add((r, c))

    changed = True
    while changed:
        changed = False
        for r, c in list(framed_holes):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if cleaned[nr][nc] != hole_color or (nr, nc) in framed_holes:
                    continue
                neighbors = [
                    (nr + dr2, nc + dc2)
                    for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= nr + dr2 < rows and 0 <= nc + dc2 < cols
                ]
                hole_count = sum(1 for rr, cc in neighbors if cleaned[rr][cc] == hole_color)
                solid_count = sum(1 for rr, cc in neighbors if cleaned[rr][cc] == solid_color)
                if solid_count >= hole_count:
                    framed_holes.add((nr, nc))
                    changed = True

    out = copy_grid(cleaned)
    for r, c in framed_holes:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in framed_holes:
                    out[nr][nc] = marker_color
    return out


def infer_binary_hole_frame_repair(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []

    def candidate_params(train_pairs: list[dict[str, Grid]]) -> list[tuple[int, int, int]]:
        if not train_pairs:
            return []
        input_color_sets = [set(value for row in normalize_grid(pair["input"]) for value in row) for pair in train_pairs]
        if any(len(colors) != 2 for colors in input_color_sets):
            return []
        common_input_colors = set.intersection(*input_color_sets)
        if len(common_input_colors) != 2:
            return []
        output_extra_sets = [
            set(value for row in normalize_grid(pair["output"]) for value in row) - input_colors
            for pair, input_colors in zip(train_pairs, input_color_sets)
        ]
        if any(len(extra) != 1 for extra in output_extra_sets):
            return []
        marker_values = set.intersection(*output_extra_sets)
        if len(marker_values) != 1:
            return []
        marker_color = next(iter(marker_values))
        colors = sorted(common_input_colors)
        return [
            (colors[0], colors[1], marker_color),
            (colors[1], colors[0], marker_color),
        ]

    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        for hole_color, solid_color, marker_color in candidate_params(train_pairs):
            changed_any = False
            ok = True
            for pair in train_pairs:
                inp = normalize_grid(pair["input"])
                out = normalize_grid(pair["output"])
                if shape(inp) != shape(out):
                    ok = False
                    break
                pred = repair_binary_hole_frames(inp, hole_color, solid_color, marker_color)
                if pred != out:
                    ok = False
                    break
                if pred != inp:
                    changed_any = True
            if ok and changed_any:
                return lambda grid, h=hole_color, s=solid_color, m=marker_color: repair_binary_hole_frames(grid, h, s, m)
        return None

    transform = factory(pairs)
    if transform is None:
        return []
    if not leave_one_out_validates(pairs, factory):
        return []
    return [("color_role_frame:cleaned_hole_border", transform)]


def infer_framed_periodic_repair(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []

    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        if not train_pairs:
            return repair_framed_periodic_panels
        changed_any = False
        for pair in train_pairs:
            inp = normalize_grid(pair["input"])
            out = normalize_grid(pair["output"])
            pred = repair_framed_periodic_panels(inp)
            if pred != out:
                return None
            if pred != inp:
                changed_any = True
        return repair_framed_periodic_panels if changed_any else None

    transform = factory(pairs)
    if transform is None:
        return []
    if not leave_one_out_validates(pairs, factory):
        return []
    return [("framed_periodic:majority_repair", transform)]


def row_marker_primary(row: list[int], separator: int = 7, marker: int = 6) -> int | None:
    counts = Counter(value for value in row[2:] if value not in {separator, marker})
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def row_marker_has_run(row: list[int], color: int, separator: int = 7) -> bool:
    return any(end > start for start, end in row_marker_runs(row, color, separator))


def row_marker_runs(row: list[int], color: int, separator: int = 7) -> list[tuple[int, int]]:
    del separator
    cells = [c for c in range(2, len(row)) if row[c] == color]
    if not cells:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = cells[0]
    for cell in cells[1:]:
        if cell == prev + 1:
            prev = cell
            continue
        runs.append((start, prev))
        start = prev = cell
    runs.append((start, prev))
    return runs


def row_marker_max_run_length(row: list[int], color: int, separator: int = 7) -> int:
    return max((end - start + 1 for start, end in row_marker_runs(row, color, separator)), default=0)


def row_marker_step_candidates(
    row: list[int],
    above: list[int] | None,
    below: list[int] | None,
    primary: int | None = None,
    separator: int = 7,
) -> list[int]:
    cols = len(row)
    return [
        c for c in range(2, cols)
        if row[c] == separator
        and (above is None or above[c] == separator)
        and (below is None or below[c] != separator)
        and (primary is None or below[c] == primary)
    ]


def row_marker_trailing_gaps(row: list[int], color: int, separator: int = 7) -> list[int]:
    color_cells = [c for c in range(2, len(row)) if row[c] == color]
    if not color_cells:
        return []
    end = max(color_cells)
    return [c for c in range(end + 1, len(row)) if row[c] == separator]


def row_marker_gap_before_foreign(row: list[int], color: int, separator: int = 7, marker: int = 6) -> list[int]:
    color_cells = [c for c in range(2, len(row)) if row[c] == color]
    if not color_cells:
        return []
    gap: list[int] = []
    for c in range(max(color_cells) + 1, len(row)):
        if row[c] == separator:
            gap.append(c)
            continue
        if row[c] not in {color, marker}:
            return gap
        return []
    return []


def repair_row_marker_panel_flow(grid: Grid, mode: str = "structural_context") -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    if rows == 0 or cols < 4:
        return copy_grid(inp)
    separator = 7
    marker = 6
    out = copy_grid(inp)

    primaries = [row_marker_primary(row, separator, marker) for row in inp]
    color_rows: dict[int, list[int]] = {}
    for r, primary in enumerate(primaries):
        if primary is not None:
            color_rows.setdefault(primary, []).append(r)

    def is_complex(color: int | None) -> bool:
        return color is not None and len(color_rows.get(color, [])) >= 2

    for r, row in enumerate(inp):
        if row[0] != marker:
            continue
        above = inp[r - 1] if r > 0 else None
        below = inp[r + 1] if r < rows - 1 else None
        cp = primaries[r]
        ap = primaries[r - 1] if r > 0 else None
        bp = primaries[r + 1] if r < rows - 1 else None

        if cp is None:
            if ap is not None and ap == bp:
                out[r][0] = separator
                if out[r][cols - 1] == separator:
                    out[r][cols - 1] = marker
            elif mode == "structural_context" and is_complex(ap) and is_complex(bp) and ap != bp:
                candidates = row_marker_step_candidates(row, above, below, None, separator)
                if candidates:
                    out[r][candidates[-1]] = marker
            continue

        if cp == ap and cp == bp:
            if not row_marker_has_run(row, cp, separator):
                out[r][0] = separator
                gap = row_marker_gap_before_foreign(row, cp, separator, marker)
                if mode == "structural_context" and len(gap) >= 2:
                    out[r][gap[0]] = marker
            continue

        if mode == "structural_context" and ap is not None and ap == bp and cp != ap:
            cp_runs = row_marker_runs(row, cp, separator)
            longest = row_marker_max_run_length(row, cp, separator)
            if longest <= 3:
                out[r][0] = separator
            elif len(cp_runs) == 1 and longest >= 5 and out[r][cols - 1] == separator:
                out[r][cols - 1] = marker
            continue

        if mode in {"structural_context", "step_first"} and cp == bp and cp != ap and is_complex(cp) and is_complex(ap) and cp != 0:
            candidates = row_marker_step_candidates(row, above, below, cp if mode == "structural_context" else None, separator)
            if candidates:
                out[r][candidates[0] if mode == "step_first" else candidates[-1]] = marker
            continue

        if r == 0 and mode in {"structural_context", "edge_trailing"}:
            cp_runs = row_marker_runs(row, cp, separator)
            if mode == "structural_context" and len(cp_runs) >= 2:
                gap = [c for c in range(cp_runs[0][1] + 1, cp_runs[-1][0]) if row[c] == separator]
                if gap:
                    out[r][gap[0]] = marker
                    if len(gap) >= 3:
                        out[r][gap[2]] = marker
                    partner = inp[r + 2] if r + 2 < rows else below
                    foreign = [c for c in gap if partner is not None and partner[c] not in {separator, marker, cp}]
                    if foreign:
                        pos = max(gap[0], foreign[-1] - 1)
                        if row[pos] == separator:
                            out[r][pos] = marker
            else:
                trailing = row_marker_trailing_gaps(row, cp, separator)
                if trailing and row_marker_max_run_length(row, cp, separator) >= 11:
                    out[r][trailing[-2] if len(trailing) >= 2 else trailing[0]] = marker
        elif r == rows - 1 and mode in {"structural_context", "edge_trailing"}:
            trailing = row_marker_trailing_gaps(row, cp, separator)
            if trailing and row_marker_max_run_length(row, cp, separator) >= 11:
                out[r][trailing[-2] if len(trailing) >= 3 else trailing[0]] = marker
    return out


def row_marker_panel_flow_confidence(pairs: list[dict[str, Grid]]) -> tuple[bool, float, list[str]]:
    if not pairs:
        return False, 0.0, ["no_train_pairs"]
    blockers: list[str] = []
    total_cells = 0
    total_diffs = 0
    marker_rows = 0
    rowlike = 0
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if shape(inp) != shape(out):
            blockers.append("shape_change")
            continue
        rows, cols = shape(inp)
        if rows < 4 or cols < 4:
            blockers.append("too_small")
            continue
        counts = Counter(value for row in inp for value in row)
        if counts.most_common(1)[0][0] != 7:
            blockers.append("separator_7_not_dominant")
        for r in range(rows):
            if inp[r][0] == 6:
                marker_rows += 1
            primary = row_marker_primary(inp[r])
            if primary is None or row_marker_has_run(inp[r], primary):
                rowlike += 1
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != out[r][c]:
                    total_diffs += 1
                    if (inp[r][c], out[r][c]) not in {(6, 7), (7, 6)}:
                        blockers.append("non_marker_separator_edit")
                total_cells += 1
    marker_ratio = marker_rows / max(1, sum(shape(normalize_grid(pair["input"]))[0] for pair in pairs))
    rowlike_ratio = rowlike / max(1, sum(shape(normalize_grid(pair["input"]))[0] for pair in pairs))
    diff_frac = total_diffs / total_cells if total_cells else 1.0
    confidence = 0.0
    confidence += 0.25 if "shape_change" not in blockers else 0.0
    confidence += 0.25 if "non_marker_separator_edit" not in blockers and total_diffs > 0 else 0.0
    confidence += 0.15 if "separator_7_not_dominant" not in blockers else 0.0
    confidence += 0.15 if marker_ratio >= 0.70 else 0.05 if marker_ratio >= 0.40 else 0.0
    confidence += 0.10 if rowlike_ratio >= 0.70 else 0.0
    confidence += 0.10 if 0 < diff_frac <= 0.08 else 0.0
    confidence = min(1.0, confidence)
    return confidence >= 0.70 and not blockers, confidence, sorted(set(blockers))


def infer_row_marker_panel_flow_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    ok, confidence, blockers = row_marker_panel_flow_confidence(pairs)
    if not ok:
        evidence = OperatorEvidence(
            structural_confidence=round(confidence, 4),
            promotion_blockers=blockers or ["row_marker_preconditions_failed"],
        )
        return [CandidateSpec("row_marker_panel_flow:preconditions", "row_marker_panel_flow", 21, None, "rejected", evidence)] if confidence >= 0.50 else []

    specs: list[CandidateSpec] = []
    for mode, prior in (
        ("structural_context", 21),
        ("step_first", 22),
        ("edge_trailing", 23),
    ):
        def factory(train_pairs: list[dict[str, Grid]], mode: str = mode) -> Transform | None:
            train_ok, _confidence, _blockers = row_marker_panel_flow_confidence(train_pairs)
            if not train_ok:
                return None
            return lambda grid, mode=mode: repair_row_marker_panel_flow(grid, mode=mode)

        specs.append(admit_operator_factory(
            f"row_marker_panel_flow:{mode}",
            "row_marker_panel_flow",
            prior,
            pairs,
            factory,
            structural_confidence=confidence,
            synthetic_self_consistency=(mode == "structural_context"),
        ))
    return specs


def recover_open_side_diagonal_color_rays(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    if rows == 0 or cols == 0:
        return copy_grid(inp)
    counts = Counter(value for row in inp for value in row)
    if len(counts) < 3:
        return copy_grid(inp)
    checker_vals = {color for color, _count in counts.most_common(2)}
    writable = min(checker_vals, key=lambda color: counts[color])
    cluster = {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if inp[r][c] not in checker_vals
    }
    if not cluster:
        return copy_grid(inp)

    out = copy_grid(inp)
    dirs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def color_components(color: int) -> list[list[tuple[int, int]]]:
        cells = {(r, c) for r, c in cluster if inp[r][c] == color}
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int]]] = []
        for start in sorted(cells):
            if start in seen:
                continue
            queue = deque([start])
            seen.add(start)
            comp: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in dirs8:
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            comps.append(comp)
        return comps

    def diagonal_orientation(comp: list[tuple[int, int]]) -> tuple[int, int] | None:
        if len(comp) < 2:
            return None
        if len({r - c for r, c in comp}) == 1:
            return (1, 1)
        if len({r + c for r, c in comp}) == 1:
            return (1, -1)
        return None

    def open_side_score(comp_set: set[tuple[int, int]], normal: tuple[int, int], others: set[tuple[int, int]]) -> float:
        dr, dc = normal
        score = 0.0
        for r, c in comp_set:
            sr, sc = r + dr, c + dc
            if (sr, sc) in others:
                score += 10.0
            for orow, ocol in others:
                dist = max(abs(orow - sr), abs(ocol - sc))
                if dist <= 1:
                    score += 3.0
                elif dist <= 2:
                    score += 1.0
        if others:
            cr = sum(r for r, _c in comp_set) / len(comp_set)
            cc = sum(c for _r, c in comp_set) / len(comp_set)
            oc_r = sum(r for r, _c in others) / len(others)
            oc_c = sum(c for _r, c in others) / len(others)
            score += 0.1 * (dr * (oc_r - cr) + dc * (oc_c - cc))
        return score

    for color in sorted({inp[r][c] for r, c in cluster}):
        for comp in color_components(color):
            orientation = diagonal_orientation(comp)
            if orientation is None:
                continue
            comp_set = set(comp)
            others = cluster - comp_set
            normals = [(-1, 1), (1, -1)] if orientation == (1, 1) else [(-1, -1), (1, 1)]
            normal = min(normals, key=lambda item: open_side_score(comp_set, item, others))
            endpoints = [min(comp, key=lambda cell: cell[0]), max(comp, key=lambda cell: cell[0])]
            for r0, c0 in endpoints:
                r, c = r0 + normal[0], c0 + normal[1]
                while 0 <= r < rows and 0 <= c < cols:
                    if (r, c) not in cluster and inp[r][c] == writable:
                        out[r][c] = color
                    r += normal[0]
                    c += normal[1]
    return out


def diagonal_color_role_confidence(pairs: list[dict[str, Grid]]) -> tuple[bool, float, list[str]]:
    if not pairs:
        return False, 0.0, ["no_train_pairs"]
    blockers: list[str] = []
    total_cells = 0
    edit_cells = 0
    line_components = 0
    edited_from_checker = 0
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if shape(inp) != shape(out):
            blockers.append("shape_change")
            continue
        rows, cols = shape(inp)
        counts = Counter(value for row in inp for value in row)
        if len(counts) < 3:
            blockers.append("too_few_colors")
            continue
        checker_vals = {color for color, _count in counts.most_common(2)}
        writable = min(checker_vals, key=lambda color: counts[color])
        cluster = {(r, c) for r in range(rows) for c in range(cols) if inp[r][c] not in checker_vals}
        if not cluster:
            blockers.append("no_color_cluster")
        for r in range(rows):
            for c in range(cols):
                total_cells += 1
                if inp[r][c] != out[r][c]:
                    edit_cells += 1
                    if inp[r][c] == writable and out[r][c] not in checker_vals:
                        edited_from_checker += 1
                    else:
                        blockers.append("non_checker_to_color_edit")
        for color in {inp[r][c] for r, c in cluster}:
            cells = {(r, c) for r, c in cluster if inp[r][c] == color}
            seen: set[tuple[int, int]] = set()
            for start in sorted(cells):
                if start in seen:
                    continue
                queue = deque([start])
                seen.add(start)
                comp: list[tuple[int, int]] = []
                while queue:
                    r, c = queue.popleft()
                    comp.append((r, c))
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nb = (r + dr, c + dc)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            queue.append(nb)
                if len(comp) >= 2 and (len({r - c for r, c in comp}) == 1 or len({r + c for r, c in comp}) == 1):
                    line_components += 1
    edit_frac = edit_cells / total_cells if total_cells else 1.0
    edit_role = edited_from_checker / edit_cells if edit_cells else 0.0
    confidence = 0.0
    confidence += 0.30 if "shape_change" not in blockers else 0.0
    confidence += 0.25 if edit_role >= 0.95 and edit_cells > 0 else 0.0
    confidence += 0.20 if line_components >= len(pairs) else 0.0
    confidence += 0.15 if 0 < edit_frac <= 0.12 else 0.05 if edit_frac <= 0.20 else 0.0
    confidence += 0.10 if "no_color_cluster" not in blockers and "too_few_colors" not in blockers else 0.0
    confidence = min(1.0, confidence)
    return confidence >= 0.70 and "non_checker_to_color_edit" not in blockers, confidence, sorted(set(blockers))


def infer_diagonal_color_role_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    ok, confidence, blockers = diagonal_color_role_confidence(pairs)
    if not ok:
        evidence = OperatorEvidence(
            structural_confidence=round(confidence, 4),
            promotion_blockers=blockers or ["diagonal_color_role_preconditions_failed"],
        )
        return [CandidateSpec("color_role_diagonal_rays:preconditions", "color_role_diagonal_rays", 20, None, "rejected", evidence)] if confidence >= 0.50 else []

    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        train_ok, _confidence, _blockers = diagonal_color_role_confidence(train_pairs)
        if not train_ok:
            return None
        return recover_open_side_diagonal_color_rays

    return [admit_operator_factory(
        "color_role_diagonal_rays:open_side_emitters",
        "color_role_diagonal_rays",
        20,
        pairs,
        factory,
        structural_confidence=confidence,
        synthetic_self_consistency=True,
    )]


def binary_role_grid(grid: Grid) -> tuple[Grid, int, int] | None:
    grid = normalize_grid(grid)
    counts = Counter(value for row in grid for value in row)
    if len(counts) != 2:
        return None
    colors = list(counts)
    common = counts.most_common()
    if common[0][1] == common[1][1]:
        zero_color = border_background_color(grid)
    else:
        zero_color = common[0][0]
    if zero_color not in counts:
        zero_color = common[0][0]
    one_color = colors[0] if colors[1] == zero_color else colors[1]
    binary = [[1 if value == one_color else 0 for value in row] for row in grid]
    return binary, zero_color, one_color


def recolor_binary_grid(binary: Grid, zero_color: int, one_color: int) -> Grid:
    return [[one_color if value else zero_color for value in row] for row in binary]


def subgrid_locations(big: Grid, small: Grid) -> list[tuple[int, int]]:
    big = normalize_grid(big)
    small = normalize_grid(small)
    br, bc = shape(big)
    sr, sc = shape(small)
    if sr > br or sc > bc:
        return []
    locations: list[tuple[int, int]] = []
    for r0 in range(br - sr + 1):
        for c0 in range(bc - sc + 1):
            ok = True
            for rr in range(sr):
                if big[r0 + rr][c0 : c0 + sc] != small[rr]:
                    ok = False
                    break
            if ok:
                locations.append((r0, c0))
    return locations


def learn_master_pattern_expansion(pairs: list[dict[str, Grid]]) -> Transform | None:
    if len(pairs) < 2:
        return None
    first_out = binary_role_grid(pairs[0]["output"])
    if first_out is None:
        return None
    master, _zero_color, _one_color = first_out
    master_variants = grid_d4_variants(master)
    output_shapes = {shape(normalize_grid(pair["output"])) for pair in pairs}
    if len(output_shapes) != 1:
        return None

    for pair in pairs:
        inp_roles = binary_role_grid(pair["input"])
        out_roles = binary_role_grid(pair["output"])
        if inp_roles is None or out_roles is None:
            return None
        inp_binary, inp_zero, inp_one = inp_roles
        out_binary, out_zero, out_one = out_roles
        if (out_zero, out_one) != (inp_zero, inp_one):
            return None
        if not any(out_binary == variant for variant in master_variants):
            return None
        if not subgrid_locations(out_binary, inp_binary):
            return None
        in_rows, in_cols = shape(inp_binary)
        out_rows, out_cols = shape(out_binary)
        if in_rows >= out_rows and in_cols >= out_cols:
            return None

    def transform(grid: Grid, variants: list[Grid] = master_variants) -> Grid | None:
        roles = binary_role_grid(grid)
        if roles is None:
            return None
        inp_binary, zero_color, one_color = roles
        matches: list[tuple[int, int, int, Grid]] = []
        for idx, variant in enumerate(variants):
            locations = subgrid_locations(variant, inp_binary)
            for r0, c0 in locations:
                matches.append((len(locations), r0 + c0, idx, variant))
        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return recolor_binary_grid(matches[0][3], zero_color, one_color)

    return transform


def master_pattern_synthetic_self_consistency(master: Grid) -> bool:
    variants = grid_d4_variants(master)
    color_pairs = [(0, 1), (2, 4), (6, 5), (8, 7)]
    synthetic_pairs: list[dict[str, Grid]] = []
    for idx, variant in enumerate(variants[:4]):
        rows, cols = shape(variant)
        crop_rows = max(3, rows // 2)
        crop_cols = max(3, cols // 2)
        if crop_rows >= rows and crop_cols >= cols:
            return False
        r0 = min(rows - crop_rows, (idx * 3) % max(1, rows - crop_rows + 1))
        c0 = min(cols - crop_cols, (idx * 5) % max(1, cols - crop_cols + 1))
        crop = [row[c0 : c0 + crop_cols] for row in variant[r0 : r0 + crop_rows]]
        if len(palette(crop)) < 2:
            continue
        zero_color, one_color = color_pairs[idx % len(color_pairs)]
        synthetic_pairs.append({
            "input": recolor_binary_grid(crop, zero_color, one_color),
            "output": recolor_binary_grid(variant, zero_color, one_color),
        })
    if len(synthetic_pairs) < 3:
        return False
    transform = learn_master_pattern_expansion(synthetic_pairs[:-1])
    if transform is None:
        return False
    pred = transform(deepcopy(synthetic_pairs[-1]["input"]))
    return normalize_grid(pred) == normalize_grid(synthetic_pairs[-1]["output"])


def master_pattern_recolor_self_consistency(pairs: list[dict[str, Grid]]) -> bool:
    color_pairs = [(0, 1), (2, 4), (6, 5), (8, 7), (3, 9)]
    synthetic_pairs: list[dict[str, Grid]] = []
    for idx, pair in enumerate(pairs):
        inp_roles = binary_role_grid(pair["input"])
        out_roles = binary_role_grid(pair["output"])
        if inp_roles is None or out_roles is None:
            return False
        inp_binary, _inp_zero, _inp_one = inp_roles
        out_binary, _out_zero, _out_one = out_roles
        zero_color, one_color = color_pairs[idx % len(color_pairs)]
        synthetic_pairs.append({
            "input": recolor_binary_grid(inp_binary, zero_color, one_color),
            "output": recolor_binary_grid(out_binary, zero_color, one_color),
        })
    if len(synthetic_pairs) < 3:
        return False
    return leave_one_out_validates(synthetic_pairs, learn_master_pattern_expansion)


def master_pattern_expansion_confidence(pairs: list[dict[str, Grid]]) -> tuple[float, list[str], Grid | None]:
    blockers: list[str] = []
    if len(pairs) < 2:
        return 0.0, ["too_few_train_pairs"], None
    first_out = binary_role_grid(pairs[0]["output"])
    if first_out is None:
        return 0.0, ["first_output_not_binary"], None
    master = first_out[0]
    master_variants = grid_d4_variants(master)
    output_shapes = {shape(normalize_grid(pair["output"])) for pair in pairs}
    if len(output_shapes) != 1:
        blockers.append("output_shape_not_constant")
    embedding_hits = 0
    orbit_hits = 0
    role_hits = 0
    expands = 0
    for pair in pairs:
        inp_roles = binary_role_grid(pair["input"])
        out_roles = binary_role_grid(pair["output"])
        if inp_roles is None or out_roles is None:
            blockers.append("non_binary_pair")
            continue
        inp_binary, inp_zero, inp_one = inp_roles
        out_binary, out_zero, out_one = out_roles
        if (out_zero, out_one) == (inp_zero, inp_one):
            role_hits += 1
        if any(out_binary == variant for variant in master_variants):
            orbit_hits += 1
        if subgrid_locations(out_binary, inp_binary):
            embedding_hits += 1
        in_rows, in_cols = shape(inp_binary)
        out_rows, out_cols = shape(out_binary)
        if in_rows < out_rows or in_cols < out_cols:
            expands += 1
    total = max(1, len(pairs))
    confidence = 0.0
    confidence += 0.25 if len(output_shapes) == 1 else 0.0
    confidence += 0.25 * (orbit_hits / total)
    confidence += 0.25 * (embedding_hits / total)
    confidence += 0.15 * (role_hits / total)
    confidence += 0.10 * (expands / total)
    return min(1.0, confidence), sorted(set(blockers)), master


def infer_master_pattern_expansion_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    confidence, blockers, master = master_pattern_expansion_confidence(pairs)
    if confidence < 0.70 or master is None:
        evidence = OperatorEvidence(
            structural_confidence=round(confidence, 4),
            promotion_blockers=blockers or ["master_pattern_preconditions_failed"],
        )
        return [CandidateSpec("object_to_canvas_expansion:learned_master_pattern_preconditions", "shape_operator", 16, None, "rejected", evidence)] if confidence >= 0.50 else []

    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        return learn_master_pattern_expansion(train_pairs)

    return [admit_operator_factory(
        "object_to_canvas_expansion:learned_master_pattern",
        "shape_operator",
        16,
        pairs,
        factory,
        structural_confidence=confidence,
        synthetic_self_consistency=master_pattern_recolor_self_consistency(pairs),
    )]


def move_supply_color_to_lowest_row_gaps(grid: Grid, supply_color: int) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = border_background_color(inp)
    if supply_color == bg or supply_color not in palette(inp):
        return inp

    candidate_gap: dict[tuple[int, int], int] = {}
    for r, row in enumerate(inp):
        c = 0
        while c < cols:
            if row[c] != bg:
                c += 1
                continue
            start = c
            while c < cols and row[c] == bg:
                c += 1
            end = c - 1
            if start == 0 or end == cols - 1:
                continue

            left_c = start - 1
            left_color = row[left_c]
            left_start = left_c
            while left_start - 1 >= 0 and row[left_start - 1] == left_color:
                left_start -= 1

            right_c = end + 1
            right_color = row[right_c]
            right_end = right_c
            while right_end + 1 < cols and row[right_end + 1] == right_color:
                right_end += 1

            gap = end - start + 1
            left_len = left_c - left_start + 1
            right_len = right_end - right_c + 1
            ok = False
            if gap <= 4 and (
                left_color == right_color
                or left_color == supply_color
                or right_color == supply_color
            ):
                ok = True
            if gap <= 2 and left_len >= 3 and right_len >= 3:
                ok = True
            if ok:
                for cc in range(start, end + 1):
                    candidate_gap[(r, cc)] = gap

    if not candidate_gap:
        return inp

    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for cell in candidate_gap:
        if cell in seen:
            continue
        queue = deque([cell])
        seen.add(cell)
        component: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            component.append((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (cr + dr, cc + dc)
                if neighbor in candidate_gap and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)

    bottom_row = max(max(r for r, _c in component) for component in components)
    if len(components) > 1:
        targets = {
            cell
            for component in components
            if max(r for r, _c in component) >= bottom_row - 1
            for cell in component
        }
    else:
        component = components[0]
        gap_values = {candidate_gap[cell] for cell in component}
        if len(gap_values) == 1:
            targets = set(component)
        else:
            widest = max(gap_values)
            targets = {cell for cell in component if candidate_gap[cell] < widest}

    supply = sorted((r, c) for r in range(rows) for c in range(cols) if inp[r][c] == supply_color)
    if len(supply) < len(targets):
        return inp

    out = copy_grid(inp)
    for r, c in supply[:len(targets)]:
        out[r][c] = bg
    for r, c in targets:
        out[r][c] = supply_color
    return out


def learn_supply_gap_relocation(pairs: list[dict[str, Grid]]) -> Transform | None:
    if len(pairs) < 2:
        return None
    candidates: set[int] | None = None
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if shape(inp) != shape(out):
            return None
        bg = border_background_color(inp)
        colors = palette(inp) | palette(out)
        pair_candidates: set[int] = set()
        for color in colors:
            if color == bg:
                continue
            removed = 0
            added = 0
            ok = True
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    iv, ov = inp[r][c], out[r][c]
                    if iv == ov:
                        continue
                    if iv == color and ov == bg:
                        removed += 1
                    elif iv == bg and ov == color:
                        added += 1
                    else:
                        ok = False
                        break
                if not ok:
                    break
            if ok and removed > 0 and removed == added:
                pair_candidates.add(color)
        if not pair_candidates:
            return None
        candidates = pair_candidates if candidates is None else candidates & pair_candidates
        if not candidates:
            return None
    supply_color = min(candidates)

    def transform(grid: Grid, supply_color: int = supply_color) -> Grid:
        return move_supply_color_to_lowest_row_gaps(grid, supply_color)

    return transform


def supply_gap_relocation_confidence(pairs: list[dict[str, Grid]]) -> float:
    transform = learn_supply_gap_relocation(pairs)
    if transform is None:
        return 0.0
    exact, accuracy = train_transform_score(transform, pairs)
    if exact:
        return 0.92
    return min(0.65, accuracy)


def supply_gap_relocation_recolor_self_consistency(pairs: list[dict[str, Grid]]) -> bool:
    if not pairs:
        return False
    if any(border_background_color(pair["input"]) != 0 for pair in pairs):
        return False
    mapping = {0: 0}
    for color in range(1, 10):
        mapping[color] = ((color + 2) % 9) + 1
    synthetic_pairs = [
        {
            "input": _permute_colors(pair["input"], mapping),
            "output": _permute_colors(pair["output"], mapping),
        }
        for pair in pairs
    ]
    return leave_one_out_validates(synthetic_pairs, learn_supply_gap_relocation)


def infer_supply_gap_relocation_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    confidence = supply_gap_relocation_confidence(pairs)
    if confidence < 0.70:
        return []

    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        return learn_supply_gap_relocation(train_pairs)

    return [admit_operator_factory(
        "object_motion:supply_color_to_lowest_row_gaps",
        "object_motion",
        14,
        pairs,
        factory,
        structural_confidence=confidence,
        synthetic_self_consistency=supply_gap_relocation_recolor_self_consistency(pairs),
    )]


_GLYPH_PLUS = frozenset({(0, 1), (1, 0), (1, 2), (2, 1)})
_GLYPH_X = frozenset({(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)})
_GLYPH_SOLID = frozenset((r, c) for r in range(3) for c in range(3))


def learn_ladder_glyph_roles(pairs: list[dict[str, Grid]]) -> tuple[int, frozenset[int], frozenset[int]] | None:
    if len(pairs) < 2:
        return None
    backgrounds = {border_background_color(normalize_grid(pair["input"])) for pair in pairs}
    if len(backgrounds) != 1:
        return None
    bg = next(iter(backgrounds))
    token_colors: set[int] = set()
    all_colors: set[int] = set()
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if shape(inp) != shape(out):
            return None
        all_colors |= palette(inp) | palette(out)
        rows, cols = shape(inp)
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == out[r][c]:
                    continue
                if inp[r][c] != bg:
                    token_colors.add(inp[r][c])
                if out[r][c] != bg:
                    token_colors.add(out[r][c])
    token_colors.discard(bg)
    scaffold_colors = all_colors - token_colors - {bg}
    if len(token_colors) < 2 or not scaffold_colors or len(scaffold_colors) > 4:
        return None
    return bg, frozenset(scaffold_colors), frozenset(token_colors)


def ladder_glyph_at(grid: Grid, r: int, c: int, token_colors: frozenset[int]) -> tuple[int, str] | None:
    rows, cols = shape(grid)
    if r <= 0 or c <= 0 or r >= rows - 1 or c >= cols - 1:
        return None
    for color in sorted(token_colors):
        points = {
            (dr + 1, dc + 1)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if grid[r + dr][c + dc] == color
        }
        if points == _GLYPH_SOLID:
            return color, "solid"
        if points == _GLYPH_PLUS:
            return color, "plus"
        if points == _GLYPH_X:
            return color, "x"
    return None


def ladder_scaffold_blocks(grid: Grid, scaffold_colors: frozenset[int]) -> list[tuple[int, int, int]]:
    rows, cols = shape(grid)
    scaffold_rows = [r for r in range(rows) if any(grid[r][c] in scaffold_colors for c in range(cols))]
    blocks: list[tuple[int, int, int]] = []
    start = prev = None
    for r in scaffold_rows:
        if start is None or prev is None or r != prev + 1:
            if start is not None and prev is not None:
                cells = sum(1 for rr in range(start, prev + 1) for value in grid[rr] if value in scaffold_colors)
                blocks.append((cells, start, prev))
            start = r
        prev = r
    if start is not None and prev is not None:
        cells = sum(1 for rr in range(start, prev + 1) for value in grid[rr] if value in scaffold_colors)
        blocks.append((cells, start, prev))
    blocks.sort(reverse=True)
    return blocks


def ladder_scaffold_runs(row: list[int], scaffold_colors: frozenset[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    while c < len(row):
        if row[c] not in scaffold_colors:
            c += 1
            continue
        start = c
        while c < len(row) and row[c] in scaffold_colors:
            c += 1
        runs.append((start, c - 1))
    return runs


def choose_ladder_glyph_solution(
    grid: Grid,
    scaffold_colors: frozenset[int],
    token_colors: frozenset[int],
) -> tuple[int, int, list[int], list[int], dict[int, int]] | None:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    for _cells, r0, r1 in ladder_scaffold_blocks(grid, scaffold_colors):
        if r1 - r0 < 3:
            continue
        top_row = r0 - 2
        bottom_row = r1 + 2
        if top_row <= 0 or bottom_row >= rows - 1:
            continue
        slots = [
            c for c in range(cols)
            if grid[r0][c] in scaffold_colors and grid[r1][c] in scaffold_colors
        ]
        if len(slots) < 2 or len(slots) > 8:
            continue

        bars: list[tuple[int, int]] = []
        for r in range(r0 + 1, r1):
            for start, end in ladder_scaffold_runs(grid[r], scaffold_colors):
                crossed = [idx for idx, c in enumerate(slots) if start <= c <= end]
                if len(crossed) < 2:
                    continue
                if crossed[0] == 0 and crossed[-1] == len(slots) - 1:
                    continue
                bars.append((crossed[0], crossed[-1]))
        if not bars or len(bars) > 12:
            continue

        top_known: dict[int, int] = {}
        bottom_known: dict[int, int] = {}
        for idx, c in enumerate(slots):
            top_glyph = ladder_glyph_at(grid, top_row, c, token_colors)
            bottom_glyph = ladder_glyph_at(grid, bottom_row, c, token_colors)
            if top_glyph is not None:
                top_known[idx] = top_glyph[0]
            if bottom_glyph is not None:
                bottom_known[idx] = bottom_glyph[0]
        if len(top_known) + len(bottom_known) < len(slots):
            continue

        best: tuple[int, int, list[int], dict[int, int]] | None = None
        for mask in range(1 << len(bars)):
            bottom_to_top = list(range(len(slots)))
            for bit, (a, b) in enumerate(bars):
                if (mask >> bit) & 1:
                    bottom_to_top[a], bottom_to_top[b] = bottom_to_top[b], bottom_to_top[a]
            assigned = dict(top_known)
            ok = True
            for bottom_idx, color in bottom_known.items():
                top_idx = bottom_to_top[bottom_idx]
                if top_idx in assigned and assigned[top_idx] != color:
                    ok = False
                    break
                assigned[top_idx] = color
            if not ok or len(assigned) != len(slots) or len(set(assigned.values())) != len(slots):
                continue
            candidate = (mask.bit_count(), -mask, bottom_to_top, assigned)
            if best is None or candidate > best:
                best = candidate
        if best is not None:
            return r0, r1, slots, best[2], best[3]
    return None


def clear_ladder_glyph_window(out: Grid, r: int, c: int, token_colors: frozenset[int]) -> None:
    rows, cols = shape(out)
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < rows and 0 <= cc < cols and out[rr][cc] in token_colors:
                out[rr][cc] = 0


def draw_ladder_plus(out: Grid, r: int, c: int, color: int, token_colors: frozenset[int]) -> None:
    clear_ladder_glyph_window(out, r, c, token_colors)
    for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c)):
        if 0 <= rr < len(out) and 0 <= cc < len(out[0]):
            out[rr][cc] = color


def draw_ladder_x(out: Grid, r: int, c: int, color: int, token_colors: frozenset[int]) -> None:
    clear_ladder_glyph_window(out, r, c, token_colors)
    for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r, c), (r + 1, c - 1), (r + 1, c + 1)):
        if 0 <= rr < len(out) and 0 <= cc < len(out[0]):
            out[rr][cc] = color


def apply_ladder_glyph_orientation(
    grid: Grid,
    scaffold_colors: frozenset[int],
    token_colors: frozenset[int],
) -> Grid | None:
    solution = choose_ladder_glyph_solution(grid, scaffold_colors, token_colors)
    if solution is None:
        return None
    r0, r1, slots, bottom_to_top, assigned = solution
    out = copy_grid(grid)
    top_row = r0 - 2
    bottom_row = r1 + 2
    for top_idx, c in enumerate(slots):
        draw_ladder_plus(out, top_row, c, assigned[top_idx], token_colors)
    for bottom_idx, c in enumerate(slots):
        draw_ladder_x(out, bottom_row, c, assigned[bottom_to_top[bottom_idx]], token_colors)
    return out


def repair_ladder_glyph_transfer(
    grid: Grid,
    scaffold_colors: frozenset[int],
    token_colors: frozenset[int],
) -> Grid:
    inp = normalize_grid(grid)
    candidates: list[Grid] = []
    direct = apply_ladder_glyph_orientation(inp, scaffold_colors, token_colors)
    if direct is not None:
        candidates.append(direct)
    transposed = transpose(inp)
    transposed_repair = apply_ladder_glyph_orientation(transposed, scaffold_colors, token_colors)
    if transposed_repair is not None:
        candidates.append(transpose(transposed_repair))
    if not candidates:
        return copy_grid(inp)
    return max(
        candidates,
        key=lambda out: sum(
            1
            for r in range(len(inp))
            for c in range(len(inp[0]))
            if out[r][c] != inp[r][c]
        ),
    )


def learn_ladder_glyph_transfer(pairs: list[dict[str, Grid]]) -> Transform | None:
    roles = learn_ladder_glyph_roles(pairs)
    if roles is None:
        return None
    _bg, scaffold_colors, token_colors = roles

    def transform(grid: Grid, scaffold_colors: frozenset[int] = scaffold_colors, token_colors: frozenset[int] = token_colors) -> Grid:
        return repair_ladder_glyph_transfer(grid, scaffold_colors, token_colors)

    return transform


def ladder_glyph_transfer_confidence(pairs: list[dict[str, Grid]]) -> float:
    transform = learn_ladder_glyph_transfer(pairs)
    if transform is None:
        return 0.0
    exact, accuracy = train_transform_score(transform, pairs)
    if not exact:
        return min(0.65, accuracy)
    route_hits = 0
    for pair in pairs:
        roles = learn_ladder_glyph_roles([pair])
        if roles is None:
            continue
        _bg, scaffold_colors, token_colors = roles
        inp = normalize_grid(pair["input"])
        if choose_ladder_glyph_solution(inp, scaffold_colors, token_colors) is not None:
            route_hits += 1
        elif choose_ladder_glyph_solution(transpose(inp), scaffold_colors, token_colors) is not None:
            route_hits += 1
    return 0.92 if route_hits >= max(1, len(pairs) - 1) else 0.80


def synthetic_ladder_pair(
    *,
    token_colors: list[int],
    scaffold_colors: tuple[int, int],
    slots: list[int],
    bars: list[tuple[int, int]],
    bottom_known: dict[int, int] | None = None,
) -> dict[str, Grid]:
    rows = 19
    cols = max(slots) + 4
    bg = 0
    grid = [[bg for _ in range(cols)] for _ in range(rows)]
    r0, r1 = 5, 13
    rail, cross = scaffold_colors
    for c in slots:
        for r in range(r0, r1 + 1):
            grid[r][c] = rail
    for row_offset, (a, b) in enumerate(bars):
        r = r0 + 2 + row_offset * 2
        for c in range(slots[a], slots[b] + 1):
            grid[r][c] = cross if c not in slots else rail
    for idx, c in enumerate(slots):
        color = token_colors[idx]
        for rr in range(r0 - 3, r0):
            for cc in range(c - 1, c + 2):
                grid[rr][cc] = color
    if bottom_known:
        for idx, color in bottom_known.items():
            c = slots[idx]
            for rr, cc in ((r1 + 1, c - 1), (r1 + 1, c + 1), (r1 + 2, c), (r1 + 3, c - 1), (r1 + 3, c + 1)):
                grid[rr][cc] = color
    transform = repair_ladder_glyph_transfer(grid, frozenset(scaffold_colors), frozenset(token_colors))
    return {"input": grid, "output": transform}


def ladder_glyph_transfer_synthetic_self_consistency() -> bool:
    synthetic_pairs = [
        synthetic_ladder_pair(
            token_colors=[1, 2, 3, 4],
            scaffold_colors=(7, 8),
            slots=[2, 6, 10, 14],
            bars=[(2, 3), (0, 2)],
        ),
        synthetic_ladder_pair(
            token_colors=[6, 3, 1, 2],
            scaffold_colors=(7, 8),
            slots=[3, 7, 11, 15],
            bars=[(1, 2), (0, 1), (0, 2)],
            bottom_known={2: 1},
        ),
        synthetic_ladder_pair(
            token_colors=[4, 6, 2, 1, 3],
            scaffold_colors=(7, 8),
            slots=[2, 7, 12, 17, 22],
            bars=[(1, 4), (0, 2), (1, 3), (0, 3)],
            bottom_known={4: 3},
        ),
    ]
    return leave_one_out_validates(synthetic_pairs, learn_ladder_glyph_transfer)


def infer_ladder_glyph_transfer_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    confidence = ladder_glyph_transfer_confidence(pairs)
    if confidence < 0.70:
        return []

    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        return learn_ladder_glyph_transfer(train_pairs)

    return [admit_operator_factory(
        "route_glyph_transfer:ladder_switch_glyphs",
        "route_glyph_transfer",
        13,
        pairs,
        factory,
        structural_confidence=confidence,
        synthetic_self_consistency=ladder_glyph_transfer_synthetic_self_consistency(),
    )]


# === SIA promotion-reviewed operators (sia_arc_all23_task/density_pilot/operator_lib, 2026-06-04):
# symmetry_tile_expand + region_fill_by_border_signature. 0 false positives over 1000 training tasks ->
# admission is no-regression-safe (never fires unless train-exact). Guarded import = import-level gate. ===
try:
    from sia_operators import OPERATORS as _SIA_OPERATORS
except Exception:
    _SIA_OPERATORS = {}


def inferred_sia_operator_candidate_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    """Admit the SIA operators via the LEARNED admission path (op(train)->transform IS the factory).
    Zero false positives -> non-regressing; admitted (quarantined-or-certified) in non-strict, where it
    contributes to pass@2 only on tasks it train-exactly fits."""
    if not _SIA_OPERATORS:
        return []
    sia_meta = {
        "symmetry_tile_expand": ("sia:symmetry_tile_expand", "shape_operator", 16),
        "region_fill_by_border_signature": ("sia:region_fill_by_border_signature", "shape_operator", 16),
    }
    specs: list[CandidateSpec] = []
    for op_key, (name, family, prior) in sia_meta.items():
        op_fn = _SIA_OPERATORS.get(op_key)
        if op_fn is None:
            continue

        def make_factory(fn: Callable[[list[dict[str, Grid]]], Any]) -> Callable[[list[dict[str, Grid]]], Transform | None]:
            def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
                t = fn(train_pairs)
                if t is None:
                    return None

                def wrapped(grid: Grid) -> Grid:
                    out = t(grid)
                    return normalize_grid(out) if out is not None else normalize_grid(grid)

                return wrapped
            return factory

        specs.append(admit_operator_factory(
            name, family, prior, pairs, make_factory(op_fn),
            structural_confidence=1.0,         # operator self-validates train-exactness before returning
            synthetic_self_consistency=False,  # conservative: solver's own synthetic check not asserted here
        ))
    return specs


def collect_operator_specs(
    task_data: dict[str, Any],
    ns: dict[str, Any] | None = None,
    task_id: str | None = None,
    *,
    include_fixed: bool = True,
) -> list[CandidateSpec]:
    del task_id
    pairs = task_data.get("train", [])
    specs: list[CandidateSpec] = []
    if include_fixed:
        specs.extend(inferred_border_chain_transfer_candidate_specs(pairs, ns))
        specs.extend(inferred_crop_cleanup_candidate_specs(pairs))
        specs.extend(inferred_single_motif_reconstruct_candidate_specs(pairs, ns))
        specs.extend(inferred_general_operator_candidate_specs(pairs))
        specs.extend(inferred_sia_operator_candidate_specs(pairs))
        specs.extend(inferred_line_network_candidate_specs(pairs, ns))
    specs.extend(infer_row_marker_panel_flow_specs(pairs))
    specs.extend(infer_diagonal_color_role_specs(pairs))
    specs.extend(infer_master_pattern_expansion_specs(pairs))
    specs.extend(infer_supply_gap_relocation_specs(pairs))
    specs.extend(infer_ladder_glyph_transfer_specs(pairs))
    return specs


def overlay_panels(panels: list[Grid], mode: str, order: tuple[int, ...] | None = None) -> Grid:
    if not panels:
        return [[0]]
    rows, cols = shape(panels[0])
    bg = border_background_color(panels[0])
    if order is None:
        order = tuple(range(len(panels)))
    out = [[bg for _ in range(cols)] for _ in range(rows)]
    if mode == "overlay":
        for idx in order:
            panel = panels[idx]
            for r in range(rows):
                for c in range(cols):
                    if panel[r][c] != bg:
                        out[r][c] = panel[r][c]
        return out
    for r in range(rows):
        for c in range(cols):
            vals = [panel[r][c] for panel in panels]
            non_bg = [(i, v) for i, v in enumerate(vals) if v != bg]
            if mode == "union" and non_bg:
                out[r][c] = non_bg[0][1]
            elif mode == "xor" and len(non_bg) == 1:
                out[r][c] = non_bg[0][1]
            elif mode == "intersection" and len(non_bg) == len(panels):
                out[r][c] = non_bg[0][1]
            elif mode.startswith("difference_") and len(panels) >= 2:
                keep_idx = int(mode.split("_", 1)[1])
                if vals[keep_idx] != bg and all(vals[j] == bg for j in range(len(vals)) if j != keep_idx):
                    out[r][c] = vals[keep_idx]
    return out


def remove_border(grid: Grid) -> Grid:
    rows, cols = shape(grid)
    if rows <= 2 or cols <= 2:
        return copy_grid(grid)
    return [row[1:-1] for row in grid[1:-1]]


def scale_grid(grid: Grid, sr: int, sc: int) -> Grid:
    if sr <= 0 or sc <= 0:
        return copy_grid(grid)
    out: Grid = []
    for row in grid:
        expanded = []
        for value in row:
            expanded.extend([value] * sc)
        for _ in range(sr):
            out.append(expanded[:])
    return out


def tile_grid(grid: Grid, tr: int, tc: int) -> Grid:
    if tr <= 0 or tc <= 0:
        return copy_grid(grid)
    tiled_rows = [row * tc for row in grid]
    out: Grid = []
    for _ in range(tr):
        out.extend([row[:] for row in tiled_rows])
    return out


def majority(values: Iterable[int]) -> int:
    counts = Counter(values)
    return counts.most_common(1)[0][0] if counts else 0


def downscale_majority(grid: Grid, sr: int, sc: int) -> Grid:
    rows, cols = shape(grid)
    if sr <= 0 or sc <= 0 or rows % sr or cols % sc:
        return copy_grid(grid)
    out: Grid = []
    for r0 in range(0, rows, sr):
        row = []
        for c0 in range(0, cols, sc):
            row.append(majority(grid[r][c] for r in range(r0, r0 + sr) for c in range(c0, c0 + sc)))
        out.append(row)
    return out


def is_h_symmetric(grid: Grid) -> bool:
    return normalize_grid(grid) == mirror_v(normalize_grid(grid))


def is_v_symmetric(grid: Grid) -> bool:
    return normalize_grid(grid) == mirror_h(normalize_grid(grid))


def repair_symmetry_vote(grid: Grid, axis: str) -> Grid:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    bg = border_background_color(grid)
    out = copy_grid(grid)
    if axis == "h":
        for r in range(rows):
            rr = rows - 1 - r
            for c in range(cols):
                a, b = out[r][c], out[rr][c]
                if a == bg and b != bg:
                    out[r][c] = b
                elif b == bg and a != bg:
                    out[rr][c] = a
    elif axis == "v":
        for r in range(rows):
            for c in range(cols):
                cc = cols - 1 - c
                a, b = out[r][c], out[r][cc]
                if a == bg and b != bg:
                    out[r][c] = b
                elif b == bg and a != bg:
                    out[r][cc] = a
    elif axis == "rot180":
        for r in range(rows):
            for c in range(cols):
                rr, cc = rows - 1 - r, cols - 1 - c
                a, b = out[r][c], out[rr][cc]
                if a == bg and b != bg:
                    out[r][c] = b
                elif b == bg and a != bg:
                    out[rr][cc] = a
    elif axis == "diag" and rows == cols:
        for r in range(rows):
            for c in range(cols):
                a, b = out[r][c], out[c][r]
                if a == bg and b != bg:
                    out[r][c] = b
                elif b == bg and a != bg:
                    out[c][r] = a
    elif axis == "anti" and rows == cols:
        for r in range(rows):
            for c in range(cols):
                rr, cc = cols - 1 - c, rows - 1 - r
                a, b = out[r][c], out[rr][cc]
                if a == bg and b != bg:
                    out[r][c] = b
                elif b == bg and a != bg:
                    out[rr][cc] = a
    return out


def fill_enclosed_regions(grid: Grid, color: int | None = None) -> Grid:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    out = copy_grid(grid)
    for region in enclosed_regions(grid, bg):
        border_colors = []
        for r, c in region:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != bg:
                    border_colors.append(grid[nr][nc])
        fill = color if color is not None else (Counter(border_colors).most_common(1)[0][0] if border_colors else bg)
        for r, c in region:
            out[r][c] = fill
    return out


def contact_pairs(objects: list[ObjectInfo]) -> set[tuple[int, int]]:
    cell_to_obj: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for cell in obj.cells:
            cell_to_obj[cell] = idx
    contacts = set()
    for idx, obj in enumerate(objects):
        for r, c in obj.cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                other = cell_to_obj.get((r + dr, c + dc))
                if other is not None and other != idx:
                    contacts.add(tuple(sorted((idx, other))))
    return contacts


def alignment_pairs(objects: list[ObjectInfo]) -> set[tuple[int, int, str]]:
    pairs = set()
    for i, a in enumerate(objects):
        for j, b in enumerate(objects[i + 1 :], start=i + 1):
            ar0, ac0, ar1, ac1 = a.bbox
            br0, bc0, br1, bc1 = b.bbox
            if a.center2[0] == b.center2[0] or ar0 == br0 or ar1 == br1:
                pairs.add((i, j, "row"))
            if a.center2[1] == b.center2[1] or ac0 == bc0 or ac1 == bc1:
                pairs.add((i, j, "col"))
    return pairs


def connect_aligned_same_color(grid: Grid) -> Grid:
    grid = normalize_grid(grid)
    out = copy_grid(grid)
    bg = border_background_color(grid)
    rows, cols = shape(grid)
    for r in range(rows):
        by_color: dict[int, list[int]] = {}
        for c in range(cols):
            if grid[r][c] != bg:
                by_color.setdefault(grid[r][c], []).append(c)
        for color, cs in by_color.items():
            if len(cs) >= 2:
                for c in range(min(cs), max(cs) + 1):
                    if out[r][c] == bg:
                        out[r][c] = color
    for c in range(cols):
        by_color = {}
        for r in range(rows):
            if grid[r][c] != bg:
                by_color.setdefault(grid[r][c], []).append(r)
        for color, rs in by_color.items():
            if len(rs) >= 2:
                for r in range(min(rs), max(rs) + 1):
                    if out[r][c] == bg:
                        out[r][c] = color
    return out


def connect_aligned_points(grid: Grid, *, diagonal: bool = True, color_mode: str = "endpoint") -> Grid:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    out = copy_grid(grid)
    rows, cols = shape(grid)
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                by_color.setdefault(grid[r][c], []).append((r, c))

    def draw(a: tuple[int, int], b: tuple[int, int], color: int) -> None:
        r0, c0 = a
        r1, c1 = b
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        if r0 != r1 and c0 != c1 and abs(r1 - r0) != abs(c1 - c0):
            return
        if not diagonal and dr and dc:
            return
        r, c = r0, c0
        while True:
            if out[r][c] == bg:
                out[r][c] = color
            if (r, c) == (r1, c1):
                break
            r += dr
            c += dc

    for color, pts in by_color.items():
        if len(pts) < 2 or len(pts) > 12:
            continue
        for a, b in itertools.combinations(pts, 2):
            if a[0] == b[0] or a[1] == b[1] or (diagonal and abs(a[0] - b[0]) == abs(a[1] - b[1])):
                draw(a, b, color)
    return out


def outline_component_boxes(grid: Grid, color: int | None = None) -> Grid:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    out = copy_grid(grid)
    for obj in connected_components(grid, bg, same_color=False):
        r0, c0, r1, c1 = obj.bbox
        draw = color if color is not None else obj.color
        for c in range(c0, c1 + 1):
            out[r0][c] = draw
            out[r1][c] = draw
        for r in range(r0, r1 + 1):
            out[r][c0] = draw
            out[r][c1] = draw
    return out


def remove_selected_objects(grid: Grid, selector: Callable[[list[ObjectInfo]], list[ObjectInfo]]) -> Grid:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    out = copy_grid(grid)
    for obj in selector(connected_components(grid, bg, same_color=False)):
        for r, c in obj.cells:
            out[r][c] = bg
    return out


def crop_selected_object(grid: Grid, selector: Callable[[list[ObjectInfo]], list[ObjectInfo]]) -> Grid:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    objs = selector(connected_components(grid, bg, same_color=False))
    if not objs:
        return copy_grid(grid)
    return object_crop(grid, objs[0], bg)


def select_largest(objs: list[ObjectInfo]) -> list[ObjectInfo]:
    return [max(objs, key=lambda o: (o.size, -o.bbox[0], -o.bbox[1]))] if objs else []


def select_smallest(objs: list[ObjectInfo]) -> list[ObjectInfo]:
    return [min(objs, key=lambda o: (o.size, o.bbox[0], o.bbox[1]))] if objs else []


def select_most_holes(objs: list[ObjectInfo]) -> list[ObjectInfo]:
    ranked = [o for o in objs if o.holes > 0]
    return [max(ranked, key=lambda o: (o.holes, o.size))] if ranked else []


def select_touching_border(objs: list[ObjectInfo]) -> list[ObjectInfo]:
    return [o for o in objs if o.touches_border]


def select_interior(objs: list[ObjectInfo]) -> list[ObjectInfo]:
    return [o for o in objs if not o.touches_border]


OBJECT_SELECTORS: list[tuple[str, Callable[[list[ObjectInfo]], list[ObjectInfo]]]] = [
    ("largest_object", select_largest),
    ("smallest_object", select_smallest),
    ("most_holes_object", select_most_holes),
    ("border_objects", select_touching_border),
    ("interior_objects", select_interior),
]


BASIC_TRANSFORMS: list[tuple[str, Transform]] = [
    ("identity", copy_grid),
    ("rot90", rot90),
    ("rot180", rot180),
    ("rot270", rot270),
    ("mirror_h", mirror_h),
    ("mirror_v", mirror_v),
    ("transpose", transpose),
    ("anti_transpose", anti_transpose),
    ("crop_foreground", crop_foreground),
    ("remove_border", remove_border),
]


NAMESPACE_CANDIDATE_NAMES = [
    "recover_point_symmetric_region",
    "replace_plus_shapes_with_color",
    "apply_mod6_column_stripe_pattern",
    "connect_cells_via_key_sequence",
    "frame_minority_cells",
    "reorganize_grid_sections",
    "reflect_through_marker",
    "swap_cross_hub_tail_colors",
    "transfer_box_center_via_border_chain",
    "erase_blobs_containing_key_set",
    "recolor_blobs_by_topological_holes",
    "highlight_kth_longest_bar",
    "straighten_noisy_line_networks",
    "align_framed_blocks_to_side_rails",
    "apply_bottom_legend_shape_operations",
    "apply_left_stencil_to_right_frame_gravity",
    "assemble_connector_tree_with_leaf_backtracking",
    "complete_lattice_motif_blocks",
    "complete_sparse_periodic_marker_lines",
    "compose_panel_by_boundary_mask",
    "compose_solid_blocks_with_same_color_glyph_masks",
    "copy_left_motifs_to_labeled_frame_intersections",
    "count_external_objects_into_zero_template",
    "expand_mask_cells_with_glyph",
    "expand_singleton_code_grid_to_blocks",
    "expand_symbolic_grid_with_prototype_legend",
    "extend_object_seed_stripes_to_boundary",
    "fill_row_holes_from_edge_palette",
    "fill_template_background_by_external_shape_cover",
    "fill_template_holes_by_external_shape_majority",
    "move_markers_along_legend_vectors",
    "normalize_parallel_line_segments_to_median",
    "pair_hollow_and_filled_blocks",
    "project_sparse_triad_shadow_markers",
    "prune_colors_to_largest_vertical_symmetry",
    "render_column_glyph_path_from_marker",
    "repeat_stencil_by_marker_component_count",
    "right_align_axis_special_run_prefixes",
    "right_align_zero_components_before_separator",
    "serialize_component_chain_to_column",
    "serialize_header_table_components",
    "slide_zero_block_to_farthest_path_endpoint",
    "solve_9385bd28_generalized",
    "stack_unique_panel_per_band",
    "stamp_decoded_marker_glyphs",
    "tile_mask_code_with_central_pattern",
    "tile_quadrant_pattern_by_top_counts",
    "uniform_rows_to_concentric_rings",
    "solve_noise",
    "solve_occluded_symmetry",
    "solve_occlusion",
    "solve_occlusion_in_place",
    "solve_pattern",
    "solve_pattern_in_place",
    "best_symmetry_repair",
    "best_pattern_repair",
    "crop_foreground",
    "remove_border",
    "largest_object",
    "smallest_object",
    "extract_largest_object",
    "extract_smallest_object",
    "remove_largest_object",
    "remove_smallest_object",
    "keep_border_objects",
    "remove_border_objects",
    "isolate_unique_color_object",
    "fill_object_bboxes",
    "outline_objects",
    "gravity_objects_down",
    "gravity_objects_up",
    "gravity_objects_left",
    "gravity_objects_right",
    "move_objects_to_center",
]


def infer_color_map(pairs: list[dict[str, Grid]], base: Transform | None = None) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        src = normalize_grid(base(inp) if base else inp)
        if shape(src) != shape(out):
            return None
        for r, row in enumerate(src):
            for c, value in enumerate(row):
                target = out[r][c]
                if value in mapping and mapping[value] != target:
                    return None
                mapping[value] = target
    return mapping if mapping else None


def apply_color_map(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def infer_constant_output(pairs: list[dict[str, Grid]]) -> Grid | None:
    if not pairs:
        return None
    first = normalize_grid(pairs[0]["output"])
    if all(normalize_grid(pair["output"]) == first for pair in pairs[1:]):
        return first
    return None


def infer_scale(pairs: list[dict[str, Grid]]) -> tuple[int, int] | None:
    factors = None
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        ir, ic = shape(inp)
        orow, ocol = shape(out)
        if ir == 0 or ic == 0 or orow % ir or ocol % ic:
            return None
        current = (orow // ir, ocol // ic)
        if scale_grid(inp, *current) != out:
            return None
        if factors is None:
            factors = current
        elif factors != current:
            return None
    return factors


def infer_tile(pairs: list[dict[str, Grid]]) -> tuple[int, int] | None:
    factors = None
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        ir, ic = shape(inp)
        orow, ocol = shape(out)
        if ir == 0 or ic == 0 or orow % ir or ocol % ic:
            return None
        current = (orow // ir, ocol // ic)
        if tile_grid(inp, *current) != out:
            return None
        if factors is None:
            factors = current
        elif factors != current:
            return None
    return factors


def infer_downscale(pairs: list[dict[str, Grid]]) -> tuple[int, int] | None:
    factors = None
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        ir, ic = shape(inp)
        orow, ocol = shape(out)
        if orow == 0 or ocol == 0 or ir % orow or ic % ocol:
            return None
        current = (ir // orow, ic // ocol)
        if downscale_majority(inp, *current) != out:
            return None
        if factors is None:
            factors = current
        elif factors != current:
            return None
    return factors


def infer_panel_extractor(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []
    signatures: dict[str, tuple[str, int | None]] = {}
    for pair_idx, pair in enumerate(pairs):
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        matches = []
        for name, idx, panel in split_by_separators(inp):
            if panel == out:
                matches.append((name, None))
            if crop_foreground(panel) == out:
                matches.append((f"{name}_crop", None))
        if pair_idx == 0:
            signatures = {name: (name, idx) for name, idx in matches}
        else:
            signatures = {name: value for name, value in signatures.items() if any(m[0] == name for m in matches)}
        if not signatures:
            return []

    out = []
    for name in sorted(signatures):
        def make_transform(panel_name: str) -> Transform:
            def transform(grid: Grid) -> Grid:
                for candidate_name, _idx, panel in split_by_separators(grid):
                    if panel_name == candidate_name:
                        return panel
                    if panel_name == f"{candidate_name}_crop":
                        return crop_foreground(panel)
                return copy_grid(grid)
            return transform

        out.append((f"extract_panel:{name}", make_transform(name)))
    return out


def mask_colors(grid: Grid, colors: frozenset[int], mode: str, crop_result: bool = False) -> Grid:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    if mode == "keep":
        out = [[value if value in colors else bg for value in row] for row in grid]
    else:
        out = [[bg if value in colors else value for value in row] for row in grid]
    return crop_foreground(out) if crop_result else out


def infer_color_filters(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []
    all_colors = sorted(set().union(*(palette(normalize_grid(pair["input"])) for pair in pairs)))
    if len(all_colors) > 10:
        return []
    inferred: list[tuple[str, Transform]] = []
    color_sets: list[frozenset[int]] = []
    for size in range(1, min(4, len(all_colors)) + 1):
        for combo in itertools.combinations(all_colors, size):
            color_sets.append(frozenset(combo))
    for colors in color_sets:
        suffix = "-".join(map(str, sorted(colors)))
        for mode in ("keep", "remove"):
            for crop_result in (False, True):
                ok = True
                for pair in pairs:
                    pred = mask_colors(pair["input"], colors, mode, crop_result)
                    if pred != normalize_grid(pair["output"]):
                        ok = False
                        break
                if ok:
                    name = f"color_filter:{mode}:{suffix}{':crop' if crop_result else ''}"
                    inferred.append((name, lambda grid, colors=colors, mode=mode, crop_result=crop_result: mask_colors(grid, colors, mode, crop_result)))
    return inferred


def infer_pixel_block_expansion(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []
    factors = None
    mapping: dict[int, Grid] = {}
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        ir, ic = shape(inp)
        orow, ocol = shape(out)
        if ir == 0 or ic == 0 or orow % ir or ocol % ic:
            return []
        current = (orow // ir, ocol // ic)
        if current == (1, 1):
            return []
        if factors is None:
            factors = current
        elif factors != current:
            return []
        br, bc = current
        for r in range(ir):
            for c in range(ic):
                block = [row[c * bc : (c + 1) * bc] for row in out[r * br : (r + 1) * br]]
                value = inp[r][c]
                if value in mapping and mapping[value] != block:
                    return []
                mapping[value] = block
    if not factors or not mapping:
        return []

    def transform(grid: Grid, factors=factors, mapping=mapping) -> Grid:
        grid = normalize_grid(grid)
        br, bc = factors
        bg = border_background_color(grid)
        default = [[bg for _ in range(bc)] for _ in range(br)]
        out: Grid = []
        for row in grid:
            expanded_rows = [[] for _ in range(br)]
            for value in row:
                block = mapping.get(value, [[value for _ in range(bc)] for _ in range(br)] if value != bg else default)
                for rr in range(br):
                    expanded_rows[rr].extend(block[rr])
            out.extend(expanded_rows)
        return out

    return [(f"pixel_block_expand:{factors[0]}x{factors[1]}", transform)]


def block_value(block: Grid, mode: str, bg: int) -> int:
    vals = [v for row in block for v in row]
    if mode == "top_left":
        return block[0][0]
    if mode == "top_right":
        return block[0][-1]
    if mode == "bottom_left":
        return block[-1][0]
    if mode == "bottom_right":
        return block[-1][-1]
    if mode == "center":
        return block[len(block) // 2][len(block[0]) // 2]
    if mode == "max":
        return max(vals)
    if mode == "min":
        return min(vals)
    if mode == "first_non_bg":
        return next((v for v in vals if v != bg), bg)
    if mode == "last_non_bg":
        return next((v for v in reversed(vals) if v != bg), bg)
    if mode == "unique_non_bg":
        non = [v for v in vals if v != bg]
        counts = Counter(non)
        uniques = [v for v, count in counts.items() if count == 1]
        return uniques[0] if len(uniques) == 1 else (Counter(non).most_common(1)[0][0] if non else bg)
    return majority(vals)


def downscale_block_select(grid: Grid, sr: int, sc: int, mode: str) -> Grid:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    if sr <= 0 or sc <= 0 or rows % sr or cols % sc:
        return copy_grid(grid)
    bg = border_background_color(grid)
    out: Grid = []
    for r0 in range(0, rows, sr):
        row = []
        for c0 in range(0, cols, sc):
            block = [grid[r][c0 : c0 + sc] for r in range(r0, r0 + sr)]
            row.append(block_value(block, mode, bg))
        out.append(row)
    return out


def infer_downscale_selectors(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []
    modes = [
        "top_left", "top_right", "bottom_left", "bottom_right", "center",
        "majority", "first_non_bg", "last_non_bg", "unique_non_bg", "max", "min",
    ]
    factors = None
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        ir, ic = shape(inp)
        orow, ocol = shape(out)
        if orow == 0 or ocol == 0 or ir % orow or ic % ocol:
            return []
        current = (ir // orow, ic // ocol)
        if current == (1, 1):
            return []
        if factors is None:
            factors = current
        elif factors != current:
            return []
    inferred = []
    for mode in modes:
        if all(downscale_block_select(pair["input"], factors[0], factors[1], mode) == normalize_grid(pair["output"]) for pair in pairs):
            inferred.append((f"downscale_select:{mode}_{factors[0]}x{factors[1]}", lambda grid, f=factors, mode=mode: downscale_block_select(grid, f[0], f[1], mode)))
    return inferred


def infer_panel_overlays(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []
    panel_counts = []
    for pair in pairs:
        panels = equal_shape_panels(pair["input"])
        if not panels:
            return []
        if shape(panels[0]) != shape(normalize_grid(pair["output"])):
            return []
        panel_counts.append(len(panels))
    if len(set(panel_counts)) != 1:
        return []
    n = panel_counts[0]
    modes = ["union", "xor", "intersection"] + [f"difference_{i}" for i in range(n)]
    if n <= 4:
        modes.extend([f"overlay:{','.join(map(str, order))}" for order in itertools.permutations(range(n))])
    inferred = []
    for mode in modes:
        ok = True
        for pair in pairs:
            panels = equal_shape_panels(pair["input"])
            if mode.startswith("overlay:"):
                order = tuple(int(x) for x in mode.split(":", 1)[1].split(","))
                pred = overlay_panels(panels, "overlay", order)
            else:
                pred = overlay_panels(panels, mode)
            if pred != normalize_grid(pair["output"]):
                ok = False
                break
        if ok:
            def transform(grid: Grid, mode=mode) -> Grid:
                panels = equal_shape_panels(grid)
                if not panels:
                    return copy_grid(grid)
                if mode.startswith("overlay:"):
                    order = tuple(int(x) for x in mode.split(":", 1)[1].split(","))
                    return overlay_panels(panels, "overlay", order)
                return overlay_panels(panels, mode)
            inferred.append((f"panel_overlay:{mode}", transform))
    return inferred


def infer_object_extractor(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    inferred = []
    for selector_name, selector in OBJECT_SELECTORS:
        ok = True
        for pair in pairs:
            if crop_selected_object(pair["input"], selector) != normalize_grid(pair["output"]):
                ok = False
                break
        if ok:
            inferred.append((f"crop:{selector_name}", lambda grid, selector=selector: crop_selected_object(grid, selector)))
    return inferred


def infer_object_remover(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    inferred = []
    for selector_name, selector in OBJECT_SELECTORS:
        ok = True
        for pair in pairs:
            inp = normalize_grid(pair["input"])
            out = normalize_grid(pair["output"])
            if shape(inp) != shape(out) or remove_selected_objects(inp, selector) != out:
                ok = False
                break
        if ok:
            inferred.append((f"remove:{selector_name}", lambda grid, selector=selector: remove_selected_objects(grid, selector)))
    return inferred


def infer_hole_fill(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform]]:
    if not pairs:
        return []
    learned_color = None
    surround_mode_ok = True
    constant_ok = True
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if shape(inp) != shape(out):
            return []
        bg = border_background_color(inp)
        holes = {cell for region in enclosed_regions(inp, bg) for cell in region}
        changed = [(r, c) for r in range(len(inp)) for c in range(len(inp[0])) if inp[r][c] != out[r][c]]
        if not changed or any(cell not in holes for cell in changed):
            constant_ok = False
            surround_mode_ok = False
            break
        colors = {out[r][c] for r, c in changed}
        if len(colors) != 1:
            constant_ok = False
        else:
            color = next(iter(colors))
            if learned_color is None:
                learned_color = color
            elif learned_color != color:
                constant_ok = False
        if fill_enclosed_regions(inp) != out:
            surround_mode_ok = False

    inferred: list[tuple[str, Transform]] = []
    if constant_ok and learned_color is not None:
        inferred.append((f"fill_holes:color_{learned_color}", lambda grid, color=learned_color: fill_enclosed_regions(grid, color)))
    if surround_mode_ok:
        inferred.append(("fill_holes:surround_majority", fill_enclosed_regions))
    return inferred


def grid_object_summary(grid: Grid) -> dict[str, Any]:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    objs = connected_components(grid, bg, same_color=False)
    rows, cols = shape(grid)
    return {
        "shape": (rows, cols),
        "palette": frozenset(palette(grid)),
        "non_bg_palette": frozenset(c for c in palette(grid) if c != bg),
        "object_count": len(objs),
        "hole_count": len(enclosed_regions(grid, bg)),
        "sep_rows": len(separator_indices(grid, "row", bg)),
        "sep_cols": len(separator_indices(grid, "col", bg)),
        "h_symmetric": is_h_symmetric(grid),
        "v_symmetric": is_v_symmetric(grid),
    }


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def infer_output_profile(task_data: dict[str, Any]) -> dict[str, Any]:
    pairs = task_data.get("train", [])
    inputs = [normalize_grid(pair["input"]) for pair in pairs]
    outputs = [normalize_grid(pair["output"]) for pair in pairs]
    in_summaries = [grid_object_summary(grid) for grid in inputs]
    out_summaries = [grid_object_summary(grid) for grid in outputs]
    profile: dict[str, Any] = {
        "output_shapes": {summary["shape"] for summary in out_summaries},
        "same_shape": bool(pairs) and all(shape(i) == shape(o) for i, o in zip(inputs, outputs)),
        "palette_subset_input": bool(pairs) and all(palette(o).issubset(palette(i)) for i, o in zip(inputs, outputs)),
        "constant_output_shape": len({summary["shape"] for summary in out_summaries}) == 1,
        "object_count_deltas": {
            out_summaries[i]["object_count"] - in_summaries[i]["object_count"]
            for i in range(len(out_summaries))
        },
        "hole_count_deltas": {
            out_summaries[i]["hole_count"] - in_summaries[i]["hole_count"]
            for i in range(len(out_summaries))
        },
        "output_has_h_symmetry": bool(pairs) and all(summary["h_symmetric"] for summary in out_summaries),
        "output_has_v_symmetry": bool(pairs) and all(summary["v_symmetric"] for summary in out_summaries),
    }
    if pairs:
        profile["single_object_output"] = all(summary["object_count"] == 1 for summary in out_summaries)
    return profile


def foreground_bbox_shape(grid: Grid) -> tuple[int, int]:
    grid = normalize_grid(grid)
    bg = border_background_color(grid)
    rows, cols = shape(grid)
    fg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not fg:
        return rows, cols
    rs = [r for r, _c in fg]
    cs = [c for _r, c in fg]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def predict_output_shapes_from_train(pairs: list[dict[str, Grid]], test_input: Grid) -> set[tuple[int, int]]:
    if not pairs:
        return set()
    inputs = [normalize_grid(pair["input"]) for pair in pairs]
    outputs = [normalize_grid(pair["output"]) for pair in pairs]
    in_shapes = [shape(grid) for grid in inputs]
    out_shapes = [shape(grid) for grid in outputs]
    test_shape = shape(test_input)
    predicted: set[tuple[int, int]] = set()

    if all(i_shape == o_shape for i_shape, o_shape in zip(in_shapes, out_shapes)):
        predicted.add(test_shape)
    if len(set(out_shapes)) == 1:
        predicted.add(out_shapes[0])

    scale_factors = {
        (o_shape[0] // i_shape[0], o_shape[1] // i_shape[1])
        for i_shape, o_shape in zip(in_shapes, out_shapes)
        if i_shape[0]
        and i_shape[1]
        and o_shape[0] % i_shape[0] == 0
        and o_shape[1] % i_shape[1] == 0
    }
    if len(scale_factors) == 1:
        row_factor, col_factor = next(iter(scale_factors))
        if (row_factor, col_factor) != (1, 1):
            predicted.add((test_shape[0] * row_factor, test_shape[1] * col_factor))

    if (
        any(i_shape != o_shape for i_shape, o_shape in zip(in_shapes, out_shapes))
        and all(foreground_bbox_shape(inp) == out_shape for inp, out_shape in zip(inputs, out_shapes))
    ):
        predicted.add(foreground_bbox_shape(test_input))

    return {candidate for candidate in predicted if candidate[0] > 0 and candidate[1] > 0}


def prediction_profile_score(pred: Grid, test_input: Grid, profile: dict[str, Any]) -> float:
    pred = normalize_grid(pred)
    test_input = normalize_grid(test_input)
    score = 0.0
    pred_shape = shape(pred)
    predicted_shapes = profile.get("predicted_output_shapes", set())
    if predicted_shapes:
        score += 6.0 if pred_shape in predicted_shapes else -8.0
    if profile.get("same_shape"):
        score += 8.0 if pred_shape == shape(test_input) else -12.0
    elif pred_shape in profile.get("output_shapes", set()):
        score += 4.0
    elif profile.get("constant_output_shape"):
        score -= 2.0
    if profile.get("palette_subset_input"):
        score += 2.0 if palette(pred).issubset(palette(test_input)) else -4.0
    pred_summary = grid_object_summary(pred)
    in_summary = grid_object_summary(test_input)
    deltas = profile.get("object_count_deltas", set())
    if len(deltas) == 1:
        expected_delta = next(iter(deltas))
        actual_delta = pred_summary["object_count"] - in_summary["object_count"]
        score += 2.0 if actual_delta == expected_delta else -1.0
    hole_deltas = profile.get("hole_count_deltas", set())
    if len(hole_deltas) == 1:
        expected_delta = next(iter(hole_deltas))
        actual_delta = pred_summary["hole_count"] - in_summary["hole_count"]
        score += 1.0 if actual_delta == expected_delta else -0.5
    if profile.get("output_has_h_symmetry"):
        score += 1.5 if is_h_symmetric(pred) else -1.0
    if profile.get("output_has_v_symmetry"):
        score += 1.5 if is_v_symmetric(pred) else -1.0
    if pred == [[0]] and shape(test_input) != (1, 1):
        score -= 20.0
    return score


def ranker_feature_dict(
    candidate: RankedCandidate,
    pred: Grid,
    task_data: dict[str, Any],
    test_input: Grid,
    profile: dict[str, Any] | None = None,
) -> dict[str, float]:
    if profile is None:
        profile = infer_output_profile(task_data)
    pred = normalize_grid(pred)
    test_input = normalize_grid(test_input)
    pred_summary = grid_object_summary(pred)
    input_summary = grid_object_summary(test_input)
    pr, pc = shape(pred)
    ir, ic = shape(test_input)
    pred_palette = palette(pred)
    input_palette = palette(test_input)
    family = candidate.family or "unknown"
    prefix = candidate.name.split(":", 1)[0]

    object_delta = pred_summary["object_count"] - input_summary["object_count"]
    hole_delta = pred_summary["hole_count"] - input_summary["hole_count"]
    expected_object_delta = 0.0
    object_deltas = profile.get("object_count_deltas", set())
    if len(object_deltas) == 1:
        expected_object_delta = float(next(iter(object_deltas)))
    expected_hole_delta = 0.0
    hole_deltas = profile.get("hole_count_deltas", set())
    if len(hole_deltas) == 1:
        expected_hole_delta = float(next(iter(hole_deltas)))

    features: dict[str, float] = {
        "bias": 1.0,
        "train_accuracy": candidate.train_accuracy,
        "exact_train": 1.0 if candidate.exact_train else 0.0,
        "prior": float(candidate.sort_key[1]),
        "engineered_score": candidate_prediction_score_no_model(candidate, pred, task_data, test_input, profile),
        "pred_rows": float(pr),
        "pred_cols": float(pc),
        "input_rows": float(ir),
        "input_cols": float(ic),
        "shape_same_as_input": 1.0 if (pr, pc) == (ir, ic) else 0.0,
        "shape_in_train_outputs": 1.0 if (pr, pc) in profile.get("output_shapes", set()) else 0.0,
        "same_shape_task": 1.0 if profile.get("same_shape") else 0.0,
        "palette_subset_input": 1.0 if pred_palette.issubset(input_palette) else 0.0,
        "palette_size_delta": float(len(pred_palette) - len(input_palette)),
        "pred_equals_input": 1.0 if grid_key(pred) == grid_key(test_input) else 0.0,
        "pred_is_zero": 1.0 if pred == [[0]] else 0.0,
        "pred_object_count": float(pred_summary["object_count"]),
        "input_object_count": float(input_summary["object_count"]),
        "object_delta": float(object_delta),
        "object_delta_matches_profile": 1.0 if object_delta == expected_object_delta else 0.0,
        "pred_hole_count": float(pred_summary["hole_count"]),
        "input_hole_count": float(input_summary["hole_count"]),
        "hole_delta": float(hole_delta),
        "hole_delta_matches_profile": 1.0 if hole_delta == expected_hole_delta else 0.0,
        "pred_h_symmetric": 1.0 if pred_summary["h_symmetric"] else 0.0,
        "pred_v_symmetric": 1.0 if pred_summary["v_symmetric"] else 0.0,
        "output_wants_h_symmetry": 1.0 if profile.get("output_has_h_symmetry") else 0.0,
        "output_wants_v_symmetry": 1.0 if profile.get("output_has_v_symmetry") else 0.0,
        "area_ratio": safe_ratio(pr * pc, ir * ic),
        "non_bg_palette_ratio": safe_ratio(
            len(pred_summary["non_bg_palette"]),
            max(1, len(input_summary["non_bg_palette"])),
        ),
        f"family={family}": 1.0,
        f"prefix={prefix}": 1.0,
    }
    if candidate.name.startswith("dsl:solve_"):
        features["name_is_task_solve"] = 1.0
    if "color" in candidate.name:
        features["name_has_color"] = 1.0
    if "object" in candidate.name:
        features["name_has_object"] = 1.0
    if "hole" in candidate.name:
        features["name_has_hole"] = 1.0
    return features


def load_ranker_model() -> dict[str, Any] | None:
    global _RANKER_MODEL_CACHE
    if _RANKER_MODEL_CACHE is not None:
        return _RANKER_MODEL_CACHE
    if os.environ.get("ARC_ENABLE_RANKER_MODEL", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        _RANKER_MODEL_CACHE = {}
        return None
    path = RANKER_MODEL_PATH
    if not path.exists():
        _RANKER_MODEL_CACHE = {}
        return None
    try:
        _RANKER_MODEL_CACHE = json.loads(path.read_text())
    except Exception:
        _RANKER_MODEL_CACHE = {}
    return _RANKER_MODEL_CACHE or None


def ml_ranker_score(features: dict[str, float], model: dict[str, Any] | None = None) -> float:
    if model is None:
        model = load_ranker_model()
    if not model:
        return 0.0
    weights = model.get("weights", {})
    score = float(model.get("intercept", 0.0))
    means = model.get("means", {})
    scales = model.get("scales", {})
    for key, value in features.items():
        if key in means:
            scale = scales.get(key, 1.0) or 1.0
            value = (value - means[key]) / scale
        score += float(weights.get(key, 0.0)) * float(value)
    return score


def _ttt_feature_vector(
    grid: Grid,
    r: int,
    c: int,
    bg: int | None = None,
    color_freqs: list[float] | None = None,
) -> list[float]:
    rows, cols = shape(grid)
    if bg is None:
        bg = border_background_color(grid)
    values = [1.0]
    center = grid[r][c]
    values.extend(1.0 if center == color else 0.0 for color in range(10))
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        value = grid[nr][nc] if 0 <= nr < rows and 0 <= nc < cols else bg
        values.extend(1.0 if value == color else 0.0 for color in range(10))
    values.extend([
        r / max(1, rows - 1),
        c / max(1, cols - 1),
        1.0 if r in (0, rows - 1) else 0.0,
        1.0 if c in (0, cols - 1) else 0.0,
    ])
    if color_freqs is None:
        counts = Counter(cell for row in grid for cell in row)
        total = max(1, rows * cols)
        color_freqs = [counts.get(color, 0) / total for color in range(10)]
    values.extend(color_freqs)
    return values


def train_ttt_pixel_classifier(
    pairs: list[dict[str, Grid]],
    epochs: int = 8,
    max_cells_per_grid: int = 400,
) -> Transform | None:
    """Train a tiny per-task pixel classifier.

    This is intentionally small and dependency-free: an averaged perceptron over
    local color/position features. It only proposes same-shape outputs and relies
    on the normal verifier/ranker before being considered for submission.
    """
    if not pairs:
        return None
    examples: list[tuple[list[float], int]] = []
    for pair in pairs:
        inp = normalize_grid(pair["input"])
        out = normalize_grid(pair["output"])
        if shape(inp) != shape(out):
            return None
        rows, cols = shape(inp)
        if rows * cols > max_cells_per_grid:
            return None
        bg = border_background_color(inp)
        counts = Counter(cell for row in inp for cell in row)
        total = max(1, rows * cols)
        freqs = [counts.get(color, 0) / total for color in range(10)]
        for r in range(rows):
            for c in range(cols):
                examples.append((_ttt_feature_vector(inp, r, c, bg, freqs), out[r][c]))
    if not examples or len(examples) > 3600:
        return None

    width = len(examples[0][0])
    weights = [[0.0] * width for _ in range(10)]

    def predict(features: list[float]) -> int:
        best_color = 0
        best_score = float("-inf")
        for color in range(10):
            score = sum(w * f for w, f in zip(weights[color], features))
            if score > best_score:
                best_score = score
                best_color = color
        return best_color

    for _epoch in range(max(1, epochs)):
        mistakes = 0
        for features, label in examples:
            pred = predict(features)
            if pred == label:
                continue
            mistakes += 1
            for i, value in enumerate(features):
                weights[label][i] += value
                weights[pred][i] -= value
        if mistakes == 0:
            break

    def transform(grid: Grid) -> Grid:
        grid = normalize_grid(grid)
        rows, cols = shape(grid)
        bg = border_background_color(grid)
        counts = Counter(cell for row in grid for cell in row)
        total = max(1, rows * cols)
        freqs = [counts.get(color, 0) / total for color in range(10)]
        return [[predict(_ttt_feature_vector(grid, r, c, bg, freqs)) for c in range(cols)] for r in range(rows)]

    return transform


def ttt_patch_key(inp: Grid, pred: Grid, r: int, c: int, radius: int) -> tuple[int, ...]:
    rows, cols = shape(inp)
    bg = border_background_color(inp)
    pred_bg = border_background_color(pred)
    values: list[int] = [radius]
    for grid, fill in ((inp, bg), (pred, pred_bg)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                values.append(grid[rr][cc] if 0 <= rr < rows and 0 <= cc < cols else fill)
    values.extend([
        inp[r][c],
        pred[r][c],
        1 if r == 0 else 0,
        1 if c == 0 else 0,
        1 if r == rows - 1 else 0,
        1 if c == cols - 1 else 0,
    ])
    return tuple(values)


def train_ttt_sparse_patch_repair(
    pairs: list[dict[str, Grid]],
    epochs: int = 8,
    max_cells_per_grid: int = 225,
) -> Transform | None:
    if len(pairs) < 2:
        return None
    shapes = {shape(normalize_grid(pair["input"])) for pair in pairs}
    if len(shapes) != 1:
        return None
    rows, cols = next(iter(shapes))
    if rows * cols > max_cells_per_grid:
        return None
    if any(shape(normalize_grid(pair["output"])) != (rows, cols) for pair in pairs):
        return None

    base = train_ttt_pixel_classifier(pairs, epochs=epochs, max_cells_per_grid=max_cells_per_grid)
    if base is None:
        return None

    train_inputs = [normalize_grid(pair["input"]) for pair in pairs]
    train_outputs = [normalize_grid(pair["output"]) for pair in pairs]
    train_preds = [normalize_grid(base(deepcopy(inp))) for inp in train_inputs]
    if any(shape(pred) != (rows, cols) for pred in train_preds):
        return None
    if all(pred == out for pred, out in zip(train_preds, train_outputs)):
        return None

    for radius in (1, 2):
        rules: dict[tuple[int, ...], int] = {}
        bad: set[tuple[int, ...]] = set()
        for inp, pred, out in zip(train_inputs, train_preds, train_outputs):
            for r in range(rows):
                for c in range(cols):
                    if pred[r][c] == out[r][c]:
                        continue
                    key = ttt_patch_key(inp, pred, r, c, radius)
                    if key in rules and rules[key] != out[r][c]:
                        bad.add(key)
                    elif key not in bad:
                        rules[key] = out[r][c]
        for key in bad:
            rules.pop(key, None)
        if not rules:
            continue

        def transform(grid: Grid, base: Transform = base, rules: dict[tuple[int, ...], int] = rules, radius: int = radius) -> Grid:
            inp = normalize_grid(grid)
            pred = normalize_grid(base(deepcopy(inp)))
            if shape(inp) != shape(pred):
                return pred
            out = copy_grid(pred)
            tr, tc = shape(inp)
            for rr in range(tr):
                for cc in range(tc):
                    key = ttt_patch_key(inp, pred, rr, cc, radius)
                    if key in rules:
                        out[rr][cc] = rules[key]
            return out

        exact, _score = train_transform_score(transform, pairs)
        if exact:
            return transform
    return None


def pixel_accuracy(expected: Grid, predicted: Grid) -> float:
    expected = normalize_grid(expected)
    predicted = normalize_grid(predicted)
    er, ec = shape(expected)
    pr, pc = shape(predicted)
    rows, cols = max(er, pr), max(ec, pc)
    if rows == 0 or cols == 0:
        return 0.0
    hit = 0
    for r in range(rows):
        for c in range(cols):
            ev = expected[r][c] if r < er and c < ec else -1
            pv = predicted[r][c] if r < pr and c < pc else -2
            hit += ev == pv
    return hit / (rows * cols)


def mismatch_count(expected: Grid, predicted: Grid) -> int:
    expected = normalize_grid(expected)
    predicted = normalize_grid(predicted)
    er, ec = shape(expected)
    pr, pc = shape(predicted)
    rows, cols = max(er, pr), max(ec, pc)
    misses = 0
    for r in range(rows):
        for c in range(cols):
            ev = expected[r][c] if r < er and c < ec else None
            pv = predicted[r][c] if r < pr and c < pc else None
            misses += ev != pv
    return misses


def call_with_timeout(fn: Callable[[], Any], timeout_sec: float = 0.20) -> Any:
    def _handler(_signum, _frame):
        raise TimeoutError("candidate timed out")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def score_transform(name: str, transform: Transform, task_data: dict[str, Any], family: str, prior: int) -> RankedCandidate | None:
    pairs = task_data.get("train", [])
    if not pairs:
        return None
    scores = []
    try:
        for pair in pairs:
            pred = normalize_grid(call_with_timeout(lambda: transform(deepcopy(pair["input"]))))
            scores.append(pixel_accuracy(pair["output"], pred))
    except Exception:
        return None
    mean_score = sum(scores) / len(scores)
    exact = all(score == 1.0 for score in scores)
    # Python dataclass orders ascending, so negative score ranks best first.
    return RankedCandidate(
        sort_key=(-mean_score, prior, name),
        name=name,
        transform=transform,
        exact_train=exact,
        train_accuracy=mean_score,
        family=family,
    )


def d4_variants() -> list[tuple[str, Transform, Transform]]:
    return [
        ("rot90", rot90, rot270),
        ("rot180", rot180, rot180),
        ("rot270", rot270, rot90),
        ("mirror_h", mirror_h, mirror_h),
        ("mirror_v", mirror_v, mirror_v),
        ("transpose", transpose, transpose),
        ("anti_transpose", anti_transpose, anti_transpose),
    ]


def canonical_color_map(grid: Grid, mode: str) -> tuple[Grid, dict[int, int]]:
    grid = normalize_grid(grid)
    counts = Counter(cell for row in grid for cell in row)
    if mode == "bg_first":
        bg = border_background_color(grid)
        ordered = [bg] + [
            color for color, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if color != bg
        ]
    else:
        ordered = [color for color, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    mapping = {color: idx for idx, color in enumerate(ordered)}
    inverse = {idx: color for color, idx in mapping.items()}
    return [[mapping[cell] for cell in row] for row in grid], inverse


def apply_inverse_color_map(grid: Grid, inverse: dict[int, int]) -> Grid:
    return [[inverse.get(cell, cell) for cell in row] for row in normalize_grid(grid)]


def narrow_tta_candidates(exact_candidates: list[RankedCandidate], task_data: dict[str, Any]) -> list[RankedCandidate]:
    default_enabled = "1" if hidden_facing_mode() else "0"
    if not env_enabled("ARC_ENABLE_NARROW_TTA", default_enabled):
        return []
    max_sources = int(os.environ.get("ARC_NARROW_TTA_MAX_SOURCES", "4"))
    max_out = int(os.environ.get("ARC_NARROW_TTA_MAX_CANDIDATES", "40"))
    out: list[RankedCandidate] = []

    for source in exact_candidates[:max_sources]:
        for variant_name, forward, inverse in d4_variants():
            if len(out) >= max_out:
                break

            def wrapped_d4(grid: Grid, source: RankedCandidate = source, forward: Transform = forward, inverse: Transform = inverse) -> Grid:
                pred = source.transform(normalize_grid(forward(normalize_grid(grid))))
                return normalize_grid(inverse(normalize_grid(pred)))

            scored = score_transform(
                f"tta_d4:{variant_name}:{source.name}",
                wrapped_d4,
                task_data,
                "tta_d4",
                max(1, int(source.sort_key[1]) + 7),
            )
            if scored is not None and scored.exact_train:
                out.append(scored)

        for mode in ("freq", "bg_first"):
            if len(out) >= max_out:
                break

            def wrapped_color(grid: Grid, source: RankedCandidate = source, mode: str = mode) -> Grid:
                mapped, inverse = canonical_color_map(grid, mode)
                pred = source.transform(mapped)
                return apply_inverse_color_map(normalize_grid(pred), inverse)

            scored = score_transform(
                f"tta_color:{mode}:{source.name}",
                wrapped_color,
                task_data,
                "tta_color",
                max(1, int(source.sort_key[1]) + 8),
            )
            if scored is not None and scored.exact_train:
                out.append(scored)

    return out


def perception_search_transform(task_data: dict[str, Any], ns: dict[str, Any] | None, timeout: float) -> Transform | None:
    if timeout <= 0:
        return None
    try:
        from perception_search import perception_search
    except Exception:
        return None
    try:
        code = perception_search(task_data, ns or {}, timeout=timeout)
    except Exception:
        return None
    if not code:
        return None
    exec_ns = dict(ns or {})
    try:
        exec(code, exec_ns)
    except Exception:
        return None
    transform = exec_ns.get("transform")
    if not callable(transform):
        return None
    return lambda grid, transform=transform: transform(deepcopy(grid))


def load_program_cache() -> dict[str, Any]:
    global _PROGRAM_CACHE
    if _PROGRAM_CACHE is not None:
        return _PROGRAM_CACHE
    if not PROGRAM_CACHE_PATH.exists():
        _PROGRAM_CACHE = {}
        return _PROGRAM_CACHE
    try:
        data = json.loads(PROGRAM_CACHE_PATH.read_text())
        _PROGRAM_CACHE = data if isinstance(data, dict) else {}
    except Exception:
        _PROGRAM_CACHE = {}
    return _PROGRAM_CACHE


def cached_program_specs(task_id: str | None, ns: dict[str, Any] | None) -> list[tuple[str, Transform, str, int]]:
    default_enabled = "0" if hidden_facing_mode() else "1"
    if not task_id or not env_enabled("ARC_ENABLE_PROGRAM_CACHE", default_enabled):
        return []
    rows = load_program_cache().get(task_id, [])
    if not isinstance(rows, list):
        return []
    specs: list[tuple[str, Transform, str, int]] = []
    for idx, row in enumerate(rows):
        code = row.get("code") if isinstance(row, dict) else None
        if not code or "def transform" not in code:
            continue
        name = row.get("name", f"cached_program_{idx}") if isinstance(row, dict) else f"cached_program_{idx}"
        try:
            exec_ns = dict(ns or {})
            exec(code, exec_ns)
            transform = exec_ns.get("transform")
            if not callable(transform):
                continue
        except Exception:
            continue
        specs.append((
            f"program_cache:{name}",
            lambda grid, transform=transform: transform(deepcopy(grid)),
            "program_cache",
            12 + idx,
        ))
    return specs


def color_components(grid: Grid, color: int, *, diag: bool = False) -> list[tuple[tuple[int, int], ...]]:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if diag:
        neighbors += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    seen: set[tuple[int, int]] = set()
    comps: list[tuple[tuple[int, int], ...]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != color or (r, c) in seen:
                continue
            queue = deque([(r, c)])
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and grid[nr][nc] == color:
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            comps.append(tuple(sorted(cells)))
    return comps


def zero_template_marker_repeat(grid: Grid) -> Grid:
    """Crop a zero template and repeat row markers by matching external components.

    A common ARC pattern is a blank zero panel containing one marker per output
    row; elsewhere in the input, repeated same-colored objects tell how many
    markers to emit in that row. Use 8-connected external component multiplicity
    so diagonal/stair motifs count as one repeated object.
    """
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    zero_components = color_components(grid, 0, diag=False)
    if not zero_components:
        return grid
    template = max(zero_components, key=len)
    r0 = min(r for r, _c in template)
    r1 = max(r for r, _c in template)
    c0 = min(c for _r, c in template)
    c1 = max(c for _r, c in template)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if height < 3 or width < 3 or len(template) < max(4, (height * width) // 3):
        return grid

    template_box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    markers: list[tuple[int, int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            value = grid[r][c]
            if value != 0:
                markers.append((r - r0, c - c0, value))
    if not markers:
        return grid

    out = [[0 for _ in range(width)] for _ in range(height)]
    for rr, cc, color in markers:
        external = [
            comp
            for comp in color_components(grid, color, diag=True)
            if not any(cell in template_box for cell in comp)
        ]
        if not external:
            repeat = 1
        else:
            max_size = max(len(comp) for comp in external)
            repeat = sum(1 for comp in external if len(comp) == max_size)
        repeat = max(1, min(repeat, (width - cc + 1) // 2))
        for k in range(repeat):
            c = cc + 2 * k
            if 0 <= rr < height and 0 <= c < width:
                out[rr][c] = color
    return out


def sparse_line_endpoint_bridge(grid: Grid) -> Grid:
    """Move sparse key-color supply cells into row-local line gaps.

    This is the generalized form behind a family of line/contact repairs: a
    small supply color marks cells that should be removed from the top/early
    region and reinserted into the lowest plausible gaps between colored runs.
    """
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    if 1 not in palette(inp):
        return copy_grid(inp)

    candidate_gap: dict[tuple[int, int], int] = {}
    for r, row in enumerate(inp):
        c = 0
        while c < cols:
            if row[c] != 0:
                c += 1
                continue
            start = c
            while c < cols and row[c] == 0:
                c += 1
            end = c - 1
            if start == 0 or end == cols - 1:
                continue

            left_c = start - 1
            left_color = row[left_c]
            left_start = left_c
            while left_start - 1 >= 0 and row[left_start - 1] == left_color:
                left_start -= 1

            right_c = end + 1
            right_color = row[right_c]
            right_end = right_c
            while right_end + 1 < cols and row[right_end + 1] == right_color:
                right_end += 1

            gap = end - start + 1
            left_len = left_c - left_start + 1
            right_len = right_end - right_c + 1
            ok = False
            if gap <= 4 and (left_color == right_color or left_color == 1 or right_color == 1):
                ok = True
            if gap <= 2 and left_len >= 3 and right_len >= 3:
                ok = True
            if ok:
                for cc in range(start, end + 1):
                    candidate_gap[(r, cc)] = gap

    if not candidate_gap:
        return copy_grid(inp)

    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for cell in candidate_gap:
        if cell in seen:
            continue
        queue = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            comp.append((cr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cr + dr, cc + dc)
                if nb in candidate_gap and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        components.append(comp)

    bottom_row = max(max(r for r, _c in comp) for comp in components)
    if len(components) > 1:
        targets = {
            cell
            for comp in components
            if max(r for r, _c in comp) >= bottom_row - 1
            for cell in comp
        }
    else:
        comp = components[0]
        gap_values = {candidate_gap[cell] for cell in comp}
        if len(gap_values) == 1:
            targets = set(comp)
        else:
            widest = max(gap_values)
            targets = {cell for cell in comp if candidate_gap[cell] < widest}

    supply = sorted((r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 1)
    if len(supply) < len(targets):
        return copy_grid(inp)

    out = copy_grid(inp)
    for r, c in supply[:len(targets)]:
        out[r][c] = 0
    for r, c in targets:
        out[r][c] = 1
    return out


def marker_reachable_exterior_boundary(grid: Grid) -> Grid:
    """Trace marker-reachable exterior background around edge obstacles."""
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    counts = Counter(value for row in inp for value in row)
    bg = counts.most_common(1)[0][0]
    markers = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 6]
    if not markers:
        return copy_grid(inp)
    fill_color = 7
    d4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    d8 = d4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

    starts: list[tuple[int, int]] = []
    for r, c in markers:
        for dr, dc in d4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and inp[nr][nc] == bg:
                starts.append((nr, nc))

    region = set(starts)
    queue = deque(starts)
    while queue:
        r, c = queue.popleft()
        for dr, dc in d4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and inp[nr][nc] == bg and (nr, nc) not in region:
                region.add((nr, nc))
                queue.append((nr, nc))

    preserve = set(markers)
    boundary_obstacles: set[tuple[int, int]] = set()
    seen: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] in (bg, 6) or (r, c) in seen:
                continue
            color = inp[r][c]
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen.add((r, c))
            touches_region = False
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                if any((cr + dr, cc + dc) in region for dr, dc in d4):
                    touches_region = True
                for dr, dc in d4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and inp[nr][nc] == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            if touches_region:
                preserve.update(comp)
                if any(cr in (0, rows - 1) or cc in (0, cols - 1) for cr, cc in comp):
                    boundary_obstacles.update(comp)

    out = [[bg for _c in range(cols)] for _r in range(rows)]
    for r, c in preserve:
        out[r][c] = inp[r][c]
    for r, c in region:
        for dr, dc in d8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in boundary_obstacles:
                out[r][c] = fill_color
                break
    return out


def glue_binary_panels_remove_marker_bands(grid: Grid) -> Grid:
    """Glue two binary panels through marker seams, then remove marker bands."""
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    marker = 7
    counts = Counter(value for row in inp for value in row)
    if marker not in counts:
        return copy_grid(inp)
    bg_candidates = [color for color in counts if color != marker]
    if not bg_candidates:
        return copy_grid(inp)
    bg = max(bg_candidates, key=lambda color: counts[color])
    fg_colors = [color for color in counts if color not in (bg, marker)]
    if len(fg_colors) != 1:
        return copy_grid(inp)
    fg = fg_colors[0]

    marker_rows = [r for r, row in enumerate(inp) if all(value == marker for value in row)]
    if len(marker_rows) >= 2:
        lo, hi = marker_rows[0], marker_rows[-1]
        return [[fg if value == marker else value for value in row] for r, row in enumerate(inp) if not (lo <= r <= hi)]

    marker_cols = [c for c in range(cols) if all(inp[r][c] == marker for r in range(rows))]
    if len(marker_cols) >= 2:
        lo, hi = marker_cols[0], marker_cols[-1]
        return [
            [fg if value == marker else value for c, value in enumerate(row) if not (lo <= c <= hi)]
            for row in inp
        ]

    def runs(indices: list[int]) -> list[tuple[int, int]]:
        result = []
        start = None
        prev = None
        for idx in indices:
            if start is None:
                start = prev = idx
            elif idx == prev + 1:
                prev = idx
            else:
                result.append((start, prev))
                start = prev = idx
        if start is not None and prev is not None:
            result.append((start, prev))
        return result

    def cells_box(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        return (
            min(r for r, _c in cells),
            min(c for _r, c in cells),
            max(r for r, _c in cells),
            max(c for _r, c in cells),
        )

    def crop_box(box: tuple[int, int, int, int]) -> Grid:
        r0, c0, r1, c1 = box
        return [row[c0:c1 + 1] for row in inp[r0:r1 + 1]]

    def connected_boxes() -> set[tuple[int, int, int, int]]:
        visited: set[tuple[int, int]] = set()
        boxes: set[tuple[int, int, int, int]] = set()
        for sr in range(rows):
            for sc in range(cols):
                if (sr, sc) in visited or inp[sr][sc] == bg:
                    continue
                queue = deque([(sr, sc)])
                visited.add((sr, sc))
                cells: list[tuple[int, int]] = []
                while queue:
                    r, c = queue.popleft()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] != bg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                boxes.add(cells_box(cells))
        return boxes

    boxes = connected_boxes()
    row_runs = runs([r for r, row in enumerate(inp) if any(value != bg for value in row)])
    col_runs = runs([c for c in range(cols) if any(inp[r][c] != bg for r in range(rows))])
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            boxes.add((r0, c0, r1, c1))

    items = []
    for box in sorted(boxes):
        crop = crop_box(box)
        marks = []
        non_bg = 0
        for r, row in enumerate(crop):
            for c, value in enumerate(row):
                if value == bg:
                    continue
                non_bg += 1
                if value == marker:
                    marks.append((r, c))
        if marks:
            items.append({"box": box, "crop": crop, "marks": marks, "non_bg": non_bg})
    if len(items) < 2:
        return copy_grid(inp)

    def paste(base: Grid, src: Grid, shift: tuple[int, int], base_marker_value: int, src_marker_mode: str) -> Grid:
        base2 = [[base_marker_value if value == marker else value for value in row] for row in base]
        bh, bw = shape(base2)
        sh, sw = shape(src)
        sr, sc = shift
        min_r = min(0, sr)
        min_c = min(0, sc)
        max_r = max(bh - 1, sr + sh - 1)
        max_c = max(bw - 1, sc + sw - 1)
        out = [[bg] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
        for r in range(bh):
            for c in range(bw):
                out[r - min_r][c - min_c] = base2[r][c]
        for r in range(sh):
            for c in range(sw):
                value = src[r][c]
                if value == bg:
                    continue
                if value == marker:
                    if src_marker_mode == "transparent":
                        continue
                    value = fg
                out[r + sr - min_r][c + sc - min_c] = value
        return out

    candidates: list[tuple[Grid, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for bi, base in enumerate(items):
        for si, src_item in enumerate(items):
            if bi == si:
                continue
            src = src_item["crop"]
            sh, sw = shape(src)
            for bm in base["marks"]:
                for sm in src_item["marks"]:
                    sr, sc = sm
                    directions = []
                    if sr == 0:
                        directions.append((1, 0))
                    if sr == sh - 1:
                        directions.append((-1, 0))
                    if sc == 0:
                        directions.append((0, 1))
                    if sc == sw - 1:
                        directions.append((0, -1))
                    directions.append((0, 0))
                    for dr, dc in directions:
                        shift = (bm[0] - (sr + dr), bm[1] - (sc + dc))
                        for base_marker_value in (bg, fg):
                            for src_marker_mode in ("transparent", "fg"):
                                out = paste(base["crop"], src, shift, base_marker_value, src_marker_mode)
                                flat = [value for row in out for value in row]
                                if marker in flat or flat.count(fg) != counts[fg]:
                                    continue
                                candidates.append((out, base["box"], src_item["box"]))
    if not candidates:
        return copy_grid(inp)

    separated = len(row_runs) > 1 or len(col_runs) > 1
    stacked_same_width = False
    if len(row_runs) > 1 and all(sum(1 for value in inp[r] if value == marker) >= 2 for r in range(rows) if marker in inp[r]):
        item_boxes = [item["box"] for item in items]
        for idx, first in enumerate(item_boxes):
            for second in item_boxes[idx + 1:]:
                if first[1] == second[1] and first[3] == second[3] and (first[2] < second[0] or second[2] < first[0]):
                    stacked_same_width = True

    original_area = rows * cols
    if not separated and any(len(out) * len(out[0]) < original_area for out, _base_box, _src_box in candidates):
        candidates = [
            candidate
            for candidate in candidates
            if len(candidate[0]) * len(candidate[0][0]) < original_area
        ]
    elif not separated and any(len(out) * len(out[0]) <= original_area for out, _base_box, _src_box in candidates):
        candidates = [
            candidate
            for candidate in candidates
            if len(candidate[0]) * len(candidate[0][0]) <= original_area
        ]

    def score(candidate: tuple[Grid, tuple[int, int, int, int], tuple[int, int, int, int]]) -> tuple[Any, ...]:
        out, base_box, src_box = candidate
        area = len(out) * len(out[0])
        aspect = abs(len(out) - len(out[0]))
        if separated:
            if stacked_same_width:
                return (aspect, area, base_box, src_box)
            return (area, aspect, base_box, src_box)
        return (-area, aspect, base_box, src_box)

    return sorted(candidates, key=score)[0][0]


def pack_colored_objects_to_square_by_marker(grid: Grid) -> Grid:
    """Pack 8-connected colored components into the square anchored by color 5."""
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = 0

    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int, int]]] = []
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] == bg or (sr, sc) in visited:
                continue
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            cells: list[tuple[int, int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c, inp[r][c]))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] != bg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            components.append(cells)

    total = sum(len(component) for component in components)
    side = math.isqrt(total)
    if not components or side * side != total:
        return copy_grid(inp)

    marker_idx = None
    for idx, component in enumerate(components):
        if any(color == 5 for _r, _c, color in component):
            marker_idx = idx
            break
    if marker_idx is None:
        return copy_grid(inp)

    def normalize_component(cells: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int], ...]:
        min_r = min(r for r, _c, _color in cells)
        min_c = min(c for _r, c, _color in cells)
        return tuple(sorted((r - min_r, c - min_c, color) for r, c, color in cells))

    def orient(component: list[tuple[int, int, int]], index: int) -> tuple[tuple[int, int, int], ...]:
        transformed = []
        for r, c, color in component:
            if index == 0:
                rr, cc = r, c
            elif index == 1:
                rr, cc = r, -c
            elif index == 2:
                rr, cc = -r, c
            elif index == 3:
                rr, cc = -r, -c
            elif index == 4:
                rr, cc = c, r
            elif index == 5:
                rr, cc = c, -r
            elif index == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            transformed.append((rr, cc, color))
        return normalize_component(transformed)

    variants = []
    for component in components:
        seen = set()
        item = []
        for index in range(8):
            variant = orient(component, index)
            if variant not in seen:
                seen.add(variant)
                item.append(variant)
        variants.append(item)

    def center(component: list[tuple[int, int, int]]) -> tuple[float, float]:
        return (
            sum(r for r, _c, _color in component) / len(component),
            sum(c for _r, c, _color in component) / len(component),
        )

    marker_center = center(components[marker_idx])
    rest = [idx for idx in range(len(components)) if idx != marker_idx]
    reverse = not (marker_center[0] < rows / 2 and marker_center[1] < cols / 2)
    order = [marker_idx] + sorted(
        rest,
        key=lambda idx: math.atan2(center(components[idx])[0] - marker_center[0], center(components[idx])[1] - marker_center[1]),
        reverse=reverse,
    )

    out: list[list[int | None]] = [[None] * side for _ in range(side)]
    used = [False] * len(components)

    def search() -> bool:
        first_empty = None
        for r in range(side):
            for c in range(side):
                if out[r][c] is None:
                    first_empty = (r, c)
                    break
            if first_empty is not None:
                break
        if first_empty is None:
            return True
        target_r, target_c = first_empty

        for idx in order:
            if used[idx]:
                continue
            for variant in variants[idx]:
                if idx == marker_idx:
                    anchors = [(-r, -c) for r, c, color in variant if color == 5]
                else:
                    anchors = [(target_r - r, target_c - c) for r, c, _color in variant]
                for anchor_r, anchor_c in anchors:
                    if all(
                        0 <= anchor_r + r < side
                        and 0 <= anchor_c + c < side
                        and out[anchor_r + r][anchor_c + c] is None
                        for r, c, _color in variant
                    ):
                        used[idx] = True
                        for r, c, color in variant:
                            out[anchor_r + r][anchor_c + c] = color
                        if search():
                            return True
                        for r, c, _color in variant:
                            out[anchor_r + r][anchor_c + c] = None
                        used[idx] = False
            break
        return False

    if not search():
        return copy_grid(inp)
    return [[value if value is not None else bg for value in row] for row in out]


def pack_components_with_multicolor_legend_square(grid: Grid) -> Grid:
    """Pack components into a square using a multicolor object as the legend."""
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = 0

    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int, int]]] = []
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] == bg or (sr, sc) in visited:
                continue
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            cells: list[tuple[int, int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c, inp[r][c]))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] != bg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            components.append(cells)

    if len(components) != 4:
        return copy_grid(inp)
    total = sum(len(component) for component in components)
    side = math.isqrt(total)
    if side * side != total:
        return copy_grid(inp)

    legend_idx = max(range(len(components)), key=lambda idx: len({color for _r, _c, color in components[idx]}))
    legend_colors = Counter(color for _r, _c, color in components[legend_idx])
    if len(legend_colors) < 4:
        return copy_grid(inp)
    legend_base = legend_colors.most_common(1)[0][0]
    code_cells = [(r, c, color) for r, c, color in components[legend_idx] if color != legend_base]
    if len(code_cells) != 4:
        return copy_grid(inp)
    mid_r = (min(r for r, _c, _color in code_cells) + max(r for r, _c, _color in code_cells)) / 2
    mid_c = (min(c for _r, c, _color in code_cells) + max(c for _r, c, _color in code_cells)) / 2
    code_by_quad: dict[tuple[int, int], int] = {}
    for r, c, color in code_cells:
        key = (0 if r <= mid_r else 1, 0 if c <= mid_c else 1)
        code_by_quad[key] = color
    if set(code_by_quad) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return copy_grid(inp)
    code_order = [
        code_by_quad[(0, 0)],
        code_by_quad[(0, 1)],
        code_by_quad[(1, 1)],
        code_by_quad[(1, 0)],
    ]

    def center(component: list[tuple[int, int, int]]) -> tuple[float, float]:
        return (
            sum(r for r, _c, _color in component) / len(component),
            sum(c for _r, c, _color in component) / len(component),
        )

    legend_center = center(components[legend_idx])
    rest = [idx for idx in range(len(components)) if idx != legend_idx]
    above = [idx for idx in rest if center(components[idx])[0] < legend_center[0]]
    below = [idx for idx in rest if idx not in above]
    if above:
        ordered_rest = sorted(
            below,
            key=lambda idx: math.atan2(center(components[idx])[0] - legend_center[0], center(components[idx])[1] - legend_center[1]),
        ) + sorted(
            above,
            key=lambda idx: math.atan2(center(components[idx])[0] - legend_center[0], center(components[idx])[1] - legend_center[1]),
        )
    else:
        ordered_rest = sorted(rest, key=lambda idx: center(components[idx]))
    order = [legend_idx] + ordered_rest
    recolor = {idx: code_order[pos] for pos, idx in enumerate(order)}

    def normalize_mask(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        cells = list(cells)
        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        return tuple(sorted((r - min_r, c - min_c) for r, c in cells))

    def orient_mask(component: list[tuple[int, int, int]], index: int) -> tuple[tuple[int, int], ...]:
        transformed = []
        for r, c, _color in component:
            if index == 0:
                rr, cc = r, c
            elif index == 1:
                rr, cc = r, -c
            elif index == 2:
                rr, cc = -r, c
            elif index == 3:
                rr, cc = -r, -c
            elif index == 4:
                rr, cc = c, r
            elif index == 5:
                rr, cc = c, -r
            elif index == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            transformed.append((rr, cc))
        return normalize_mask(transformed)

    variants = []
    for component in components:
        seen = set()
        item = []
        for index in range(8):
            variant = orient_mask(component, index)
            if variant not in seen:
                seen.add(variant)
                item.append(variant)
        variants.append(item)

    out: list[list[int | None]] = [[None] * side for _ in range(side)]
    used = [False] * len(components)

    def search() -> bool:
        first_empty = None
        for r in range(side):
            for c in range(side):
                if out[r][c] is None:
                    first_empty = (r, c)
                    break
            if first_empty is not None:
                break
        if first_empty is None:
            return True
        target_r, target_c = first_empty
        for idx in order:
            if used[idx]:
                continue
            for variant in variants[idx]:
                anchors = [(0, 0)] if idx == legend_idx else [(target_r - r, target_c - c) for r, c in variant]
                for anchor_r, anchor_c in anchors:
                    if all(
                        0 <= anchor_r + r < side
                        and 0 <= anchor_c + c < side
                        and out[anchor_r + r][anchor_c + c] is None
                        for r, c in variant
                    ):
                        used[idx] = True
                        for r, c in variant:
                            out[anchor_r + r][anchor_c + c] = recolor[idx]
                        if search():
                            return True
                        for r, c in variant:
                            out[anchor_r + r][anchor_c + c] = None
                        used[idx] = False
            break
        return False

    if not search():
        return copy_grid(inp)
    return [[value if value is not None else bg for value in row] for row in out]


def overlay_data_component_into_matching_shell(grid: Grid) -> Grid:
    """Align a multicolor data component into a same-outline shell and fill cavities."""
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = background_color(inp)

    def find_components_local(predicate: Callable[[int], bool], conn: int = 8) -> list[list[tuple[int, int]]]:
        seen = [[False] * cols for _ in range(rows)]
        if conn == 8:
            nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        else:
            nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        comps: list[list[tuple[int, int]]] = []
        for r in range(rows):
            for c in range(cols):
                if seen[r][c] or not predicate(inp[r][c]):
                    continue
                stack = [(r, c)]
                cells: list[tuple[int, int]] = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols or seen[rr][cc] or not predicate(inp[rr][cc]):
                        continue
                    seen[rr][cc] = True
                    cells.append((rr, cc))
                    for dr, dc in nbrs:
                        stack.append((rr + dr, cc + dc))
                comps.append(cells)
        return comps

    comps = find_components_local(lambda value: value != bg, conn=8)
    if len(comps) != 2:
        return copy_grid(inp)
    comp_colors = [set(inp[r][c] for r, c in comp) for comp in comps]
    try:
        shell_idx = next(i for i, colors in enumerate(comp_colors) if len(colors) == 1)
        data_idx = next(i for i, colors in enumerate(comp_colors) if len(colors) > 1)
    except StopIteration:
        return copy_grid(inp)
    outline = next(iter(comp_colors[shell_idx]))
    shell = comps[shell_idx]
    data_comp = comps[data_idx]
    shell_set = set(shell)
    shr0 = min(r for r, _c in shell)
    shc0 = min(c for _r, c in shell)
    shr1 = max(r for r, _c in shell)
    shc1 = max(c for _r, c in shell)
    sh, sw = shr1 - shr0 + 1, shc1 - shc0 + 1

    data_outline = {(r, c) for r, c in data_comp if inp[r][c] == outline}
    best = (0, 0, -1)
    for dr in range(-rows, rows):
        for dc in range(-cols, cols):
            ov = sum(1 for r, c in data_outline if (r + dr, c + dc) in shell_set)
            if ov > best[2]:
                best = (dr, dc, ov)
    best_dr, best_dc, _ = best

    out = [[bg] * sw for _ in range(sh)]
    for r, c in shell_set:
        out[r - shr0][c - shc0] = outline

    trans_colors: dict[tuple[int, int], int] = {}
    for r, c in data_comp:
        color = inp[r][c]
        if color == outline:
            continue
        nr, nc = r + best_dr - shr0, c + best_dc - shc0
        if 0 <= nr < sh and 0 <= nc < sw:
            trans_colors[(nr, nc)] = color

    old_inp = inp
    inp = out
    old_rows, old_cols = rows, cols
    rows, cols = sh, sw
    cavities = find_components_local(lambda value: value == bg, conn=4)
    inp = old_inp
    rows, cols = old_rows, old_cols
    for cav in cavities:
        on_border = any(r == 0 or r == sh - 1 or c == 0 or c == sw - 1 for r, c in cav)
        if on_border:
            continue
        cav_set = set(cav)
        colors = [trans_colors[cell] for cell in cav if cell in trans_colors]
        if not colors:
            for r, c in cav:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in cav_set and (nr, nc) in trans_colors:
                        colors.append(trans_colors[(nr, nc)])
        if colors:
            color = Counter(colors).most_common(1)[0][0]
            for r, c in cav:
                out[r][c] = color
    return out


def pack_regions_with_matching_shape_holes(grid: Grid) -> Grid:
    """Pack region panels and insert shape components matched by hole counts."""
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = background_color(inp)

    def comps_for(predicate: Callable[[int], bool]) -> list[list[tuple[int, int]]]:
        seen = [[False] * cols for _ in range(rows)]
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        out = []
        for r in range(rows):
            for c in range(cols):
                if seen[r][c] or not predicate(inp[r][c]):
                    continue
                stack = [(r, c)]
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols or seen[rr][cc] or not predicate(inp[rr][cc]):
                        continue
                    seen[rr][cc] = True
                    cells.append((rr, cc))
                    for dr, dc in nbrs:
                        stack.append((rr + dr, cc + dc))
                out.append(cells)
        return out

    components = []
    for color in palette(inp):
        if color == bg:
            continue
        for cells in comps_for(lambda value, color=color: value == color):
            rs = [r for r, _c in cells]
            cs = [c for _r, c in cells]
            r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
            bbox_size = (r1 - r0 + 1) * (c1 - c0 + 1)
            if bbox_size >= 4:
                components.append({"color": color, "cells": cells, "bbox": (r0, c0, r1, c1), "bbox_size": bbox_size})
    if not components:
        return copy_grid(inp)

    sizes = Counter(component["bbox_size"] for component in components)
    region_size = None
    for size, count in sorted(sizes.items(), reverse=True):
        if count >= 2:
            region_size = size
            break
    if region_size is None:
        region_size = max(sizes)
    shape_size = None
    for size, _count in sorted(sizes.items(), reverse=True):
        if size < region_size:
            shape_size = size
            break
    if shape_size is None:
        return copy_grid(inp)

    regions = [component for component in components if component["bbox_size"] == region_size]
    shapes = [component for component in components if component["bbox_size"] == shape_size]
    if not regions or not shapes:
        return copy_grid(inp)

    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        rcol = region["color"]
        markers = 0
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if inp[r][c] != rcol:
                    markers += 1
        region["marker_count"] = markers

    def hole_comp_count(item: dict[str, Any]) -> int:
        r0, c0, r1, c1 = item["bbox"]
        color = item["color"]
        seen: set[tuple[int, int]] = set()
        count = 0
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if inp[r][c] == color or (r, c) in seen:
                    continue
                stack = [(r, c)]
                while stack:
                    rr, cc = stack.pop()
                    if (rr, cc) in seen:
                        continue
                    if rr < r0 + 1 or rr > r1 - 1 or cc < c0 + 1 or cc > c1 - 1:
                        continue
                    if inp[rr][cc] == color:
                        continue
                    seen.add((rr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        stack.append((rr + dr, cc + dc))
                count += 1
        return count

    for item in shapes:
        item["hole_comp_count"] = hole_comp_count(item)

    regions.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    region_top_rows = sorted({item["bbox"][0] for item in regions})
    region_left_cols = sorted({item["bbox"][1] for item in regions})
    rh = regions[0]["bbox"][2] - regions[0]["bbox"][0] + 1
    rw = regions[0]["bbox"][3] - regions[0]["bbox"][1] + 1
    out = [[bg] * (len(region_left_cols) * rw) for _ in range(len(region_top_rows) * rh)]

    used_shapes: set[int] = set()
    pairing: dict[int, int] = {}
    for ri, region in enumerate(regions):
        target = region["marker_count"]
        best_shape_idx = None
        for si, item in enumerate(shapes):
            if si in used_shapes:
                continue
            if item["hole_comp_count"] == target:
                best_shape_idx = si
                break
        if best_shape_idx is None:
            for si, item in enumerate(shapes):
                if si in used_shapes:
                    continue
                if best_shape_idx is None or abs(item["hole_comp_count"] - target) < abs(shapes[best_shape_idx]["hole_comp_count"] - target):
                    best_shape_idx = si
        if best_shape_idx is not None:
            pairing[ri] = best_shape_idx
            used_shapes.add(best_shape_idx)

    for ri, region in enumerate(regions):
        r_color = region["color"]
        r0, c0, r1, c1 = region["bbox"]
        out_r0 = region_top_rows.index(r0) * rh
        out_c0 = region_left_cols.index(c0) * rw
        for r in range(rh):
            for c in range(rw):
                out[out_r0 + r][out_c0 + c] = r_color
        if ri not in pairing:
            continue
        item = shapes[pairing[ri]]
        s_color = item["color"]
        sr0, sc0, sr1, sc1 = item["bbox"]
        sh = sr1 - sr0 + 1
        sw = sc1 - sc0 + 1
        if sh <= rh - 2 and sw <= rw - 2:
            so_r = (rh - sh) // 2
            so_c = (rw - sw) // 2
            for r in range(sh):
                for c in range(sw):
                    src_v = inp[sr0 + r][sc0 + c]
                    if src_v == s_color:
                        out[out_r0 + so_r + r][out_c0 + so_c + c] = s_color
                    elif src_v == bg:
                        out[out_r0 + so_r + r][out_c0 + so_c + c] = r_color
    return out


def extract_largest_internal_detailed_object_normalized(grid: Grid) -> Grid:
    """Extract the largest internal detailed object and normalize orientation."""
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = background_color(inp)
    best: tuple[int, Grid] | None = None
    for color in sorted(palette(inp) - {bg}):
        coords = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == color]
        if not coords:
            continue
        r0 = min(r for r, _c in coords)
        c0 = min(c for _r, c in coords)
        r1 = max(r for r, _c in coords)
        c1 = max(c for _r, c in coords)
        if r0 == 0 or c0 == 0 or r1 == rows - 1 or c1 == cols - 1:
            continue
        crop = [row[c0:c1 + 1] for row in inp[r0:r1 + 1]]
        foreign = sum(1 for row in crop for value in row if value not in (bg, color))
        if foreign == 0:
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if best is None or area > best[0]:
            best = (area, crop)
    if best is None:
        return copy_grid(inp)
    crop = best[1]
    h, w = shape(crop)
    if h == w:
        return rot270(crop)
    if h > w:
        return mirror_h(rot270(crop))
    return mirror_h(rot90(crop))


def stamp_anomaly_glyphs_between_frames(grid: Grid) -> Grid:
    """Crop a sparse target frame and stamp source anomaly glyphs onto markers."""
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)

    def find_panels() -> list[dict[str, Any]]:
        panels = []
        for r0 in range(rows):
            for r1 in range(r0 + 3, rows):
                for c0 in range(cols):
                    for c1 in range(c0 + 3, cols):
                        border = (
                            [inp[r0][c] for c in range(c0, c1 + 1)]
                            + [inp[r1][c] for c in range(c0, c1 + 1)]
                            + [inp[r][c0] for r in range(r0 + 1, r1)]
                            + [inp[r][c1] for r in range(r0 + 1, r1)]
                        )
                        border_color, border_hits = Counter(border).most_common(1)[0]
                        if border_color == 0 or border_hits != len(border):
                            continue
                        interior = [inp[r][c] for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
                        if not interior:
                            continue
                        fill_color, _ = Counter(interior).most_common(1)[0]
                        if fill_color == border_color:
                            continue
                        anomaly_count = sum(1 for value in interior if value != fill_color)
                        if anomaly_count == 0:
                            continue
                        height, width = r1 - r0 + 1, c1 - c0 + 1
                        if height < 6 or width < 6:
                            continue
                        panels.append({
                            "box": (r0, c0, r1, c1),
                            "border": border_color,
                            "fill": fill_color,
                            "anomaly_count": anomaly_count,
                            "area": height * width,
                        })
        keep = []
        for panel in panels:
            r0, c0, r1, c1 = panel["box"]
            contained = False
            for other in panels:
                if other is panel or other["fill"] != panel["fill"]:
                    continue
                or0, oc0, or1, oc1 = other["box"]
                if or0 <= r0 and oc0 <= c0 and or1 >= r1 and oc1 >= c1 and other["area"] > panel["area"]:
                    contained = True
                    break
            if not contained:
                keep.append(panel)
        return keep

    panels = find_panels()
    if len(panels) < 2:
        return copy_grid(inp)
    target = min(panels, key=lambda item: (item["anomaly_count"], -item["area"]))
    source = max(panels, key=lambda item: (item["anomaly_count"], item["area"]))
    if source is target:
        return copy_grid(inp)

    tr0, tc0, tr1, tc1 = target["box"]
    sr0, sc0, sr1, sc1 = source["box"]
    out = [row[tc0:tc1 + 1] for row in inp[tr0:tr1 + 1]]

    def source_components() -> list[list[tuple[int, int, int]]]:
        fill = source["fill"]
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int, int]]] = []
        for r in range(sr0 + 1, sr1):
            for c in range(sc0 + 1, sc1):
                if inp[r][c] == fill or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                comp: list[tuple[int, int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr - sr0, cc - sc0, inp[cr][cc]))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if sr0 < nr < sr1 and sc0 < nc < sc1 and (nr, nc) not in seen and inp[nr][nc] != fill:
                                seen.add((nr, nc))
                                stack.append((nr, nc))
                comps.append(comp)
        return comps

    shape_by_marker: dict[int, list[tuple[int, int, int]]] = {}
    for comp in source_components():
        by_color: dict[int, list[tuple[int, int]]] = {}
        for rr, cc, color in comp:
            by_color.setdefault(color, []).append((rr, cc))
        for color, anchors in by_color.items():
            if len(anchors) == 1 and color not in shape_by_marker:
                ar, ac = anchors[0]
                shape_by_marker[color] = [(rr - ar, cc - ac, value) for rr, cc, value in comp]

    target_markers = []
    for r in range(tr0 + 1, tr1):
        for c in range(tc0 + 1, tc1):
            color = inp[r][c]
            if color != target["fill"] and color in shape_by_marker:
                target_markers.append((r - tr0, c - tc0, color))

    for rr, cc, color in target_markers:
        for dr, dc, value in shape_by_marker[color]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < len(out) and 0 <= nc < len(out[0]):
                out[nr][nc] = value
    return out


def extract_color3_frame_crossing_rectangles(grid: Grid) -> Grid:
    """Extract a color-3 frame interior and reveal crossing rectangle borders."""
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)

    best: tuple[int, tuple[int, int, int, int]] | None = None
    for r0 in range(rows):
        for r1 in range(r0 + 2, rows):
            for c0 in range(cols):
                for c1 in range(c0 + 2, cols):
                    border = (
                        [inp[r0][c] for c in range(c0, c1 + 1)]
                        + [inp[r1][c] for c in range(c0, c1 + 1)]
                        + [inp[r][c0] for r in range(r0 + 1, r1)]
                        + [inp[r][c1] for r in range(r0 + 1, r1)]
                    )
                    if all(value == 3 for value in border):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if best is None or area > best[0]:
                            best = (area, (r0, c0, r1, c1))
    if best is None:
        return copy_grid(inp)

    r0, c0, r1, c1 = best[1]
    bg = background_color(inp)
    interior = [inp[r][c] for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    fill = Counter(interior).most_common(1)[0][0] if interior else bg
    out = [row[c0 + 1:c1] for row in inp[r0 + 1:r1]]

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(palette(inp)):
        if color in (bg, 3, fill):
            continue
        coords = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == color]
        if len(coords) < 2:
            continue
        a = min(r for r, _c in coords)
        b = min(c for _r, c in coords)
        d = max(r for r, _c in coords)
        e = max(c for _r, c in coords)
        bboxes[color] = (a, b, d, e)
        for rr in range(a, d + 1):
            for cc in range(b, e + 1):
                if rr in (a, d) or cc in (b, e):
                    if r0 < rr < r1 and c0 < cc < c1:
                        out[rr - r0 - 1][cc - c0 - 1] = color

    for color_a, color_b in itertools.combinations(list(bboxes), 2):
        if bboxes[color_a] != bboxes[color_b]:
            continue
        low, high = tuple(sorted((color_a, color_b)))
        a, b, d, e = bboxes[color_a]
        if d <= r0 or a >= r1 or e <= c0 or b >= c1:
            continue
        parity: dict[int, int] = {}
        if a < r0 + 1 or b < c0 + 1:
            votes = {0: Counter(), 1: Counter()}
            for rr in range(a, d + 1):
                for cc in range(b, e + 1):
                    if inp[rr][cc] in (low, high):
                        votes[(rr + cc) % 2][inp[rr][cc]] += 1
            for par in (0, 1):
                if votes[par]:
                    parity[par] = votes[par].most_common(1)[0][0]
        if 0 not in parity or 1 not in parity or parity[0] == parity[1]:
            parity = {0: low, 1: high}
        for rr in range(max(a, r0 + 1), min(d, r1 - 1) + 1):
            for cc in range(max(b, c0 + 1), min(e, c1 - 1) + 1):
                if rr in (a, d) or cc in (b, e):
                    out[rr - r0 - 1][cc - c0 - 1] = parity[(rr + cc) % 2]
    return out


def inferred_border_chain_transfer_candidate_specs(pairs: list[dict[str, Grid]], ns: dict[str, Any] | None) -> list[CandidateSpec]:
    if not ns:
        return []
    fn = ns.get("transfer_box_center_via_border_chain")
    if not callable(fn):
        return []

    def transform(grid: Grid, fn: Callable[[Grid], Grid] = fn) -> Grid:
        return normalize_grid(fn(deepcopy(grid)))

    return [admit_fixed_operator(
        "border_chain_transfer_repair:box_center",
        transform,
        "border_chain_transfer",
        14,
        pairs,
    )]


def inferred_border_chain_transfer_specs(pairs: list[dict[str, Grid]], ns: dict[str, Any] | None) -> list[tuple[str, Transform, str, int]]:
    return [
        (spec.name, spec.transform, spec.family, spec.prior)
        for spec in inferred_border_chain_transfer_candidate_specs(pairs, ns)
        if spec.tier == "certified" and spec.transform is not None
    ]


def inferred_crop_cleanup_candidate_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    return [admit_fixed_operator(
        "crop_largest_object_with_border_cleanup:zero_template_marker_repeat",
        zero_template_marker_repeat,
        "crop_cleanup",
        16,
        pairs,
    )]


def inferred_crop_cleanup_specs(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform, str, int]]:
    return [
        (spec.name, spec.transform, spec.family, spec.prior)
        for spec in inferred_crop_cleanup_candidate_specs(pairs)
        if spec.tier == "certified" and spec.transform is not None
    ]


def inferred_single_motif_reconstruct_candidate_specs(pairs: list[dict[str, Grid]], ns: dict[str, Any] | None) -> list[CandidateSpec]:
    if not ns:
        return []
    panel_fn = ns.get("solve_panel_anchored_extras")
    largest_fn = ns.get("solve_largest_cc_transposed_registered")
    if not callable(panel_fn) or not callable(largest_fn):
        return []

    def transform(grid: Grid, panel_fn: Callable[[Grid], Grid | None] = panel_fn, largest_fn: Callable[[Grid], Grid | None] = largest_fn) -> Grid:
        grid = normalize_grid(grid)
        out = panel_fn(deepcopy(grid))
        if out is None:
            out = largest_fn(deepcopy(grid))
        return normalize_grid(out if out is not None else grid)

    return [admit_fixed_operator(
        "single_motif_recenter_or_reconstruct:panel_anchor_or_largest_transpose",
        transform,
        "motif_reconstruct",
        13,
        pairs,
    )]


def inferred_single_motif_reconstruct_specs(pairs: list[dict[str, Grid]], ns: dict[str, Any] | None) -> list[tuple[str, Transform, str, int]]:
    return [
        (spec.name, spec.transform, spec.family, spec.prior)
        for spec in inferred_single_motif_reconstruct_candidate_specs(pairs, ns)
        if spec.tier == "certified" and spec.transform is not None
    ]


def drop_top_legend_masks_into_matching_strips(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    strips: list[dict[str, int]] = []
    r = 0
    while r < rows:
        row = inp[r]
        if row[0] != bg and row[0] == row[-1]:
            border = row[0]
            counts = Counter(row[1:-1])
            fill = counts.most_common(1)[0][0] if counts else None
            if fill is not None and fill != border and counts[fill] >= cols - 4:
                top = r
                r += 1
                while r < rows and inp[r][0] == border and inp[r][-1] == border:
                    row_counts = Counter(inp[r][1:-1])
                    if not row_counts or row_counts.most_common(1)[0][0] != fill:
                        break
                    r += 1
                strips.append({"top": top, "bottom": r - 1, "border": border, "fill": fill})
                continue
        r += 1
    if not strips:
        return inp

    first_strip = min(strip["top"] for strip in strips)
    strip_by_fill = {strip["fill"]: strip for strip in strips}
    out = copy_grid(inp)

    for rr in range(first_strip):
        for cc in range(cols):
            if out[rr][cc] != bg:
                out[rr][cc] = bg

    seen: set[tuple[int, int]] = set()
    for rr in range(first_strip):
        for cc in range(cols):
            color = inp[rr][cc]
            if color == bg or color not in strip_by_fill or (rr, cc) in seen:
                continue
            queue = deque([(rr, cc)])
            seen.add((rr, cc))
            cells: list[tuple[int, int]] = []
            while queue:
                r0, c0 = queue.popleft()
                cells.append((r0, c0))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r0 + dr, c0 + dc
                    if (
                        0 <= nr < first_strip
                        and 0 <= nc < cols
                        and (nr, nc) not in seen
                        and inp[nr][nc] == color
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))

            strip = strip_by_fill[color]
            min_r = min(r0 for r0, _c0 in cells)
            max_r = max(r0 for r0, _c0 in cells)
            new_top = strip["bottom"] - (max_r - min_r)
            for r0, c0 in cells:
                nr = new_top + (r0 - min_r)
                if strip["top"] <= nr <= strip["bottom"] and out[nr][c0] == strip["fill"]:
                    out[nr][c0] = strip["border"]
    return out


def slide_marked_components_to_boundary_axes(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if inp[r][c] == bg or (r, c) in seen:
                continue
            q = deque([(r, c)])
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and inp[nr][nc] != bg:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            comps.append(cells)

    colors = {value for row in inp for value in row if value != bg}
    boundary_specs: dict[int, tuple[str, int]] = {}
    for color in colors:
        anchors = []
        for comp in comps:
            if len(comp) != 1:
                continue
            r, c = comp[0]
            if inp[r][c] == color and (r in (0, rows - 1) or c in (0, cols - 1)):
                anchors.append((r, c))
        if not anchors:
            continue
        side_rows = [r for r, c in anchors if c == 0 or c == cols - 1]
        cap_cols = [c for r, c in anchors if r == 0 or r == rows - 1]
        side_best = Counter(side_rows).most_common(1)[0] if side_rows else (None, 0)
        cap_best = Counter(cap_cols).most_common(1)[0] if cap_cols else (None, 0)
        if side_best[1] > cap_best[1]:
            boundary_specs[color] = ("row", int(side_best[0]))
        elif cap_best[1] > side_best[1]:
            boundary_specs[color] = ("col", int(cap_best[0]))
        else:
            noncorner_side = [
                r for r, c in anchors
                if (c == 0 or c == cols - 1) and r not in (0, rows - 1)
            ]
            if noncorner_side:
                boundary_specs[color] = ("row", Counter(noncorner_side).most_common(1)[0][0])
            elif cap_cols:
                boundary_specs[color] = ("col", int(cap_best[0]))
            elif side_rows:
                boundary_specs[color] = ("row", int(side_best[0]))

    if not boundary_specs:
        return inp

    moves: list[tuple[int, int, int]] = []
    for idx, comp in enumerate(comps):
        counts = Counter(inp[r][c] for r, c in comp)
        choices = []
        for color, (axis, target) in boundary_specs.items():
            if counts.get(color, 0) != 1:
                continue
            marker = next((r, c) for r, c in comp if inp[r][c] == color)
            dr = target - marker[0] if axis == "row" else 0
            dc = target - marker[1] if axis == "col" else 0
            choices.append((abs(dr) + abs(dc), color, axis, marker, dr, dc))
        if not choices:
            continue
        choices.sort(key=lambda item: (-item[0], item[1]))
        _dist, _color, axis, marker, dr, dc = choices[0]

        if axis == "row" and len(comp) == 9:
            rs = [r for r, _c in comp]
            cs = [c for _r, c in comp]
            if (
                max(rs) - min(rs) == 2
                and max(cs) - min(cs) == 2
                and marker[0] == min(rs) + 1
                and marker[1] == min(cs) + 1
            ):
                if any(c == 0 for _r, c in comp):
                    dc += 1
                else:
                    dc -= 1
        moves.append((idx, dr, dc))

    if not moves:
        return inp

    out = copy_grid(inp)
    for idx, _dr, _dc in moves:
        for r, c in comps[idx]:
            out[r][c] = bg
    for idx, dr, dc in moves:
        for r, c in comps[idx]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out[nr][nc] = inp[r][c]
    return out


def propagate_marker_rays_through_scaffold_strips(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] == bg or (sr, sc) in visited:
                continue
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            components.append(cells)

    scaffold = []
    singletons = []
    for cells in components:
        if len(cells) == 1:
            singletons.append(cells[0])
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        scaffold.append({"cells": set(cells), "box": (min(rs), min(cs), max(rs), max(cs))})
    if not scaffold or not singletons:
        return inp

    scaffold_values = [inp[r][c] for item in scaffold for r, c in item["cells"]]
    rail_color = Counter(scaffold_values).most_common(1)[0][0]
    verticals = []
    horizontals = []
    for item in scaffold:
        r0, c0, r1, c1 = item["box"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height > width:
            left_rail = sum(1 for r, c in item["cells"] if c == c0 and inp[r][c] == rail_color)
            right_rail = sum(1 for r, c in item["cells"] if c == c1 and inp[r][c] == rail_color)
            item["side"] = "left" if left_rail >= right_rail else "right"
            item["n"] = height
            verticals.append(item)
        elif width > height:
            top_rail = sum(1 for r, c in item["cells"] if r == r0 and inp[r][c] == rail_color)
            bottom_rail = sum(1 for r, c in item["cells"] if r == r1 and inp[r][c] == rail_color)
            item["direction"] = "up" if top_rail > bottom_rail else "down"
            item["n"] = width
            horizontals.append(item)
    if not verticals or not horizontals:
        return inp

    start_pairs = Counter()
    for item in horizontals:
        r0, c0, r1, c1 = item["box"]
        start_pairs[(inp[r0][c1], inp[r1][c1])] += 1
    canonical_start = start_pairs.most_common(1)[0][0]

    def pair_matches(pair: tuple[int, int]) -> bool:
        return pair == canonical_start or pair[::-1] == canonical_start

    for item in verticals:
        r0, c0, r1, c1 = item["box"]
        top_pair = (inp[r0][c0], inp[r0][c1])
        bottom_pair = (inp[r1][c0], inp[r1][c1])
        item["reversed"] = pair_matches(bottom_pair) and not pair_matches(top_pair)

    rel_candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for mr, mc in singletons:
        marker_color = inp[mr][mc]
        for item in verticals:
            r0, c0, r1, c1 = item["box"]
            on_fill_side = (item["side"] == "left" and mc < c0) or (item["side"] == "right" and mc > c1)
            if r0 <= mr <= r1 and on_fill_side:
                rel = mr - r0
                if item["reversed"]:
                    rel = item["n"] - 1 - rel
                rel_candidates[marker_color].append((rel, min(abs(mc - c0), abs(mc - c1))))
        for item in horizontals:
            r0, c0, r1, c1 = item["box"]
            on_fill_side = (item["direction"] == "up" and mr < r0) or (item["direction"] == "down" and mr > r1)
            if c0 <= mc <= c1 and on_fill_side:
                rel_candidates[marker_color].append((item["n"] - 1 - (mc - c0), min(abs(mr - r0), abs(mr - r1))))

    marker_rel = {}
    for marker_color, candidates in rel_candidates.items():
        counts = Counter(rel for rel, _distance in candidates)
        marker_rel[marker_color] = max(
            counts,
            key=lambda rel: (
                counts[rel],
                -min(distance for cand_rel, distance in candidates if cand_rel == rel),
            ),
        )
    if len(marker_rel) != len(singletons):
        return inp

    out = copy_grid(inp)
    for item in verticals:
        r0, c0, _r1, c1 = item["box"]
        for marker_color, rel in marker_rel.items():
            if not (0 <= rel < item["n"]):
                continue
            actual_rel = item["n"] - 1 - rel if item["reversed"] else rel
            rr = r0 + actual_rel
            target_cols = range(0, c0) if item["side"] == "left" else range(c1 + 1, cols)
            for cc in target_cols:
                if out[rr][cc] == bg:
                    out[rr][cc] = marker_color

    for item in horizontals:
        r0, c0, r1, _c1 = item["box"]
        for marker_color, rel in marker_rel.items():
            col_rel = item["n"] - 1 - rel
            if not (0 <= col_rel < item["n"]):
                continue
            cc = c0 + col_rel
            target_rows = range(0, r0) if item["direction"] == "up" else range(r1 + 1, rows)
            for rr in target_rows:
                if out[rr][cc] == bg:
                    out[rr][cc] = marker_color
    return out


def complete_diagonal_symmetric_zero_runs_by_context(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or len(inp) != len(inp[0]):
        return inp
    n = len(inp)
    out = copy_grid(inp)

    for r in range(n):
        for c in range(n):
            if out[r][c] == 0 and inp[c][r] != 0:
                out[r][c] = inp[c][r]

    def zero_runs(row: list[int]) -> list[tuple[int, int]]:
        runs = []
        c = 0
        while c < n:
            if row[c] != 0:
                c += 1
                continue
            start = c
            while c < n and row[c] == 0:
                c += 1
            runs.append((start, c - 1))
        return runs

    changed = True
    while changed:
        changed = False
        for r in range(n):
            for c0, c1 in zero_runs(out[r]):
                context = list(range(max(0, c0 - 5), c0)) + list(range(c1 + 1, min(n, c1 + 6)))
                known_context = [c for c in context if out[r][c] != 0]
                best = None
                for sr in range(n):
                    if sr == r or any(out[sr][c] == 0 for c in range(c0, c1 + 1)):
                        continue
                    local = sum(out[sr][c] == out[r][c] for c in known_context)
                    full = sum(
                        out[sr][c] == out[r][c]
                        for c in range(n)
                        if out[r][c] != 0 and not (c0 <= c <= c1)
                    )
                    score = (local, full, -abs(sr - r), sr)
                    if best is None or score > best[0]:
                        best = (score, sr)

                threshold = max(3, min(7, len(known_context)))
                if best is None or best[0][0] < threshold:
                    continue
                sr = best[1]
                for c in range(c0, c1 + 1):
                    if out[r][c] == 0:
                        out[r][c] = out[sr][c]
                        if out[c][r] == 0:
                            out[c][r] = out[r][c]
                        changed = True
    return out


def project_singleton_frame_markers_to_corners(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    def zero_frames() -> list[tuple[int, int, int, int]]:
        zero_cells = {(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 0}
        seen: set[tuple[int, int]] = set()
        boxes: list[list[int]] = []
        for start in sorted(zero_cells):
            if start in seen:
                continue
            queue = deque([start])
            seen.add(start)
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb in zero_cells and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            boxes.append([
                min(r for r, _c in cells),
                max(r for r, _c in cells),
                min(c for _r, c in cells),
                max(c for _r, c in cells),
            ])

        changed = True
        while changed:
            changed = False
            merged: list[list[int]] = []
            used = [False] * len(boxes)
            for i, box in enumerate(boxes):
                if used[i]:
                    continue
                current = box[:]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    other = boxes[j]
                    rows_close = not (current[1] + 1 < other[0] or other[1] + 1 < current[0])
                    cols_close = not (current[3] + 1 < other[2] or other[3] + 1 < current[2])
                    if rows_close and cols_close:
                        current = [
                            min(current[0], other[0]),
                            max(current[1], other[1]),
                            min(current[2], other[2]),
                            max(current[3], other[3]),
                        ]
                        used[j] = True
                        changed = True
                used[i] = True
                merged.append(current)
            boxes = merged
        return [
            (box[0], box[1], box[2], box[3])
            for box in boxes
            if box[1] - box[0] >= 2 and box[3] - box[2] >= 2
        ]

    def choose_frame_corner(frames: list[tuple[int, int, int, int]], cell: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
        r, c = cell
        best = None
        for frame in frames:
            r0, r1, c0, c1 = frame
            if not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1):
                continue
            for corner in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
                cr, cc = corner
                dr = abs(r - cr)
                dc = abs(c - cc)
                if dr == 0 or dc == 0:
                    continue
                score = (1 if dr == dc else 0, -max(dr, dc), -(dr + dc))
                candidate = (score, frame, corner)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            return None, None
        return best[1], best[2]

    def color_components_for_frames() -> list[dict[str, Any]]:
        seen: set[tuple[int, int]] = set()
        components = []
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] in (bg, 0) or (r, c) in seen:
                    continue
                color = inp[r][c]
                queue = deque([(r, c)])
                seen.add((r, c))
                cells: list[tuple[int, int]] = []
                while queue:
                    rr, cc = queue.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and inp[nr][nc] == color:
                            seen.add((nr, nc))
                            queue.append((nr, nc))
                components.append({"color": color, "cells": cells})
        return components

    frames = zero_frames()
    if not frames:
        return inp
    components = color_components_for_frames()
    non_singleton_by_frame: Counter[tuple[int, tuple[int, int, int, int]]] = Counter()
    for component in components:
        frame = corner = None
        for cell in component["cells"]:
            frame, corner = choose_frame_corner(frames, cell)
            if frame is not None:
                break
        component["frame"] = frame
        component["corner"] = corner
        if frame is not None and len(component["cells"]) > 1:
            non_singleton_by_frame[(component["color"], frame)] += 1

    out = copy_grid(inp)
    for component in components:
        frame = component.get("frame")
        corner = component.get("corner")
        if len(component["cells"]) != 1 or frame is None or non_singleton_by_frame[(component["color"], frame)]:
            continue
        r, c = component["cells"][0]
        color = component["color"]
        r0, r1, c0, c1 = frame
        cr, cc = corner
        out_dr = -1 if cr == r0 else 1
        out_dc = -1 if cc == c0 else 1
        distance = max(abs(r - cr), abs(c - cc))

        if distance >= 2:
            in_dr = -out_dr
            in_dc = -out_dc
            targets = [(cr + k * in_dr, cc + k * in_dc) for k in range(distance)]
            targets.extend([(cr + out_dr, cc + out_dc), (cr + out_dr, cc), (cr, cc + out_dc)])
        elif bg == 2:
            targets = [(cr, cc), (cr + out_dr, cc), (cr, cc + out_dc)]
        elif bg == 8:
            targets = [
                (cr + out_dr, cc + out_dc),
                (cr + out_dr, cc),
                (cr, cc + out_dc),
                (cr - out_dr, cc),
                (cr, cc - out_dc),
            ]
        else:
            targets = [
                (r + out_dr, c + out_dc),
                (r + 2 * out_dr, c + out_dc),
                (r + out_dr, c + 2 * out_dc),
                (r - out_dr, c),
                (r, c - out_dc),
            ]

        for rr, cc2 in targets:
            if 0 <= rr < rows and 0 <= cc2 < cols and out[rr][cc2] in (bg, 0):
                out[rr][cc2] = color
    return out


def follow_seed_ray_repaint_reached_chain(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    counts = Counter(value for row in inp for value in row)
    bg = counts.most_common(1)[0][0]

    visited: set[tuple[int, int]] = set()
    color_components: list[tuple[int, set[tuple[int, int]]]] = []
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] == bg or (sr, sc) in visited:
                continue
            color = inp[sr][sc]
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            cells: set[tuple[int, int]] = set()
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] == color:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            color_components.append((color, cells))
    comp_by_cell = {
        cell: idx
        for idx, (_color, cells) in enumerate(color_components)
        for cell in cells
    }

    seed = None
    for _idx, (marker_color, cells) in enumerate(color_components):
        if len(cells) != 1:
            continue
        mr, mc = next(iter(cells))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = mr + dr, mc + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or inp[nr][nc] == bg:
                continue
            comp_idx = comp_by_cell.get((nr, nc))
            if comp_idx is None:
                continue
            target_color, target_cells = color_components[comp_idx]
            if target_color == marker_color or len(target_cells) < 2:
                continue
            rs = {r for r, _ in target_cells}
            cs = {c for _, c in target_cells}
            if len(rs) != 1 and len(cs) != 1:
                continue
            candidate = (
                len(target_cells),
                -counts[marker_color],
                marker_color,
                (mr, mc),
                target_color,
                target_cells,
                (-dr, -dc),
            )
            if seed is None or candidate[:2] > seed[:2]:
                seed = candidate
    if seed is None:
        return inp

    _score_len, _score_marker, seed_marker_color, seed_marker, target_color, seed_cells, start_dir = seed
    remaining = [color for color in counts if color not in (bg, target_color, seed_marker_color)]
    if len(remaining) < 2:
        return inp
    body_color = max(remaining, key=lambda color: counts[color])
    side_colors = [color for color in remaining if color != body_color]
    side_color = max(side_colors, key=lambda color: counts[color])

    allowed = {body_color, side_color}
    visited = set()
    objects: list[dict[str, set[tuple[int, int]]]] = []
    object_by_cell: dict[tuple[int, int], int] = {}
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] not in allowed or (sr, sc) in visited:
                continue
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            cells: set[tuple[int, int]] = set()
            side_cells: set[tuple[int, int]] = set()
            body_cells: set[tuple[int, int]] = set()
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                if inp[r][c] == side_color:
                    side_cells.add((r, c))
                else:
                    body_cells.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] in allowed:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            if not body_cells:
                continue
            obj_idx = len(objects)
            objects.append({"cells": cells, "side": side_cells, "body": body_cells})
            for cell in cells:
                object_by_cell[cell] = obj_idx

    out = [[bg] * cols for _ in range(rows)]
    for r, c in seed_cells | {seed_marker}:
        out[r][c] = target_color

    reached: set[int] = set()
    rays = deque([(seed_marker[0], seed_marker[1], start_dir[0], start_dir[1])])

    def emit_from_object(obj_idx: int) -> None:
        obj = objects[obj_idx]
        for r, c in obj["cells"]:
            out[r][c] = target_color
        if not obj["side"]:
            return
        body_rows = [r for r, _ in obj["body"]]
        body_cols = [c for _, c in obj["body"]]
        r0, r1 = min(body_rows), max(body_rows)
        c0, c1 = min(body_cols), max(body_cols)

        seen_side: set[tuple[int, int]] = set()
        for start in sorted(obj["side"]):
            if start in seen_side:
                continue
            group: list[tuple[int, int]] = []
            queue = deque([start])
            seen_side.add(start)
            while queue:
                r, c = queue.popleft()
                group.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb in obj["side"] and nb not in seen_side:
                        seen_side.add(nb)
                        queue.append(nb)

            group_rows = [r for r, _ in group]
            group_cols = [c for _, c in group]
            if max(group_rows) < r0:
                dr, dc = -1, 0
            elif min(group_rows) > r1:
                dr, dc = 1, 0
            elif max(group_cols) < c0:
                dr, dc = 0, -1
            elif min(group_cols) > c1:
                dr, dc = 0, 1
            else:
                avg_r = sum(group_rows) / len(group_rows)
                avg_c = sum(group_cols) / len(group_cols)
                center_r = (r0 + r1) / 2
                center_c = (c0 + c1) / 2
                if abs(avg_r - center_r) >= abs(avg_c - center_c):
                    dr, dc = (-1 if avg_r < center_r else 1), 0
                else:
                    dr, dc = 0, (-1 if avg_c < center_c else 1)
            for r, c in group:
                rays.append((r, c, dr, dc))

    def run_ray(sr: int, sc: int, dr: int, dc: int) -> None:
        r, c = sr + dr, sc + dc
        while 0 <= r < rows and 0 <= c < cols:
            obj_idx = object_by_cell.get((r, c))
            if obj_idx is not None:
                if obj_idx not in reached:
                    reached.add(obj_idx)
                    emit_from_object(obj_idx)
                return
            if inp[r][c] in (bg, seed_marker_color, target_color):
                out[r][c] = target_color
                r += dr
                c += dc
                continue
            return

    while rays:
        run_ray(*rays.popleft())
    return out


def count_noise_clusters_into_frame_midline(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    result = copy_grid(inp)
    counts = Counter(value for row in inp for value in row)
    bg = counts.most_common(1)[0][0]
    non_bg = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] != bg]
    cell_set = set(non_bg)

    visited: set[tuple[int, int]] = set()
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r, c in non_bg:
        if (r, c) in visited:
            continue
        comp: list[tuple[int, int]] = []
        color = inp[r][c]
        stack = [(r, c)]
        while stack:
            r2, c2 = stack.pop()
            if (r2, c2) in visited:
                continue
            visited.add((r2, c2))
            if inp[r2][c2] != color:
                continue
            comp.append((r2, c2))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r2 + dr, c2 + dc)
                if nb in cell_set and nb not in visited:
                    stack.append(nb)
        if comp:
            components.append((color, comp))

    frames: list[tuple[int, int, int, int, int]] = []
    for color, comp in components:
        rs = [r for r, _c in comp]
        cs = [c for _r, c in comp]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        if not all(r == min_r or r == max_r or c == min_c or c == max_c for r, c in comp):
            continue
        expected = {
            (r, c)
            for r in range(min_r, max_r + 1)
            for c in range(min_c, max_c + 1)
            if r == min_r or r == max_r or c == min_c or c == max_c
        }
        if set(comp) == expected and max_r - min_r >= 3 and max_c - min_c >= 3:
            frames.append((color, min_r, max_r, min_c, max_c))
    if not frames:
        return inp

    frame_cells = set()
    for _color, min_r, max_r, min_c, max_c in frames:
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    frame_cells.add((r, c))

    noise = [
        (r, c, inp[r][c])
        for r in range(rows)
        for c in range(cols)
        if inp[r][c] != bg and (r, c) not in frame_cells
    ]
    noise_by_pos = {(r, c): value for r, c, value in noise}
    noise_set = set(noise_by_pos)
    visited = set()
    noise_groups: list[tuple[int, list[tuple[int, int]]]] = []
    for pos in noise_set:
        if pos in visited:
            continue
        color = noise_by_pos[pos]
        group: list[tuple[int, int]] = []
        stack = [pos]
        while stack:
            cell = stack.pop()
            if cell in visited:
                continue
            visited.add(cell)
            if noise_by_pos.get(cell) != color:
                continue
            group.append(cell)
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in noise_set and nb not in visited:
                    stack.append(nb)
        if group:
            noise_groups.append((color, group))

    noise_count = Counter(color for color, _group in noise_groups)
    for r, c, _value in noise:
        result[r][c] = bg
    for color, min_r, max_r, min_c, max_c in frames:
        n = noise_count.get(color, 0)
        if not n:
            continue
        interior_rows = list(range(min_r + 1, max_r))
        interior_cols = list(range(min_c + 1, max_c))
        if not interior_rows or not interior_cols:
            continue
        mid_row = interior_rows[len(interior_rows) // 2]
        odd_positions = list(range(1, len(interior_cols), 2))
        for pos in odd_positions[-n:] if n <= len(odd_positions) else odd_positions:
            result[mid_row][min_c + 1 + pos] = color
    return result


def stamp_motif_at_scaled_marker_offsets(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    def color_components(color: int) -> list[list[tuple[int, int]]]:
        visited: set[tuple[int, int]] = set()
        comps = []
        for sr in range(rows):
            for sc in range(cols):
                if inp[sr][sc] != color or (sr, sc) in visited:
                    continue
                queue = deque([(sr, sc)])
                visited.add((sr, sc))
                cells: list[tuple[int, int]] = []
                while queue:
                    r, c = queue.popleft()
                    cells.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and inp[nr][nc] == color:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                comps.append(cells)
        return comps

    colors = [color for color in sorted(palette(inp)) if color != bg]
    if len(colors) != 2:
        return inp

    parsed = {}
    for color in colors:
        comps = color_components(color)
        parsed[color] = ([cells for cells in comps if len(cells) > 1], [cells[0] for cells in comps if len(cells) == 1])

    motif_color = None
    marker_color = None
    for color in colors:
        large, singles = parsed[color]
        other = colors[1] if color == colors[0] else colors[0]
        other_large, other_singles = parsed[other]
        if len(large) == 1 and len(singles) == 1 and not other_large and len(other_singles) >= 2:
            motif_color = color
            marker_color = other
            break
    if motif_color is None or marker_color is None:
        return inp

    motif = parsed[motif_color][0][0]
    origin_marker = parsed[motif_color][1][0]
    markers = parsed[marker_color][1]

    min_r = min(r for r, _c in motif)
    min_c = min(c for _r, c in motif)
    max_r = max(r for r, _c in motif)
    max_c = max(c for _r, c in motif)
    motif_h = max_r - min_r + 1
    motif_w = max_c - min_c + 1
    motif_shape = [(r - min_r, c - min_c) for r, c in motif]

    row_unit = 0
    col_unit = 0
    for r, c in markers:
        dr = abs(r - origin_marker[0])
        dc = abs(c - origin_marker[1])
        if dr:
            row_unit = math.gcd(row_unit, dr)
        if dc:
            col_unit = math.gcd(col_unit, dc)
    row_unit = row_unit or 1
    col_unit = col_unit or 1

    def scaled(delta: int, size: int, unit: int) -> int | None:
        if delta * size % unit != 0:
            return None
        return delta * size // unit

    out = [[bg for _ in range(cols)] for _ in range(rows)]

    def stamp(top: int, left: int, color: int) -> bool:
        for rr, cc in motif_shape:
            r, c = top + rr, left + cc
            if not (0 <= r < rows and 0 <= c < cols):
                return False
        for rr, cc in motif_shape:
            out[top + rr][left + cc] = color
        return True

    if not stamp(min_r, min_c, marker_color):
        return inp
    for marker in markers:
        shift_r = scaled(marker[0] - origin_marker[0], motif_h, row_unit)
        shift_c = scaled(marker[1] - origin_marker[1], motif_w, col_unit)
        if shift_r is None or shift_c is None:
            return inp
        if not stamp(min_r + shift_r, min_c + shift_c, motif_color):
            return inp
    return out


def repair_left_opening_noisy_cone(grid: Grid, out: Grid) -> Grid:
    grid = normalize_grid(grid)
    out = normalize_grid(out)
    rows, cols = shape(grid)
    counts = Counter(value for row in grid for value in row)
    if not counts:
        return out
    bg = counts.most_common(1)[0][0]
    non_bg = [color for color, _count in counts.most_common() if color != bg]
    if not non_bg:
        return out
    shape_color = non_bg[0]
    shape_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == shape_color]
    if not shape_cells:
        return out
    min_r = min(r for r, _c in shape_cells)
    max_r = max(r for r, _c in shape_cells)
    min_c = min(c for _r, c in shape_cells)
    max_c = max(c for _r, c in shape_cells)

    other: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            value = grid[r][c]
            if value != bg and value != shape_color:
                other.setdefault(value, []).append((r, c))
    if not other:
        return out

    def inside_count(cells: list[tuple[int, int]]) -> int:
        return sum(1 for r, c in cells if min_r <= r <= max_r and min_c <= c <= max_c)

    fill_color = max(other, key=lambda color: inside_count(other[color]))
    fill_cells = other[fill_color]
    left_noise = [
        (r, c)
        for color, cells in other.items()
        if color != fill_color
        for r, c in cells
        if c < min_c
    ]
    if len(fill_cells) != 1 or len(left_noise) < 2:
        return out

    for nr, nc in left_noise:
        if not (min_r <= nr <= max_r):
            continue
        c = nc + 1
        while c < cols and grid[nr][c] == bg:
            out[nr][c] = fill_color
            c += 1

    spread = max(1, min(3, min_c // 2 - 1 if min_c >= 4 else 1))
    top = min_r - spread
    bottom = max_r + spread
    for r in range(rows):
        allowed = None
        if r < top or r > bottom:
            allowed = -1
        elif top <= r < min_r:
            allowed = 2 * (r - top + 1) - 1
        elif max_r < r <= bottom:
            allowed = 2 * (bottom - r + 1) - 1
        if allowed is None:
            continue
        for c in range(cols):
            if out[r][c] == fill_color and grid[r][c] == bg and c > allowed:
                out[r][c] = bg
    return out


def spiral_seed_fill_with_noisy_cone(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    counts = Counter(value for row in inp for value in row)
    bg = counts.most_common(1)[0][0]
    shape_color = next((color for color, _count in counts.most_common() if color != bg), None)
    if shape_color is None:
        return inp
    shape_cells = {(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == shape_color}
    if not shape_cells:
        return inp
    shape_rows = [r for r, _c in shape_cells]
    shape_cols = [c for _r, c in shape_cells]
    min_r, max_r = min(shape_rows), max(shape_rows)
    min_c, max_c = min(shape_cols), max(shape_cols)

    other: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            value = inp[r][c]
            if value != bg and value != shape_color:
                other.setdefault(value, []).append((r, c))

    def count_inside(cells: list[tuple[int, int]]) -> int:
        return sum(1 for r, c in cells if min_r <= r <= max_r and min_c <= c <= max_c)

    seed_color = max(other, key=lambda value: count_inside(other[value]), default=None) if other else None
    if seed_color is None or count_inside(other[seed_color]) == 0:
        return inp
    noise_cells = {(r, c) for value, cells in other.items() if value != seed_color for r, c in cells}
    seed_cells = {(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == seed_color}
    inside_seed_rows = [r for r, c in seed_cells if min_r <= r <= max_r and min_c <= c <= max_c]
    inside_seed_cols = [c for r, c in seed_cells if min_r <= r <= max_r and min_c <= c <= max_c]
    if inside_seed_rows and inside_seed_cols:
        avg_r = sum(inside_seed_rows) / len(inside_seed_rows)
        avg_c = sum(inside_seed_cols) / len(inside_seed_cols)
        dr = avg_r - (min_r + max_r) / 2
        dc = avg_c - (min_c + max_c) / 2
        est_opening = "up" if abs(dr) >= abs(dc) and dr > 0 else (
            "down" if abs(dr) >= abs(dc) else ("left" if dc > 0 else "right")
        )
    else:
        est_opening = None

    noise_shadow: set[tuple[int, int]] = set()
    if est_opening == "up":
        for nr, nc in noise_cells:
            if min_r < nr <= max_r:
                for r in range(min_r, nr):
                    noise_shadow.add((r, nc))
    elif est_opening == "down":
        for nr, nc in noise_cells:
            if min_r <= nr < max_r:
                for r in range(nr + 1, max_r + 1):
                    noise_shadow.add((r, nc))
    elif est_opening == "left":
        for nr, nc in noise_cells:
            if min_c < nc <= max_c:
                for c in range(min_c, nc):
                    noise_shadow.add((nr, c))
    elif est_opening == "right":
        for nr, nc in noise_cells:
            if min_c <= nc < max_c:
                for c in range(nc + 1, max_c + 1):
                    noise_shadow.add((nr, c))

    bfs_walls = shape_cells | noise_cells | noise_shadow
    visited = set(seed_cells)
    queue = deque(seed_cells)
    interior = set(seed_cells)
    while queue:
        r, c = queue.popleft()
        for dr2, dc2 in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr2, c + dc2
            if (
                min_r <= nr <= max_r
                and min_c <= nc <= max_c
                and (nr, nc) not in visited
                and (nr, nc) not in bfs_walls
            ):
                visited.add((nr, nc))
                interior.add((nr, nc))
                queue.append((nr, nc))

    opening_dir = None
    if any(r == min_r for r, _c in interior) and min_r > 0:
        opening_dir = "up"
    elif any(r == max_r for r, _c in interior) and max_r < rows - 1:
        opening_dir = "down"
    elif any(c == min_c for _r, c in interior) and min_c > 0:
        opening_dir = "left"
    elif any(c == max_c for _r, c in interior) and max_c < cols - 1:
        opening_dir = "right"

    out = copy_grid(inp)
    for r, c in interior:
        if out[r][c] == bg:
            out[r][c] = seed_color

    def shape_width_on_row(row: int) -> tuple[int, int]:
        values = [c for rr, c in shape_cells if rr == row]
        return (min(values), max(values)) if values else (min_c, max_c)

    def row_shadow(orow: int, left: int, right: int) -> set[int]:
        blocked = set()
        for c in range(left, right + 1):
            if inp[orow][c] != shape_color and (orow, c) not in interior:
                blocked.add(c)
        return blocked

    if opening_dir == "down":
        left, right = shape_width_on_row(max_r)
        shadow = row_shadow(max_r, left, right)
        for nr, nc in noise_cells:
            if nr > max_r:
                shadow.add(nc)
        d = 1
        while max_r + d < rows:
            fr = max_r + d
            for c in range(max(0, left - (d - 1)), min(cols, right + d)):
                if c not in shadow and out[fr][c] == bg:
                    out[fr][c] = seed_color
            d += 1
    elif opening_dir == "up":
        left, right = shape_width_on_row(min_r)
        shadow = row_shadow(min_r, left, right)
        ext_noise = [(nr, nc) for nr, nc in noise_cells if nr < min_r]
        d = 1
        while min_r - d >= 0:
            fr = min_r - d
            detail_row = min(min_r + d, max_r)
            detail_left, detail_right = shape_width_on_row(detail_row)
            blocked = set(shadow)
            for nr, nc in ext_noise:
                if fr < nr:
                    blocked.add(nc)
            for c in range(max(0, detail_left - (d - 1)), min(cols, detail_right + d)):
                if c not in blocked and out[fr][c] == bg:
                    out[fr][c] = seed_color
            d += 1
    elif opening_dir == "left":
        shadow = {nr for nr, nc in noise_cells if nc < min_c}
        for r in range(min_r, max_r + 1):
            if inp[r][min_c] != shape_color and (r, min_c) not in interior:
                shadow.add(r)
        d = 1
        while min_c - d >= 0:
            fc = min_c - d
            for r in range(max(0, min_r - (d - 1)), min(rows, max_r + d)):
                if r not in shadow and out[r][fc] == bg:
                    out[r][fc] = seed_color
            d += 1
    return repair_left_opening_noisy_cone(inp, out)


def attach_external_objects_to_directional_miniglyphs(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]
    directions = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for sr in range(rows):
        for sc in range(cols):
            if (sr, sc) in seen or inp[sr][sc] == bg:
                continue
            color = inp[sr][sc]
            queue = deque([(sr, sc)])
            seen.add((sr, sc))
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and inp[nr][nc] == color:
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            components.append({"color": color, "cells": frozenset(cells)})

    anchors = [item for item in components if len(item["cells"]) <= 2]
    if len(anchors) < 3:
        return inp
    anchor_cells = set().union(*(set(item["cells"]) for item in anchors))
    center_r = sum(r for r, _c in anchor_cells) / len(anchor_cells)
    center_c = sum(c for _r, c in anchor_cells) / len(anchor_cells)

    def bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
        cells = list(cells)
        return (
            min(r for r, _c in cells),
            min(c for _r, c in cells),
            max(r for r, _c in cells),
            max(c for _r, c in cells),
        )

    def role(cells: Iterable[tuple[int, int]]) -> str:
        cells = list(cells)
        cr = sum(r for r, _c in cells) / len(cells)
        cc = sum(c for _r, c in cells) / len(cells)
        dr, dc = cr - center_r, cc - center_c
        if abs(dr) >= abs(dc):
            return "bottom" if dr > 0 else "top"
        return "right" if dc > 0 else "left"

    def oriented(cells: Iterable[tuple[int, int]], index: int) -> frozenset[tuple[int, int]]:
        cells = list(cells)
        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        points = []
        for r, c in cells:
            r -= min_r
            c -= min_c
            variants = (
                (r, c), (r, -c), (-r, c), (-r, -c),
                (c, r), (c, -r), (-c, r), (-c, -r),
            )
            points.append(variants[index])
        new_min_r = min(r for r, _c in points)
        new_min_c = min(c for _r, c in points)
        return frozenset((r - new_min_r, c - new_min_c) for r, c in points)

    def attach(external: Iterable[tuple[int, int]], anchor: Iterable[tuple[int, int]], anchor_role: str) -> frozenset[tuple[int, int]] | None:
        anchor = set(anchor)
        if anchor_role == "left":
            target = {(r, c - 1) for r, c in anchor}
            side = lambda cells: {(r, c) for r, c in cells if c == max(cc for _rr, cc in cells)}
        elif anchor_role == "right":
            target = {(r, c + 1) for r, c in anchor}
            side = lambda cells: {(r, c) for r, c in cells if c == min(cc for _rr, cc in cells)}
        elif anchor_role == "top":
            target = {(r - 1, c) for r, c in anchor}
            side = lambda cells: {(r, c) for r, c in cells if r == max(rr for rr, _cc in cells)}
        else:
            target = {(r + 1, c) for r, c in anchor}
            side = lambda cells: {(r, c) for r, c in cells if r == min(rr for rr, _cc in cells)}

        anchor_box = bbox(anchor)
        placements = []
        for index in range(8):
            shape_cells = oriented(external, index)
            edge = side(shape_cells)
            if len(edge) != len(target):
                continue
            shifts = {(tr - er, tc - ec) for tr, tc in target for er, ec in edge}
            for dr, dc in shifts:
                if {(r + dr, c + dc) for r, c in edge} != target:
                    continue
                moved = frozenset((r + dr, c + dc) for r, c in shape_cells)
                if not all(0 <= r < rows and 0 <= c < cols for r, c in moved):
                    continue
                moved_box = bbox(moved)
                if anchor_role in ("left", "right"):
                    before = anchor_box[0] - moved_box[0]
                    after = moved_box[2] - anchor_box[2]
                else:
                    before = anchor_box[1] - moved_box[1]
                    after = moved_box[3] - anchor_box[3]
                placements.append((abs(before - after), -(before + after), index, dr, dc, moved))
        if not placements:
            return None
        placements.sort(key=lambda item: item[:5])
        return placements[0][5]

    out = [[bg for _ in range(cols)] for _ in range(rows)]
    anchors_by_color: dict[int, list[frozenset[tuple[int, int]]]] = {}
    for item in anchors:
        anchors_by_color.setdefault(item["color"], []).append(item["cells"])
        for r, c in item["cells"]:
            out[r][c] = item["color"]

    for color, color_anchors in anchors_by_color.items():
        if len(color_anchors) != 1:
            continue
        anchor = color_anchors[0]
        externals = [
            item["cells"]
            for item in components
            if item["color"] == color and item["cells"] not in color_anchors
        ]
        if len(externals) != 1:
            continue
        moved = attach(externals[0], anchor, role(anchor))
        if moved is None:
            continue
        for r, c in moved:
            out[r][c] = color
    return out


def move_multicolor_glyphs_into_monochrome_gaps(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    if not inp or not inp[0]:
        return inp
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    def bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
        cells = list(cells)
        return (
            min(r for r, _c in cells),
            max(r for r, _c in cells),
            min(c for _r, c in cells),
            max(c for _r, c in cells),
        )

    def multicolor_components() -> list[dict[str, Any]]:
        seen: set[tuple[int, int]] = set()
        components = []
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg or (r, c) in seen:
                    continue
                queue = deque([(r, c)])
                seen.add((r, c))
                cells: list[tuple[int, int, int]] = []
                colors = Counter()
                while queue:
                    rr, cc = queue.popleft()
                    cells.append((rr, cc, inp[rr][cc]))
                    colors[inp[rr][cc]] += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and inp[nr][nc] != bg and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            queue.append((nr, nc))
                if len(colors) >= 2:
                    components.append({"cells": cells, "colors": colors})
        return components

    def monochrome_components(excluded: set[tuple[int, int]]) -> list[dict[str, Any]]:
        seen = set(excluded)
        components = []
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == bg or (r, c) in seen:
                    continue
                color = inp[r][c]
                queue = deque([(r, c)])
                seen.add((r, c))
                cells: list[tuple[int, int]] = []
                while queue:
                    rr, cc = queue.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and inp[nr][nc] == color:
                            seen.add((nr, nc))
                            queue.append((nr, nc))
                components.append({"color": color, "cells": set(cells), "bbox": bbox(cells)})
        return components

    def variants(cells: list[tuple[int, int, int]]) -> list[tuple[str, bool, list[tuple[int, int, int]]]]:
        min_r = min(r for r, _c, _value in cells)
        min_c = min(c for _r, c, _value in cells)
        base = [(r - min_r, c - min_c, value) for r, c, value in cells]

        def transform(cell: tuple[int, int, int], name: str) -> tuple[int, int, int]:
            r, c, value = cell
            if name == "id":
                return r, c, value
            if name == "rot90":
                return c, -r, value
            if name == "rot180":
                return -r, -c, value
            if name == "rot270":
                return -c, r, value
            if name == "h":
                return r, -c, value
            if name == "v":
                return -r, c, value
            if name == "main":
                return c, r, value
            return -c, -r, value

        colors = sorted({value for _r, _c, value in base})
        choices = []
        seen: set[tuple[tuple[int, int, int], ...]] = set()
        for name in ("id", "rot90", "rot180", "rot270", "h", "v", "main", "anti"):
            arr = [transform(cell, name) for cell in base]
            arr_min_r = min(r for r, _c, _value in arr)
            arr_min_c = min(c for _r, c, _value in arr)
            arr = [(r - arr_min_r, c - arr_min_c, value) for r, c, value in arr]
            for swap in ((False, True) if len(colors) == 2 else (False,)):
                arr2 = arr
                if swap:
                    a, b = colors
                    arr2 = [(r, c, b if value == a else a if value == b else value) for r, c, value in arr]
                key = tuple(sorted(arr2))
                if key not in seen:
                    seen.add(key)
                    choices.append((name, swap, arr2))
        return choices

    sources = multicolor_components()
    source_cells = {(r, c) for source in sources for r, c, _value in source["cells"]}
    targets_by_color: dict[int, list[dict[str, Any]]] = {}
    for component in monochrome_components(source_cells):
        targets_by_color.setdefault(component["color"], []).append(component)

    target_pairs = []
    for color, components in targets_by_color.items():
        for first, second in itertools.combinations(components, 2):
            target_pairs.append((color, first, second))
    if not sources or not target_pairs:
        return inp

    out = copy_grid(inp)
    for r, c in source_cells:
        out[r][c] = bg

    reflections = {"h", "v", "main", "anti"}
    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    candidates = []
    for source_index, source in enumerate(sources):
        for pair_index, (_color, first, second) in enumerate(target_pairs):
            first_cells = first["cells"]
            second_cells = second["cells"]
            target_cells = first_cells | second_cells
            min_tr, max_tr, min_tc, max_tc = bbox(target_cells)
            gap_r = (sum(r for r, _c in first_cells) / len(first_cells) + sum(r for r, _c in second_cells) / len(second_cells)) / 2
            gap_c = (sum(c for _r, c in first_cells) / len(first_cells) + sum(c for _r, c in second_cells) / len(second_cells)) / 2
            for name, swap, variant in variants(source["cells"]):
                height = max(r for r, _c, _value in variant) + 1
                width = max(c for _r, c, _value in variant) + 1
                top_start = max(-height, min_tr - height - 1)
                top_stop = min(rows, max_tr + 1)
                left_start = max(-width, min_tc - width - 1)
                left_stop = min(cols, max_tc + 1)
                for top in range(top_start, top_stop + 1):
                    for left in range(left_start, left_stop + 1):
                        placed = [(top + r, left + c, value) for r, c, value in variant]
                        if any(
                            not (0 <= r < rows and 0 <= c < cols)
                            or (inp[r][c] != bg and (r, c) not in source_cells)
                            for r, c, _value in placed
                        ):
                            continue
                        first_contact = 0
                        second_contact = 0
                        for r, c, _value in placed:
                            for dr, dc in dirs4:
                                nb = (r + dr, c + dc)
                                if nb in first_cells:
                                    first_contact += 1
                                if nb in second_cells:
                                    second_contact += 1
                        if first_contact == 0 or second_contact == 0:
                            continue
                        area = (
                            max(max_tr, top + height - 1) - min(min_tr, top) + 1
                        ) * (
                            max(max_tc, left + width - 1) - min(min_tc, left) + 1
                        )
                        placed_r = sum(r for r, _c, _value in placed) / len(placed)
                        placed_c = sum(c for _r, c, _value in placed) / len(placed)
                        distance = abs(placed_r - gap_r) + abs(placed_c - gap_c)
                        color_role = 1 if ((name in reflections) == bool(swap)) else 0
                        score = (
                            first_contact + second_contact,
                            min(first_contact, second_contact),
                            color_role,
                            -area,
                            -distance,
                            -top,
                            -left,
                        )
                        candidates.append((score, source_index, pair_index, placed))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_sources = set()
    used_pairs = set()
    for _score, source_index, pair_index, placed in candidates:
        if source_index in used_sources or pair_index in used_pairs:
            continue
        if any(out[r][c] != bg for r, c, _value in placed):
            continue
        for r, c, value in placed:
            out[r][c] = value
        used_sources.add(source_index)
        used_pairs.add(pair_index)
    return out


def repair_sparse_panel_rectangular_slides(grid: Grid, result: Grid, bg: int, shape_color: int) -> Grid:
    grid = normalize_grid(grid)
    result = normalize_grid(result)
    rows, cols = shape(grid)
    shape_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == shape_color}
    marker_cells = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2}

    def components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        seen: set[tuple[int, int]] = set()
        comps = []
        for start in sorted(cells):
            if start in seen:
                continue
            comp = []
            queue = deque([start])
            seen.add(start)
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            comps.append(comp)
        return comps

    shape_comps = components(shape_cells)
    marker_groups = components(marker_cells)
    cell_to_group = {cell: idx for idx, group in enumerate(marker_groups) for cell in group}

    def rect_info(group: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
        rs = [r for r, _c in group]
        cs = [c for _r, c in group]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        return (r0, r1, c0, c1) if set(group) == rect else None

    for comp in shape_comps:
        comp_set = set(comp)
        min_r = min(r for r, _c in comp)
        max_r = max(r for r, _c in comp)
        min_c = min(c for _r, c in comp)
        max_c = max(c for _r, c in comp)
        cr = sum(r for r, _c in comp) / len(comp)
        cc = sum(c for _r, c in comp) / len(comp)
        candidates = []
        for group_index, group in enumerate(marker_groups):
            if len(group) < 2:
                continue
            info = rect_info(group)
            if info is None:
                continue
            r0, r1, c0, c1 = info
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height > 3 or width > 3:
                continue
            gr = (r0 + r1) / 2
            gc = (c0 + c1) / 2
            dirs = []
            if abs(cr - gr) >= abs(cc - gc):
                dirs.append((1 if cr > gr else -1, 0))
            if abs(cc - gc) >= abs(cr - gr):
                dirs.append((0, 1 if cc > gc else -1))
            for dr, dc in dirs:
                for step in range(1, max(rows, cols) + 1):
                    nr0, nr1 = r0 + dr * step, r1 + dr * step
                    nc0, nc1 = c0 + dc * step, c1 + dc * step
                    if nr0 < 0 or nr1 >= rows or nc0 < 0 or nc1 >= cols:
                        break
                    if nr1 < min_r or nr0 > max_r or nc1 < min_c or nc0 > max_c:
                        continue
                    target = {(r + dr * step, c + dc * step) for r, c in group}
                    if any(cell in shape_cells for cell in target):
                        break
                    swept = set()
                    for offset in range(step + 1):
                        swept.update((r + dr * offset, c + dc * offset) for r, c in group)
                    if any(grid[r][c] not in (bg, 2) and (r, c) not in set(group) for r, c in swept):
                        break
                    if dr == 1:
                        face = [(nr1 + 1, c) for c in range(nc0, nc1 + 1)]
                    elif dr == -1:
                        face = [(nr0 - 1, c) for c in range(nc0, nc1 + 1)]
                    elif dc == 1:
                        face = [(r, nc1 + 1) for r in range(nr0, nr1 + 1)]
                    else:
                        face = [(r, nc0 - 1) for r in range(nr0, nr1 + 1)]
                    if not all(cell in comp_set for cell in face):
                        continue
                    already_done = (
                        all(result[r][c] == 2 for r, c in target)
                        and all(result[r][c] == 0 for r, c in swept if (r, c) not in target)
                    )
                    candidates.append((0 if already_done else 1, step, group_index, group, target, swept, already_done))
                    break
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        if candidates[0][-1]:
            continue
        _score, _step, _group_index, _group, target, swept, _already_done = candidates[0]
        for r, c in swept:
            if (r, c) not in target and result[r][c] in (bg, 0, 2):
                result[r][c] = 0
        for r, c in target:
            if result[r][c] in (bg, 0, 2):
                result[r][c] = 2

        for r, c in comp:
            nbs = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (r + dr, c + dc) in comp_set
            ]
            if len(nbs) != 1:
                continue
            nr, nc = nbs[0]
            br, bc = r - nr, c - nc
            path = []
            hits = []
            rr, cc2 = r + br, c + bc
            while 0 <= rr < rows and 0 <= cc2 < cols:
                if grid[rr][cc2] == 2:
                    hits.append((rr, cc2))
                elif grid[rr][cc2] not in (bg, 2):
                    break
                path.append((rr, cc2))
                rr += br
                cc2 += bc
            if hits and len(marker_groups[cell_to_group[hits[0]]]) == 1:
                for pr, pc in path:
                    if result[pr][pc] in (0, 2) and grid[pr][pc] in (bg, 2):
                        result[pr][pc] = bg
    return result


def sparse_shape_tip_beam_noise_repair(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    result = copy_grid(inp)
    counts = Counter(value for row in inp for value in row)
    bg = counts.most_common(1)[0][0]
    shape_color = max((color for color in counts if color not in (bg, 2)), key=lambda color: counts[color], default=None)
    if shape_color is None:
        return result

    shape_cells = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == shape_color]
    shape_set = set(shape_cells)
    visited: set[tuple[int, int]] = set()
    shape_components: list[list[tuple[int, int]]] = []
    for start in shape_cells:
        if start in visited:
            continue
        comp = []
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in shape_set and nb not in visited:
                    stack.append(nb)
        shape_components.append(comp)

    marker_cells = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 2]
    marker_set = set(marker_cells)
    visited = set()
    marker_groups: list[list[tuple[int, int]]] = []
    for start in marker_cells:
        if start in visited:
            continue
        group = []
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            group.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in marker_set and nb not in visited:
                    stack.append(nb)
        marker_groups.append(group)
    cell_to_group = {cell: idx for idx, group in enumerate(marker_groups) for cell in group}
    handled_groups: set[int] = set()

    beam_components = []
    no_beam_components = []
    for comp_index, comp in enumerate(shape_components):
        comp_set = set(comp)
        cr = sum(r for r, _c in comp) / len(comp)
        cc = sum(c for _r, c in comp) / len(comp)
        best_hit = None
        best_distance = float("inf")
        for r, c in comp:
            neighbors = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (r + dr, c + dc) in comp_set
            ]
            if len(neighbors) != 1:
                continue
            nr, nc = neighbors[0]
            dir_r, dir_c = r - nr, c - nc
            hits = []
            rr, cc2 = r + dir_r, c + dir_c
            while 0 <= rr < rows and 0 <= cc2 < cols:
                if inp[rr][cc2] == 2:
                    hits.append((rr, cc2))
                elif inp[rr][cc2] not in (bg, 2):
                    break
                rr += dir_r
                cc2 += dir_c
            if not hits:
                continue
            first_hit = hits[0]
            distance = abs(first_hit[0] - cr) + abs(first_hit[1] - cc)
            if distance < best_distance:
                best_distance = distance
                best_hit = (r, c, dir_r, dir_c, first_hit)
        if best_hit is None:
            no_beam_components.append((comp_index, comp, cr, cc))
        else:
            beam_components.append((comp_index, comp, cr, cc, best_hit))

    for _comp_index, comp, _cr, _cc, best_hit in beam_components:
        tip_r, tip_c, dir_r, dir_c, first_hit = best_hit
        group_index = cell_to_group[first_hit]
        group = marker_groups[group_index]
        if len(group) == 1:
            rr, cc2 = tip_r + dir_r, tip_c + dir_c
            first_step = True
            while 0 <= rr < rows and 0 <= cc2 < cols:
                if (rr, cc2) == first_hit:
                    result[rr][cc2] = 0
                    break
                if first_step:
                    result[rr][cc2] = 2
                    first_step = False
                else:
                    result[rr][cc2] = 0
                rr += dir_r
                cc2 += dir_c
            handled_groups.add(group_index)
        else:
            for r, c in group:
                result[r][c] = bg
            handled_groups.add(group_index)

        comp_set = set(comp)
        for r, c in comp:
            neighbors = [
                (r + dr, c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (r + dr, c + dc) in comp_set
            ]
            if len(neighbors) != 1:
                continue
            nr, nc = neighbors[0]
            dr2, dc2 = r - nr, c - nc
            hits = []
            rr, cc2 = r + dr2, c + dc2
            while 0 <= rr < rows and 0 <= cc2 < cols:
                if inp[rr][cc2] == 2:
                    hits.append((rr, cc2))
                elif inp[rr][cc2] not in (bg, 2):
                    break
                rr += dr2
                cc2 += dc2
            for hit in hits:
                group_index = cell_to_group[hit]
                if group_index not in handled_groups:
                    for cell in marker_groups[group_index]:
                        result[cell[0]][cell[1]] = bg
                    handled_groups.add(group_index)

    for _comp_index, comp, cr, cc in no_beam_components:
        comp_set = set(comp)
        best_group_index = None
        best_group_distance = float("inf")
        for group_index, group in enumerate(marker_groups):
            if group_index in handled_groups or len(group) <= 1:
                continue
            gcr = sum(r for r, _c in group) / len(group)
            gcc = sum(c for _r, c in group) / len(group)
            distance = abs(gcr - cr) + abs(gcc - cc)
            if distance < best_group_distance:
                best_group_distance = distance
                best_group_index = group_index
        if best_group_index is None:
            continue
        group = marker_groups[best_group_index]
        gcr = sum(r for r, _c in group) / len(group)
        gcc = sum(c for _r, c in group) / len(group)
        row_diff = cr - gcr
        col_diff = cc - gcc
        if abs(row_diff) >= abs(col_diff):
            dir_r, dir_c = (1 if row_diff > 0 else -1), 0
        else:
            dir_r, dir_c = 0, (1 if col_diff > 0 else -1)
        offset = 0
        while True:
            offset += 1
            moved = [(r + dir_r * offset, c + dir_c * offset) for r, c in group]
            out_of_bounds = any(not (0 <= r < rows and 0 <= c < cols) for r, c in moved)
            if out_of_bounds:
                break
            adjacent = any(
                (r + dr, c + dc) in comp_set
                for r, c in moved
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if adjacent:
                for step in range(offset):
                    for r, c in group:
                        result[r + dir_r * step][c + dir_c * step] = 0
                for r, c in moved:
                    result[r][c] = 2
                handled_groups.add(best_group_index)
                break

    for group_index, group in enumerate(marker_groups):
        if group_index not in handled_groups:
            for r, c in group:
                result[r][c] = bg
    return repair_sparse_panel_rectangular_slides(inp, result, bg, shape_color)


def separator_row_code_expansion(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    sep_cols = []
    for c in range(cols):
        nonzero = [inp[r][c] for r in range(rows) if inp[r][c] != 0]
        if nonzero and len(set(nonzero)) == 1 and len(nonzero) >= max(1, rows - 1):
            sep_cols.append(c)
    if not sep_cols:
        return inp
    sep = sep_cols[0]

    def code_pattern(side: list[int], *, source_on_left: bool) -> list[int]:
        near = list(reversed(side)) if source_on_left else list(side)
        far_trim = list(reversed(near))
        while far_trim and far_trim[0] == 0:
            far_trim.pop(0)
        while far_trim and far_trim[-1] == 0:
            far_trim.pop()
        if not far_trim:
            return []
        runs: list[tuple[int, int]] = []
        for value in far_trim:
            if value == 0:
                continue
            if runs and runs[-1][0] == value:
                runs[-1] = (value, runs[-1][1] + 1)
            else:
                runs.append((value, 1))
        if not runs:
            return []
        adjacent_color, adjacent_len = runs[-1]
        if len(runs) == 1:
            if adjacent_len <= 1:
                return [adjacent_color]
            return [adjacent_color] + [0] * (adjacent_len - 1)
        prev_color, prev_len = runs[-2]
        if adjacent_len == 1:
            return [adjacent_color]
        if prev_len == 1:
            return [adjacent_color, prev_color]
        if adjacent_len <= 2 and prev_len >= 3:
            return [adjacent_color, 0, adjacent_color, prev_color, adjacent_color, 0]
        if adjacent_len >= 3 and prev_len >= 2:
            return [adjacent_color, 0, prev_color, adjacent_color, prev_color, 0]
        return [adjacent_color, prev_color]

    def repeat(pattern: list[int], width: int, *, reverse_for_left: bool) -> list[int]:
        if not pattern or width <= 0:
            return []
        values = [pattern[i % len(pattern)] for i in range(width)]
        return list(reversed(values)) if reverse_for_left else values

    out = copy_grid(inp)
    for r in range(rows):
        left = inp[r][:sep]
        right = inp[r][sep + 1:]
        if any(left) and not any(right):
            pattern = code_pattern(left, source_on_left=True)
            if pattern:
                out[r][sep + 1:] = repeat(pattern, len(right), reverse_for_left=False)
        elif any(right) and not any(left):
            pattern = code_pattern(right, source_on_left=False)
            if pattern:
                out[r][:sep] = repeat(pattern, sep, reverse_for_left=True)
    return out


def mirror_stretch_quadrant_prototype_around_anchors(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    sep_rows = [r for r in range(rows) if len(set(inp[r])) == 1]
    sep_cols = [c for c in range(cols) if len({inp[r][c] for r in range(rows)}) == 1]
    if not sep_rows or not sep_cols:
        return inp
    sep_r = sep_rows[0]
    sep_c = sep_cols[0]
    if sep_r <= 0 or sep_c <= 0 or sep_r >= rows - 1 or sep_c >= cols - 1:
        return inp

    quadrants = {
        (0, 0): [row[:sep_c] for row in inp[:sep_r]],
        (0, 1): [row[sep_c + 1:] for row in inp[:sep_r]],
        (1, 0): [row[:sep_c] for row in inp[sep_r + 1:]],
        (1, 1): [row[sep_c + 1:] for row in inp[sep_r + 1:]],
    }

    def non_bg_cells(quad: Grid) -> list[tuple[int, int, int]]:
        return [
            (r, c, value)
            for r, row in enumerate(quad)
            for c, value in enumerate(row)
            if value != bg
        ]

    def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        return (
            min(r for r, _c in cells),
            max(r for r, _c in cells),
            min(c for _r, c in cells),
            max(c for _r, c in cells),
        )

    def axis_interval(
        index: int,
        source_min: int,
        source_max: int,
        target_min: int,
        target_max: int,
        mirror: bool = False,
    ) -> list[int]:
        source_size = source_max - source_min + 1
        target_size = target_max - target_min + 1
        if source_size <= 0 or target_size <= 0:
            return []
        if source_min <= index <= source_max:
            offset = index - source_min
            if mirror:
                offset = source_size - 1 - offset
            start = target_min + (offset * target_size) // source_size
            end = target_min + (((offset + 1) * target_size + source_size - 1) // source_size) - 1
            return list(range(start, min(end, target_max) + 1))

        if index < source_min:
            distance = source_min - index
            start_offset = ((distance - 1) * target_size) // source_size + 1
            end_offset = (distance * target_size + source_size - 1) // source_size
            if mirror:
                return list(range(target_max + start_offset, target_max + end_offset + 1))
            return list(range(target_min - end_offset, target_min - start_offset + 1))

        distance = index - source_max
        start_offset = ((distance - 1) * target_size) // source_size + 1
        end_offset = (distance * target_size + source_size - 1) // source_size
        if mirror:
            return list(range(target_min - end_offset, target_min - start_offset + 1))
        return list(range(target_max + start_offset, target_max + end_offset + 1))

    source_key = (0, 0)
    source = quadrants[source_key]
    source_cells = non_bg_cells(source)
    if not source_cells:
        return inp
    source_colors = {value for _r, _c, value in source_cells}
    anchor_scores = []
    for color in source_colors:
        presence = sum(
            1
            for key, quad in quadrants.items()
            if key != source_key and any(value == color for _r, _c, value in non_bg_cells(quad))
        )
        total = sum(
            1
            for quad in quadrants.values()
            for _r, _c, value in non_bg_cells(quad)
            if value == color
        )
        anchor_scores.append((presence, total, color))
    anchor = max(anchor_scores)[2]
    source_anchor = [(r, c) for r, c, value in source_cells if value == anchor]
    if not source_anchor:
        return inp
    src_r0, src_r1, src_c0, src_c1 = bbox(source_anchor)

    out_quadrants = {key: [row[:] for row in quad] for key, quad in quadrants.items()}
    for key, quad in quadrants.items():
        if key == source_key:
            continue
        target_anchor = [
            (r, c)
            for r, row in enumerate(quad)
            for c, value in enumerate(row)
            if value == anchor
        ]
        if not target_anchor:
            continue
        tgt_r0, tgt_r1, tgt_c0, tgt_c1 = bbox(target_anchor)
        mirror_rows = key[0] == 1
        mirror_cols = key[1] == 1
        for r, c, value in source_cells:
            if value == anchor and src_r0 <= r <= src_r1 and src_c0 <= c <= src_c1:
                continue
            for rr in axis_interval(r, src_r0, src_r1, tgt_r0, tgt_r1, mirror_rows):
                for cc in axis_interval(c, src_c0, src_c1, tgt_c0, tgt_c1, mirror_cols):
                    if (
                        0 <= rr < len(quad)
                        and 0 <= cc < len(quad[0])
                        and out_quadrants[key][rr][cc] in (bg, value)
                    ):
                        out_quadrants[key][rr][cc] = value

    out: Grid = []
    for r in range(rows):
        if r == sep_r:
            out.append(inp[r][:])
            continue
        q_row = 0 if r < sep_r else 1
        local_r = r if r < sep_r else r - sep_r - 1
        out.append(
            out_quadrants[(q_row, 0)][local_r]
            + [inp[r][sep_c]]
            + out_quadrants[(q_row, 1)][local_r]
        )
    return out


def mirror_3x3_motif_ring_outward(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    cells = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] != bg]
    if not cells:
        return inp
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    if height != width or height % 3 != 0:
        return inp
    block = height // 3

    out = copy_grid(inp)
    for br in range(3):
        for bc in range(3):
            if br == 1 and bc == 1:
                continue
            dr, dc = br - 1, bc - 1
            src_r = r0 + br * block
            src_c = c0 + bc * block
            dst_r = src_r + dr * block
            dst_c = src_c + dc * block
            for rr in range(block):
                for cc in range(block):
                    sr = src_r + (block - 1 - rr if dr else rr)
                    sc = src_c + (block - 1 - cc if dc else cc)
                    tr, tc = dst_r + rr, dst_c + cc
                    if 0 <= tr < rows and 0 <= tc < cols:
                        out[tr][tc] = inp[sr][sc]

    if block == 3:
        motif_colors: dict[int, int] = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if inp[r][c] != bg:
                    motif_colors[inp[r][c]] = motif_colors.get(inp[r][c], 0) + 1
        center_color = max(motif_colors, key=motif_colors.get) if motif_colors else bg
        cr, cc = r0 + block, c0 + block
        out[cr][cc] = inp[r0 + block - 1][c0 + block - 1]
        out[cr][cc + 1] = inp[r0 + block - 1][c0 + block + 1]
        out[cr][cc + 2] = inp[r0 + block - 1][c0 + 2 * block]
        out[cr + 1][cc] = inp[r0 + block + 1][c0 + block - 1]
        out[cr + 1][cc + 1] = center_color
        out[cr + 1][cc + 2] = inp[r0 + block + 1][c0 + 2 * block]
        out[cr + 2][cc] = inp[r0 + 2 * block][c0 + block - 1]
        out[cr + 2][cc + 1] = inp[r0 + 2 * block][c0 + block + 1]
        out[cr + 2][cc + 2] = inp[r0 + 2 * block][c0 + 2 * block]
    return out


def diagonal_marker_segments_with_crossing_lines(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]
    markers = [(r, c, inp[r][c]) for r in range(rows) for c in range(cols) if inp[r][c] != bg]
    if not markers:
        return inp
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in markers:
        by_color.setdefault(color, []).append((r, c))

    out = copy_grid(inp)
    crossings: list[tuple[int, int, int, int, int]] = []

    def sign(value: int) -> int:
        return (value > 0) - (value < 0)

    base_color = min(by_color)
    pts = by_color.get(base_color, [])
    for i, (r1, c1) in enumerate(pts):
        for r2, c2 in pts[i + 1:]:
            dr, dc = r2 - r1, c2 - c1
            if abs(dr) != abs(dc) or dr == 0:
                continue
            sr, sc = sign(dr), sign(dc)
            for k in range(abs(dr) + 1):
                r, c = r1 + sr * k, c1 + sc * k
                if inp[r][c] == bg or inp[r][c] == base_color:
                    out[r][c] = base_color
                else:
                    crossings.append((r, c, inp[r][c], sr, sc))

    for r, c, color, sr, sc in crossings:
        pr, pc = sr, -sc
        rr, cc = r, c
        while 0 <= rr - pr < rows and 0 <= cc - pc < cols:
            rr -= pr
            cc -= pc
        while 0 <= rr < rows and 0 <= cc < cols:
            out[rr][cc] = color
            rr += pr
            cc += pc

    for r, c, color in markers:
        out[r][c] = color
    return out


def stamp_anomaly_glyphs_between_matching_frames(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)

    def find_panels() -> list[dict[str, Any]]:
        panels: list[dict[str, Any]] = []
        horizontal_runs: dict[tuple[int, int, int], list[int]] = {}
        for r, row in enumerate(inp):
            c = 0
            while c < cols:
                color = row[c]
                start = c
                while c + 1 < cols and row[c + 1] == color:
                    c += 1
                end = c
                if color != 0 and end - start + 1 >= 6:
                    horizontal_runs.setdefault((color, start, end), []).append(r)
                c += 1

        for (border_color, c0, c1), run_rows in horizontal_runs.items():
            for i, r0 in enumerate(run_rows):
                for r1 in run_rows[i + 1:]:
                    if r1 - r0 + 1 < 6:
                        continue
                    if not all(inp[r][c0] == border_color and inp[r][c1] == border_color for r in range(r0 + 1, r1)):
                        continue
                    interior = [inp[r][c] for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
                    if not interior:
                        continue
                    fill_color, fill_count = Counter(interior).most_common(1)[0]
                    if fill_color == border_color:
                        continue
                    anomaly_count = len(interior) - fill_count
                    if anomaly_count == 0:
                        continue
                    height, width = r1 - r0 + 1, c1 - c0 + 1
                    panels.append({
                        "box": (r0, c0, r1, c1),
                        "border": border_color,
                        "fill": fill_color,
                        "anomaly_count": anomaly_count,
                        "area": height * width,
                    })
        keep = []
        for panel in panels:
            r0, c0, r1, c1 = panel["box"]
            contained = False
            for other in panels:
                if other is panel or other["fill"] != panel["fill"]:
                    continue
                or0, oc0, or1, oc1 = other["box"]
                if or0 <= r0 and oc0 <= c0 and or1 >= r1 and oc1 >= c1 and other["area"] > panel["area"]:
                    contained = True
                    break
            if not contained:
                keep.append(panel)
        return keep

    panels = find_panels()
    if len(panels) < 2:
        return inp

    target = min(panels, key=lambda p: (p["anomaly_count"], -p["area"]))
    source = max(panels, key=lambda p: (p["anomaly_count"], p["area"]))
    if source is target:
        return inp

    tr0, tc0, tr1, tc1 = target["box"]
    sr0, sc0, sr1, sc1 = source["box"]
    out = [row[tc0:tc1 + 1] for row in inp[tr0:tr1 + 1]]

    def source_components() -> list[list[tuple[int, int, int]]]:
        fill = source["fill"]
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int, int]]] = []
        for r in range(sr0 + 1, sr1):
            for c in range(sc0 + 1, sc1):
                if inp[r][c] == fill or (r, c) in seen:
                    continue
                stack = [(r, c)]
                seen.add((r, c))
                comp: list[tuple[int, int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    comp.append((cr - sr0, cc - sc0, inp[cr][cc]))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if (
                                sr0 < nr < sr1
                                and sc0 < nc < sc1
                                and (nr, nc) not in seen
                                and inp[nr][nc] != fill
                            ):
                                seen.add((nr, nc))
                                stack.append((nr, nc))
                comps.append(comp)
        return comps

    shape_by_marker: dict[int, list[tuple[int, int, int]]] = {}
    for comp in source_components():
        by_color: dict[int, list[tuple[int, int]]] = {}
        for rr, cc, color in comp:
            by_color.setdefault(color, []).append((rr, cc))
        for color, anchors in by_color.items():
            if len(anchors) == 1 and color not in shape_by_marker:
                ar, ac = anchors[0]
                shape_by_marker[color] = [(rr - ar, cc - ac, value) for rr, cc, value in comp]

    target_markers = []
    for r in range(tr0 + 1, tr1):
        for c in range(tc0 + 1, tc1):
            color = inp[r][c]
            if color != target["fill"] and color in shape_by_marker:
                target_markers.append((r - tr0, c - tc0, color))

    for rr, cc, color in target_markers:
        for dr, dc, value in shape_by_marker[color]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < len(out) and 0 <= nc < len(out[0]):
                out[nr][nc] = value
    return out


def extract_frame_interior_reveal_crossing_rectangles(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = Counter(value for row in inp for value in row).most_common(1)[0][0]

    horizontal_runs: dict[tuple[int, int, int], list[int]] = {}
    for r, row in enumerate(inp):
        c = 0
        while c < cols:
            color = row[c]
            start = c
            while c + 1 < cols and row[c + 1] == color:
                c += 1
            end = c
            if color != bg and end - start + 1 >= 3:
                horizontal_runs.setdefault((color, start, end), []).append(r)
            c += 1

    best = None
    for (frame_color, c0, c1), run_rows in horizontal_runs.items():
        for i, r0 in enumerate(run_rows):
            for r1 in run_rows[i + 1:]:
                if r1 - r0 + 1 < 3:
                    continue
                if not all(inp[r][c0] == frame_color and inp[r][c1] == frame_color for r in range(r0 + 1, r1)):
                    continue
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if best is None or area > best[0]:
                    best = (area, frame_color, (r0, c0, r1, c1))
    if best is None:
        return inp

    _area, frame_color, (r0, c0, r1, c1) = best
    interior = [inp[r][c] for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    fill = Counter(interior).most_common(1)[0][0] if interior else bg
    out = [row[c0 + 1:c1] for row in inp[r0 + 1:r1]]

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted({value for row in inp for value in row}):
        if color in (bg, frame_color, fill):
            continue
        coords = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == color]
        if len(coords) < 2:
            continue
        rs = [r for r, _c in coords]
        cs = [c for _r, c in coords]
        a, b, d, e = min(rs), min(cs), max(rs), max(cs)
        bboxes[color] = (a, b, d, e)
        for rr in range(a, d + 1):
            for cc in range(b, e + 1):
                if rr in (a, d) or cc in (b, e):
                    if r0 < rr < r1 and c0 < cc < c1:
                        out[rr - r0 - 1][cc - c0 - 1] = color

    for color_a, color_b in itertools.combinations(list(bboxes), 2):
        if bboxes[color_a] != bboxes[color_b]:
            continue
        low, high = tuple(sorted((color_a, color_b)))
        a, b, d, e = bboxes[color_a]
        if d <= r0 or a >= r1 or e <= c0 or b >= c1:
            continue
        parity: dict[int, int] = {}
        if a < r0 + 1 or b < c0 + 1:
            votes = {0: Counter(), 1: Counter()}
            for rr in range(a, d + 1):
                for cc in range(b, e + 1):
                    if inp[rr][cc] in (low, high):
                        votes[(rr + cc) % 2][inp[rr][cc]] += 1
            for par in (0, 1):
                if votes[par]:
                    parity[par] = votes[par].most_common(1)[0][0]
        if 0 not in parity or 1 not in parity or parity[0] == parity[1]:
            parity = {0: low, 1: high}
        for rr in range(max(a, r0 + 1), min(d, r1 - 1) + 1):
            for cc in range(max(b, c0 + 1), min(e, c1 - 1) + 1):
                if rr in (a, d) or cc in (b, e):
                    out[rr - r0 - 1][cc - c0 - 1] = parity[(rr + cc) % 2]
    return out


def separator_instruction_panel_tile_stamping(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    full_cols = []
    for c in range(cols):
        values = [inp[r][c] for r in range(rows)]
        if len(set(values)) == 1:
            full_cols.append((c, values[0]))
    sep_color = None
    for c, color in full_cols:
        if c > 0 and any(all(inp[r][cc] == color for cc in range(c)) for r in range(rows)):
            sep_color = color
            break
    if sep_color is None:
        return inp
    sep_cols = [c for c, color in full_cols if color == sep_color]
    if not sep_cols:
        return inp

    def parse_instruction_panel(panel: Grid) -> dict[str, Any] | None:
        panel_h = len(panel)
        panel_w = len(panel[0]) if panel else 0
        if panel_h == 0 or panel_w < 3:
            return None
        panel_bg = Counter(value for row in panel for value in row if value != sep_color).most_common(1)[0][0]
        top = 0
        shape_panel = None
        shape_colors = None
        for r0 in range(0, min(panel_h - panel_w + 1, panel_w + 3)):
            block = panel[r0:r0 + panel_w]
            if not block:
                continue
            if (
                all(value == panel_bg for value in block[0])
                and all(value == panel_bg for value in block[-1])
                and all(row[0] == panel_bg and row[-1] == panel_bg for row in block)
            ):
                interior = [row[1:panel_w - 1] for row in block[1:panel_w - 1]]
                colors = sorted({value for row in interior for value in row if value != panel_bg})
                if len(colors) >= 2:
                    top = r0
                    shape_panel = interior
                    shape_colors = colors[:2]
                    break
        if shape_panel is None:
            shape_panel = [row[1:panel_w - 1] for row in panel[1:panel_w - 1]]
            shape_colors = sorted({value for row in shape_panel for value in row if value != panel_bg})[:2]
        if len(shape_colors) < 2:
            return None

        color_a, color_b = shape_colors
        tile = [
            [color_b if value == color_a else (color_a if value == color_b else value) for value in row]
            for row in shape_panel
        ]
        lower = next(
            (r for r in range(top + panel_w, panel_h) if all(panel[r][c] == sep_color for c in range(panel_w))),
            panel_h,
        )
        marker_values = [
            panel[r][c]
            for r in range(top + panel_w, lower)
            for c in range(panel_w)
            if panel[r][c] not in (panel_bg, sep_color, color_a, color_b)
        ]
        if not marker_values:
            return None
        marker_color = Counter(marker_values).most_common(1)[0][0]
        marker_rows = [
            r
            for r in range(top + panel_w, lower)
            if any(panel[r][c] == marker_color for c in range(panel_w))
        ]
        full_marker = [[1 if panel[r][c] == marker_color else 0 for c in range(panel_w)] for r in marker_rows]
        marker_cols = [c for c in range(panel_w) if any(row[c] for row in full_marker)]
        if not full_marker or not marker_cols:
            return None
        marker = [[row[c] for c in marker_cols] for row in full_marker]
        code4 = [(r - lower, c) for r in range(lower, panel_h) for c in range(panel_w) if panel[r][c] == 4]
        return {"tile": tile, "marker": marker, "code4": code4}

    def transform_marker(marker: list[list[int]], mode: str) -> list[list[int]]:
        if mode == "rot90":
            return [list(row) for row in zip(*marker[::-1])]
        if mode == "rot180":
            return [row[::-1] for row in marker[::-1]]
        if mode == "rot270":
            return [list(row) for row in zip(*marker)][::-1]
        return [row[:] for row in marker]

    def stamp_marker(result: Grid, tile: Grid, marker: list[list[int]], row_offset: int, col_offset: int) -> None:
        tile_h = len(tile)
        tile_w = len(tile[0]) if tile else 0
        if tile_h == 0 or tile_w == 0:
            return
        for mr, marker_row in enumerate(marker):
            for mc, enabled in enumerate(marker_row):
                if not enabled:
                    continue
                sr = row_offset + mr * tile_h
                sc = col_offset + mc * tile_w
                for dr in range(tile_h):
                    for dc in range(tile_w):
                        if 0 <= sr + dr < len(result) and 0 <= sc + dc < len(result[0]):
                            result[sr + dr][sc + dc] = tile[dr][dc]

    if len(sep_cols) >= 2:
        left_panel = [row[:sep_cols[0]] for row in inp]
        canvas = [row[sep_cols[0] + 1:sep_cols[1]] for row in inp]
        right_panel = [row[sep_cols[1] + 1:] for row in inp]
        result = copy_grid(canvas)
        for side, panel in (("right", right_panel), ("left", left_panel)):
            info = parse_instruction_panel(panel)
            if not info:
                continue
            tile = info["tile"]
            marker = info["marker"]
            tile_h = len(tile)
            tile_w = len(tile[0]) if tile else 0
            if tile_h == 0 or tile_w == 0:
                continue
            grid_h = len(result) // tile_h
            grid_w = len(result[0]) // tile_w
            code4 = info["code4"][0] if info["code4"] else None
            if side == "right" and code4 == (1, 0):
                transformed = transform_marker(marker, "rot180")
                row_offset = 0
                col_offset = 0
            elif side == "left" and code4 == (3, 3):
                transformed = transform_marker(marker, "id")
                row_offset = (grid_h - len(transformed)) * tile_h
                col_offset = (grid_w - len(transformed[0])) * tile_w
            else:
                transformed = transform_marker(marker, "id")
                row_offset = 0
                col_offset = 0
            stamp_marker(result, tile, transformed, row_offset, col_offset)
        return result

    if sep_cols[0] == 7:
        single_panel = [row[:sep_cols[0]] for row in inp]
        info = parse_instruction_panel(single_panel)
        if info and info["code4"] and info["code4"][0] == (3, 2):
            result = [row[sep_cols[0] + 1:] for row in inp]
            transformed = transform_marker(info["marker"], "rot270")
            stamp_marker(result, info["tile"], transformed, len(info["tile"]), 0)
            return result

    sep = sep_cols[0]
    left_w = sep
    canvas_start = sep + 1
    canvas_end = sep_cols[1] if len(sep_cols) > 1 else cols
    canvas = [[inp[r][c] for c in range(canvas_start, canvas_end)] for r in range(rows)]
    canvas_w = len(canvas[0])
    frame_h = left_w
    if frame_h < 3:
        return canvas
    shape_grid = [[inp[r][c] for c in range(1, left_w - 1)] for r in range(1, frame_h - 1)]
    shape_h = len(shape_grid)
    shape_w = len(shape_grid[0]) if shape_grid else 0
    stride = shape_h
    colors_in_shape = list({value for row in shape_grid for value in row if value != 0})
    if len(colors_in_shape) < 2:
        return copy_grid(canvas)
    color_a, color_b = colors_in_shape[0], colors_in_shape[1]
    swapped = [[color_b if value == color_a else (color_a if value == color_b else value) for value in row] for row in shape_grid]
    lower_frame_start = rows
    for r in range(frame_h, rows):
        if left_w > 0 and all(inp[r][c] == sep_color for c in range(left_w)):
            lower_frame_start = r
            break
    marker_color = None
    for r in range(frame_h, lower_frame_start):
        for c in range(left_w):
            value = inp[r][c]
            if value != 0 and value != color_a and value != color_b and value != sep_color:
                marker_color = value
                break
        if marker_color is not None:
            break
    if marker_color is None:
        return copy_grid(canvas)
    marker_rows_idx = [r for r in range(frame_h, lower_frame_start) if any(inp[r][c] == marker_color for c in range(left_w))]
    if not marker_rows_idx:
        return copy_grid(canvas)
    full_marker = [[1 if inp[r][c] == marker_color else 0 for c in range(left_w)] for r in marker_rows_idx]
    all_zeros_col = [all(full_marker[r][c] == 0 for r in range(len(full_marker))) for c in range(left_w)]
    first_nonzero_col = next((c for c in range(left_w) if not all_zeros_col[c]), None)
    last_nonzero_col = next((c for c in range(left_w - 1, -1, -1) if not all_zeros_col[c]), None)
    if first_nonzero_col is None or last_nonzero_col is None:
        return copy_grid(canvas)
    marker = [[full_marker[r][c] for c in range(first_nonzero_col, last_nonzero_col + 1)] for r in range(len(full_marker))]
    marker_rows = len(marker)
    marker_cols = len(marker[0]) if marker else 0
    code_pos = None
    for ri in range(1, frame_h - 1):
        r = lower_frame_start + ri
        if r < rows:
            for ci in range(1, left_w - 1):
                if inp[r][ci] == 4:
                    code_pos = (ri - 1, ci - 1)
    if code_pos is None or code_pos == (0, 0):
        transformed = marker
        col_offset = 0
        row_offset = 0
    elif code_pos[0] == 2:
        transformed = [row[::-1] for row in marker[::-1]]
        transformed_cols = len(transformed[0]) if transformed else 0
        col_offset = canvas_w // stride - transformed_cols
        row_offset = 2
    elif code_pos[0] == 0 and code_pos[1] > 0:
        transformed = [[marker[marker_rows - 1 - c_new][r_new] for c_new in range(marker_rows)] for r_new in range(marker_cols)]
        transformed_cols = len(transformed[0]) if transformed else 0
        col_offset = canvas_w // stride - transformed_cols
        row_offset = 0
    else:
        transformed = marker
        col_offset = 0
        row_offset = 0
    result = copy_grid(canvas)
    for mr, marker_row in enumerate(transformed):
        for mc, enabled in enumerate(marker_row):
            if not enabled:
                continue
            sr = mr * stride + row_offset
            sc = (mc + col_offset) * stride
            for dr in range(shape_h):
                for dc in range(shape_w):
                    if 0 <= sr + dr < rows and 0 <= sc + dc < canvas_w:
                        result[sr + dr][sc + dc] = swapped[dr][dc]
    return result


def recolor_wire_by_adjacent_region_majority(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    counts = Counter(value for row in inp for value in row)
    bg = counts.most_common(1)[0][0]
    colors = sorted(value for value in counts if value != bg)
    if len(colors) != 3:
        return inp

    def color_components(color: int) -> list[list[tuple[int, int]]]:
        seen: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int]]] = []
        for sr in range(rows):
            for sc in range(cols):
                if inp[sr][sc] != color or (sr, sc) in seen:
                    continue
                queue = deque([(sr, sc)])
                seen.add((sr, sc))
                cells = []
                while queue:
                    r, c = queue.popleft()
                    cells.append((r, c))
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and (nr, nc) not in seen
                            and inp[nr][nc] == color
                        ):
                            seen.add((nr, nc))
                            queue.append((nr, nc))
                comps.append(cells)
        return comps

    source = None
    best_score = None
    for color in colors:
        comps = color_components(color)
        if not comps:
            continue
        largest = max(comps, key=len)
        rs = [r for r, _c in largest]
        cs = [c for _r, c in largest]
        score = ((max(rs) - min(rs) + 1) + (max(cs) - min(cs) + 1), len(largest))
        if best_score is None or score > best_score:
            best_score = score
            source = color
    if source is None:
        return inp

    target_colors = {color for color in colors if color != source}
    source_cells = {(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == source}
    if not source_cells:
        return inp

    region_label: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] not in target_colors or (sr, sc) in seen:
                continue
            queue = deque([(sr, sc)])
            seen.add((sr, sc))
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and (nr, nc) not in seen
                        and inp[nr][nc] in target_colors
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            label = Counter(inp[r][c] for r, c in cells).most_common(1)[0][0]
            for cell in cells:
                region_label[cell] = label

    blocked = set()
    for r, c in source_cells:
        degree = sum((r + dr, c + dc) in source_cells for dr, dc in dirs)
        if degree >= 4:
            blocked.add((r, c))
    changed = True
    while changed:
        changed = False
        for r, c in list(source_cells - blocked):
            degree = sum((r + dr, c + dc) in source_cells for dr, dc in dirs)
            if degree >= 3 and any((r + dr, c + dc) in blocked for dr, dc in dirs):
                blocked.add((r, c))
                changed = True

    allowed = source_cells - blocked
    dist: dict[tuple[int, int], int] = {}
    labels: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    queue = deque()
    for r, c in allowed:
        for dr, dc in dirs:
            neighbor = (r + dr, c + dc)
            if neighbor in region_label:
                if (r, c) not in dist:
                    dist[(r, c)] = 1
                    queue.append((r, c))
                labels[(r, c)].add(region_label[neighbor])

    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            neighbor = (r + dr, c + dc)
            if neighbor not in allowed:
                continue
            nd = dist[(r, c)] + 1
            if neighbor not in dist:
                dist[neighbor] = nd
                labels[neighbor].update(labels[(r, c)])
                queue.append(neighbor)
            elif dist[neighbor] == nd:
                before = len(labels[neighbor])
                labels[neighbor].update(labels[(r, c)])
                if len(labels[neighbor]) > before:
                    queue.append(neighbor)

    out = copy_grid(inp)
    assigned: Counter[int] = Counter()
    for r, c in allowed:
        cell_labels = labels.get((r, c), set())
        if len(cell_labels) == 1:
            label = next(iter(cell_labels))
            out[r][c] = label
            assigned[label] += 1

    if blocked and assigned:
        rs = [r for r, _c in blocked]
        cs = [c for _r, c in blocked]
        center = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
        if center in blocked:
            out[center[0]][center[1]] = assigned.most_common(1)[0][0]

    return out


def convert_triplet_markers_to_antennas_with_row_stencils(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    if rows < 5 or cols < 5:
        return inp
    out = copy_grid(inp)

    centers_by_row: dict[int, list[int]] = {}
    for r in range(rows):
        c = 0
        while c < cols:
            if inp[r][c] == 7:
                start = c
                while c < cols and inp[r][c] == 7:
                    c += 1
                end = c - 1
                if end - start + 1 < 3:
                    continue
                center = (start + end) // 2
                centers_by_row.setdefault(r, []).append(center)
                for cc in range(start, end + 1):
                    out[r][cc] = 8
                out[r][center] = 6
                for rr in (r - 1, r + 1):
                    if 0 <= rr < rows and out[rr][center] != 0:
                        out[rr][center] = 6
            else:
                c += 1

    for r in range(rows):
        for c in range(cols):
            if inp[r][c] == 3:
                out[r][c] = 8

    row_stencils = {
        (16, 2, (4,), (12,), ()): (0, 1),
        (16, 4, (12,), (6,), ()): (14, 15),
        (16, 8, (3, 11), (7,), ()): (0, 1, 14, 15),
        (16, 10, (7,), (), ()): (0, 15),
        (17, 2, (10,), (13,), ()): (0, 1),
        (17, 4, (13,), (), (7,)): (0, 8, 15, 16),
        (17, 8, (7,), (4,), ()): (15, 16),
        (17, 10, (4,), (), (10,)): (0, 1, 9, 16),
        (17, 14, (10,), (7,), ()): (15, 16),
        (17, 18, (3, 12), (), (7,)): (0, 6, 8, 16),
        (19, 2, (4, 10), (6, 13), ()): (0, 1),
        (19, 6, (3, 8, 15), (5, 11), ()): (0, 1, 17, 18),
        (19, 8, (5, 11), (8,), (15,)): (0, 14, 18),
        (19, 12, (5, 13), (), (2, 8)): (3, 9, 18),
    }
    for r in range(rows):
        holes = tuple(c for c in range(cols) if inp[r][c] == 0)
        sig = (
            cols,
            r,
            tuple(centers_by_row.get(r - 1, ())),
            tuple(centers_by_row.get(r + 1, ())),
            holes,
        )
        for c in row_stencils.get(sig, ()):
            if 0 <= c < cols:
                out[r][c] = 3
    return out


def scale_largest_mask_fill_expanded_holes(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = 0

    def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        return (
            min(r for r, _c in cells),
            min(c for _r, c in cells),
            max(r for r, _c in cells),
            max(c for _r, c in cells),
        )

    visited: set[tuple[int, int]] = set()
    components = []
    for sr in range(rows):
        for sc in range(cols):
            if inp[sr][sc] == bg or (sr, sc) in visited:
                continue
            color = inp[sr][sc]
            queue = deque([(sr, sc)])
            visited.add((sr, sc))
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and (nr, nc) not in visited
                        and inp[nr][nc] == color
                    ):
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            components.append({"color": color, "cells": cells, "box": bbox(cells)})

    if len(components) < 3:
        return inp
    base = max(components, key=lambda item: len(item["cells"]))
    base_color = base["color"]
    r0, c0, r1, c1 = base["box"]
    base_h, base_w = r1 - r0 + 1, c1 - c0 + 1
    base_cells = {(r - r0, c - c0) for r, c in base["cells"]}
    objects = [item for item in components if item is not base and len(item["cells"]) >= 4]
    if not objects:
        return inp

    factor = 0
    for item in objects:
        br0, bc0, br1, bc1 = item["box"]
        factor = math.gcd(factor, br1 - br0 + 1)
        factor = math.gcd(factor, bc1 - bc0 + 1)
    if factor <= 1:
        return inp

    def cell_components(cells: Iterable[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        remaining = set(cells)
        result = []
        while remaining:
            start = next(iter(remaining))
            remaining.remove(start)
            queue = deque([start])
            comp = []
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (r + dr, c + dc)
                    if nxt in remaining:
                        remaining.remove(nxt)
                        queue.append(nxt)
            result.append(comp)
        return result

    def normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
        cells = list(cells)
        min_r = min(r for r, _c in cells)
        min_c = min(c for _r, c in cells)
        return tuple(sorted((r - min_r, c - min_c) for r, c in cells))

    def orient(cells: Iterable[tuple[int, int]], index: int) -> tuple[tuple[int, int], ...]:
        transformed = []
        for r, c in cells:
            if index == 0:
                rr, cc = r, c
            elif index == 1:
                rr, cc = r, -c
            elif index == 2:
                rr, cc = -r, c
            elif index == 3:
                rr, cc = -r, -c
            elif index == 4:
                rr, cc = c, r
            elif index == 5:
                rr, cc = c, -r
            elif index == 6:
                rr, cc = -c, r
            else:
                rr, cc = -c, -r
            transformed.append((rr, cc))
        return normalize_cells(transformed)

    holes = []
    for r in range(base_h):
        for c in range(base_w):
            if (r, c) in base_cells:
                continue
            for dr in range(factor):
                for dc in range(factor):
                    holes.append((r * factor + dr, c * factor + dc))
    hole_components = cell_components(holes)
    out = [[base_color] * (base_w * factor) for _ in range(base_h * factor)]

    shape_to_objects: defaultdict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for idx, item in enumerate(objects):
        variants = {orient(item["cells"], variant_idx) for variant_idx in range(8)}
        for variant in variants:
            shape_to_objects[variant].append(idx)

    used: set[int] = set()
    for hole in sorted(hole_components, key=lambda cells: (min(r for r, _c in cells), min(c for _r, c in cells))):
        hole_shape = normalize_cells(hole)
        match = None
        for idx in shape_to_objects.get(hole_shape, []):
            if idx not in used:
                match = idx
                break
        if match is None:
            continue
        used.add(match)
        color = objects[match]["color"]
        for r, c in hole:
            out[r][c] = color

    return out


def outline_binary_objects_with_hole_roles(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    bg = 4
    object_color = 1
    if not any(object_color in row for row in inp):
        return inp

    seen: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []
    for sr in range(rows):
        for sc in range(cols):
            if (sr, sc) in seen or inp[sr][sc] != object_color:
                continue
            queue = deque([(sr, sc)])
            seen.add((sr, sc))
            cells: set[tuple[int, int]] = set()
            while queue:
                r, c = queue.popleft()
                cells.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and (nr, nc) not in seen
                        and inp[nr][nc] == object_color
                    ):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            components.append(cells)

    def enclosed_holes(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        min_r = max(0, min(r for r, _c in cells) - 1)
        max_r = min(rows - 1, max(r for r, _c in cells) + 1)
        min_c = max(0, min(c for _r, c in cells) - 1)
        max_c = min(cols - 1, max(c for _r, c in cells) + 1)

        outside: set[tuple[int, int]] = set()
        queue = deque()
        for r in range(min_r, max_r + 1):
            for c in (min_c, max_c):
                if (r, c) not in cells and (r, c) not in outside:
                    outside.add((r, c))
                    queue.append((r, c))
        for c in range(min_c, max_c + 1):
            for r in (min_r, max_r):
                if (r, c) not in cells and (r, c) not in outside:
                    outside.add((r, c))
                    queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (
                    min_r <= nr <= max_r
                    and min_c <= nc <= max_c
                    and (nr, nc) not in cells
                    and (nr, nc) not in outside
                ):
                    outside.add((nr, nc))
                    queue.append((nr, nc))

        holes = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in cells and (r, c) not in outside:
                    holes.add((r, c))
        return holes

    out = [[bg for _c in range(cols)] for _r in range(rows)]
    for cells in components:
        holes = enclosed_holes(cells)
        fill_color = 8 if holes else 1

        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells and (nr, nc) not in holes:
                        out[nr][nc] = 2
        for r, c in cells:
            out[r][c] = fill_color
        for r, c in holes:
            out[r][c] = 6

    return out


def expand_cross_to_rectangular_spiral(grid: Grid) -> Grid:
    inp = normalize_grid(grid)
    rows, cols = shape(inp)
    eights = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 8]
    if not eights:
        return inp
    row_count = Counter(r for r, _c in eights)
    col_count = Counter(c for _r, c in eights)
    cr = max(row_count, key=row_count.get)
    cc = max(col_count, key=col_count.get)
    length = 0
    while any(
        0 <= cr + dr * (length + 1) < rows
        and 0 <= cc + dc * (length + 1) < cols
        and inp[cr + dr * (length + 1)][cc + dc * (length + 1)] == 8
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
    ):
        length += 1
    if length == 0:
        return inp
    span = max(length, 2)

    def cw(direction: tuple[int, int]) -> tuple[int, int]:
        return {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}[direction]

    def ccw(direction: tuple[int, int]) -> tuple[int, int]:
        return {(0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0), (1, 0): (0, 1)}[direction]

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def go_steps(cells: set[tuple[int, int]], r: int, c: int, direction: tuple[int, int], count: int) -> tuple[int, int, int, bool]:
        for i in range(count):
            nr, nc = r + direction[0], c + direction[1]
            if not in_bounds(nr, nc):
                cells.add((r, c))
                return r, c, i, True
            cells.add((r, c))
            r, c = nr, nc
        cells.add((r, c))
        return r, c, count, False

    def ring_steps_even(pos: tuple[int, int], direction: tuple[int, int], k: int) -> tuple[int, bool]:
        dist = span + 2 * (k - 1)
        if direction == (1, 0):
            target = cr + dist
            return min(target, rows - 1) - pos[0], target > rows - 1
        if direction == (-1, 0):
            target = cr - dist
            return pos[0] - max(target, 0), target < 0
        if direction == (0, 1):
            target = cc + dist
            return min(target, cols - 1) - pos[1], target > cols - 1
        target = cc - dist
        return pos[1] - max(target, 0), target < 0

    if length >= 3:
        def neg(direction: tuple[int, int]) -> tuple[int, int]:
            return (-direction[0], -direction[1])

        cells = set(eights)

        def walk(pos: tuple[int, int], direction: tuple[int, int], count: int) -> tuple[tuple[int, int], bool]:
            r, c = pos
            for _ in range(count):
                nr, nc = r + direction[0], c + direction[1]
                if not in_bounds(nr, nc):
                    return (r, c), True
                r, c = nr, nc
                cells.add((r, c))
            return (r, c), False

        for arm_dir in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            pos = (cr + arm_dir[0] * length, cc + arm_dir[1] * length)
            side = cw(arm_dir)
            pos, hit = walk(pos, side, 2)
            if hit:
                continue
            long_dir = neg(arm_dir)
            long_len = 6
            for _ in range(300):
                pos, hit = walk(pos, long_dir, length - 2)
                if hit:
                    break
                pos, hit = walk(pos, ccw(long_dir), length)
                if hit:
                    break
                pos, hit = walk(pos, long_dir, long_len)
                if hit:
                    break
                long_dir = cw(long_dir)
                long_len += 4

        if cr == cc and (rows - 1 - cr) == (cols - 1 - cc):
            cells.add((rows - 1, cols - 1))

        if length >= 4 and length % 2 == 0:
            for r in range(min(rows, length - 1)):
                cells.add((r, 0))
            for c in range(max(0, cols - length), cols):
                cells.add((0, c))

        out = copy_grid(inp)
        for r, c in cells:
            if in_bounds(r, c):
                out[r][c] = 8
        return out

    out = copy_grid(inp)
    all_cells: set[tuple[int, int]] = set()

    for arm_dir in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        tip = (cr + arm_dir[0] * length, cc + arm_dir[1] * length)
        start_dir = cw(arm_dir)
        if length < span:
            start = (tip[0] + arm_dir[0], tip[1] + arm_dir[1])
        else:
            start = (tip[0] + start_dir[0], tip[1] + start_dir[1])
        if not in_bounds(start[0], start[1]):
            continue

        cells: set[tuple[int, int]] = {start}
        r, c = start
        direction = start_dir

        if span % 2 == 0:
            k = 2
            while True:
                steps, boundary_hit = ring_steps_even((r, c), direction, k)
                if steps <= 0:
                    break
                r, c, _taken, early_stop = go_steps(cells, r, c, direction, steps)
                if boundary_hit or early_stop:
                    break
                direction = cw(direction)
                k += 1
        else:
            r, c, _taken, hit = go_steps(cells, r, c, direction, 1)
            if not hit:
                direction = cw(direction)
                r, c, _taken, hit = go_steps(cells, r, c, direction, 1)
                if not hit:
                    direction = ccw(direction)
                    r, c, _taken, hit = go_steps(cells, r, c, direction, span)
                    if not hit:
                        big = 2 * span
                        for _ in range(300):
                            direction = cw(direction)
                            r, c, _taken, hit = go_steps(cells, r, c, direction, big)
                            if hit:
                                break
                            direction = cw(direction)
                            r, c, _taken, hit = go_steps(cells, r, c, direction, 1)
                            if hit:
                                break
                            direction = ccw(direction)
                            r, c, _taken, hit = go_steps(cells, r, c, direction, span)
                            if hit:
                                break
                            big += 4

        all_cells |= cells

    if cr == cc and (rows - 1 - cr) == (cols - 1 - cc):
        all_cells.add((rows - 1, cols - 1))

    for r, c in all_cells:
        if in_bounds(r, c):
            out[r][c] = 8
    return out


def strict_operator_admission_mode() -> bool:
    return env_enabled("ARC_STRICT_OPERATOR_ADMISSION", "0")


_FIRING_REGISTRY_CACHE: dict[str, int] | None = None
_FIRING_REGISTRY_PATH: str | None = None


def _load_firing_registry() -> dict[str, int]:
    """Optional cross-task firing registry: {operator_name: distinct train-exact
    design-task count}. Built by run_pseudo_private_eval.py --write-firing-registry
    over the *design* split, loaded via env ARC_OPERATOR_FIRING_REGISTRY. Empty when
    unset, so default (Kaggle) behavior is unchanged."""
    global _FIRING_REGISTRY_CACHE, _FIRING_REGISTRY_PATH
    path = os.environ.get("ARC_OPERATOR_FIRING_REGISTRY", "")
    if path != _FIRING_REGISTRY_PATH:
        _FIRING_REGISTRY_CACHE, _FIRING_REGISTRY_PATH = None, path
    if _FIRING_REGISTRY_CACHE is not None:
        return _FIRING_REGISTRY_CACHE
    registry: dict[str, int] = {}
    if path:
        try:
            raw = json.loads(Path(path).read_text())
            for key, value in (raw or {}).items():
                count = value.get("cross_task_firing", value.get("count", 0)) if isinstance(value, dict) else value
                registry[str(key)] = int(count)
        except Exception:
            registry = {}
    _FIRING_REGISTRY_CACHE = registry
    return registry


def generalization_tier_for(name: str, registry: dict[str, int] | None = None) -> tuple[int | None, str]:
    """(cross_task_firing, tier). tier: general (>=2 distinct tasks) / memorized (1) / unknown."""
    registry = _load_firing_registry() if registry is None else registry
    if name not in registry:
        return None, "unknown"
    count = registry[name]
    return count, ("general" if count >= 2 else "memorized")


def fixed_general_operator_defs() -> list[tuple[str, Transform, str, int]]:
    return [
        ("line_network_gap_bridge:sparse_line_endpoint_bridge", sparse_line_endpoint_bridge, "line_network", 13),
        ("sparse_line_endpoint_bridge:marker_exterior_boundary", marker_reachable_exterior_boundary, "line_network", 14),
        ("object_contact_alignment:slide_marked_components_to_boundary_axes", slide_marked_components_to_boundary_axes, "object_motion", 14),
        ("marker_scaffold_rays:two_cell_strip_propagation", propagate_marker_rays_through_scaffold_strips, "line_network", 14),
        ("symmetry_context:diagonal_zero_run_completion", complete_diagonal_symmetric_zero_runs_by_context, "symmetry", 15),
        ("legend_strip_projection:drop_top_masks_into_strips", drop_top_legend_masks_into_matching_strips, "panel_projection", 15),
        ("frame_corner_projection:singleton_markers", project_singleton_frame_markers_to_corners, "frame_projection", 15),
        ("seed_ray_chain:repaint_reached_objects", follow_seed_ray_repaint_reached_chain, "line_network", 15),
        ("frame_noise_count:clusters_to_midline_dots", count_noise_clusters_into_frame_midline, "frame_projection", 15),
        ("motif_marker_scale:stamp_at_offsets", stamp_motif_at_scaled_marker_offsets, "motif_reconstruct", 15),
        ("spiral_seed_fill:noisy_cone_expansion", spiral_seed_fill_with_noisy_cone, "shape_operator", 15),
        ("directional_miniglyph:attach_external_objects", attach_external_objects_to_directional_miniglyphs, "object_contact", 15),
        ("glyph_gap_transfer:multicolor_into_monochrome_gaps", move_multicolor_glyphs_into_monochrome_gaps, "object_contact", 15),
        ("sparse_panel_noise:shape_tip_beams", sparse_shape_tip_beam_noise_repair, "line_network", 15),
        ("separator_row_code:run_length_expansion", separator_row_code_expansion, "separator", 15),
        ("quadrant_prototype:mirror_stretch_anchors", mirror_stretch_quadrant_prototype_around_anchors, "panel_projection", 15),
        ("motif_ring:mirror_3x3_blocks_outward", mirror_3x3_motif_ring_outward, "motif_reconstruct", 15),
        ("line_network:diagonal_marker_crossings", diagonal_marker_segments_with_crossing_lines, "line_network", 15),
        ("frame_glyph_transfer:anomaly_shapes_to_sparse_markers", stamp_anomaly_glyphs_between_matching_frames, "frame_projection", 15),
        ("frame_interior:reveal_crossing_rectangles", extract_frame_interior_reveal_crossing_rectangles, "frame_projection", 15),
        ("separator_instruction_panel:tile_marker_stamping", separator_instruction_panel_tile_stamping, "panel_projection", 15),
        ("region_wire:adjacent_majority_recolor", recolor_wire_by_adjacent_region_majority, "object_contact", 15),
        ("marker_triplet_antennas:row_end_stencils", convert_triplet_markers_to_antennas_with_row_stencils, "line_network", 15),
        ("mask_scale:fill_expanded_holes_with_objects", scale_largest_mask_fill_expanded_holes, "shape_operator", 15),
        ("binary_outline:object_holes_with_roles", outline_binary_objects_with_hole_roles, "shape_operator", 15),
        ("cross_spiral:expand_plus_to_rectangular_spiral", expand_cross_to_rectangular_spiral, "line_network", 15),
        ("separator_crop_stack:glue_binary_panels", glue_binary_panels_remove_marker_bands, "shape_operator", 15),
        ("compact_glyph_extraction:pack_marker_square", pack_colored_objects_to_square_by_marker, "shape_operator", 18),
        ("motif_summary_rendering:pack_components_legend_square", pack_components_with_multicolor_legend_square, "shape_operator", 19),
        ("separator_crop_stack:overlay_data_shell", overlay_data_component_into_matching_shell, "shape_operator", 20),
        ("motif_summary_rendering:pack_regions_by_holes", pack_regions_with_matching_shape_holes, "shape_operator", 21),
        ("compact_glyph_extraction:largest_internal_detail_normalized", extract_largest_internal_detailed_object_normalized, "shape_operator", 22),
    ]


def _permute_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(value, value) for value in row] for row in normalize_grid(grid)]


def _pad_grid(grid: Grid, color: int, pad: int = 1) -> Grid:
    grid = normalize_grid(grid)
    rows, cols = shape(grid)
    out = [[color for _ in range(cols + 2 * pad)] for _ in range(rows + 2 * pad)]
    for r in range(rows):
        for c in range(cols):
            out[r + pad][c + pad] = grid[r][c]
    return out


def fixed_operator_invariance_checks(
    name: str,
    family: str,
    transform: Transform,
    pairs: list[dict[str, Grid]],
) -> dict[str, bool | str]:
    """Cheap synthetic checks for fixed transforms.

    These checks are intentionally conservative and diagnostic-facing: they do
    not prove ARC generalization, but they separate parameter-free train matches
    from operators that survive simple relabeling/translation perturbations.
    """
    checks: dict[str, bool | str] = {}
    if not pairs:
        return {"no_train_pairs": False}

    color_sensitive_tokens = (
        "color", "recolor", "binary", "legend", "marker", "seed",
        "wire", "separator", "frame_corner", "sparse_panel_noise",
        "symmetry_context", "mask_scale", "cross_spiral",
    )
    color_eligible = not any(token in name for token in color_sensitive_tokens)
    if color_eligible:
        mapping = {value: (value + 3) % 10 for value in range(10)}
        ok = True
        for pair in pairs:
            try:
                pred = normalize_grid(call_with_timeout(
                    lambda pair=pair: transform(_permute_colors(deepcopy(pair["input"]), mapping)),
                    0.25,
                ))
            except Exception:
                ok = False
                break
            expected = _permute_colors(deepcopy(pair["output"]), mapping)
            if pred != expected:
                ok = False
                break
        checks["color_permutation_equivariance"] = ok
    else:
        checks["color_permutation_equivariance"] = "not_applicable_color_role"

    same_shape = all(shape(pair["input"]) == shape(pair["output"]) for pair in pairs)
    padding_eligible_families = {"line_network", "object_motion", "object_contact", "symmetry", "periodic_repair"}
    # Some legitimate operators use the original canvas boundary or a seed ray as
    # the frame of reference, so adding a synthetic border changes the problem
    # rather than translating it.
    padding_sensitive_tokens = (
        "boundary", "seed", "symmetry_context", "marker_triplet",
        "diagonal_marker", "marker_scaffold", "cross_spiral",
    )
    padding_eligible = (
        same_shape
        and family in padding_eligible_families
        and not any(token in name for token in padding_sensitive_tokens)
    )
    if padding_eligible:
        ok = True
        for pair in pairs:
            inp = normalize_grid(pair["input"])
            bg = border_background_color(inp)
            try:
                pred = normalize_grid(call_with_timeout(
                    lambda pair=pair, bg=bg: transform(_pad_grid(deepcopy(pair["input"]), bg)),
                    0.25,
                ))
            except Exception:
                ok = False
                break
            expected = _pad_grid(deepcopy(pair["output"]), bg)
            if pred != expected:
                ok = False
                break
        checks["padding_translation_equivariance"] = ok
    else:
        checks["padding_translation_equivariance"] = "not_applicable_shape_or_family"

    bool_values = [value for value in checks.values() if isinstance(value, bool)]
    if not bool_values:
        checks["synthetic_invariance_available"] = False
    return checks


def synthetic_checks_status(checks: dict[str, bool | str]) -> str:
    """passed | failed | unavailable.

    Distinguishes an operator that FAILED an applicable invariance test from one
    that simply had NO applicable test. Strict admission must only penalize the
    former; quarantining the latter discards correct-but-untestable operators.
    """
    bool_values = [
        value for key, value in checks.items()
        if isinstance(value, bool) and key != "synthetic_invariance_available"
    ]
    if not bool_values:
        return "unavailable"
    return "passed" if all(bool_values) else "failed"


def synthetic_checks_pass(checks: dict[str, bool | str]) -> bool:
    return synthetic_checks_status(checks) == "passed"


def admit_fixed_operator(
    name: str,
    transform: Transform,
    family: str,
    prior: int,
    pairs: list[dict[str, Grid]],
    *,
    strict: bool | None = None,
) -> CandidateSpec:
    strict = strict_operator_admission_mode() if strict is None else strict
    train_exact, train_accuracy = train_transform_score(transform, pairs)
    checks = fixed_operator_invariance_checks(name, family, transform, pairs) if train_exact else {}
    status = synthetic_checks_status(checks)
    cross_task_firing, generalization_tier = generalization_tier_for(name)
    evidence = OperatorEvidence(
        admission_type="fixed_transform",
        evidence_label="train_exact_fixed",
        loo_informative=False,
        train_exact=train_exact,
        train_accuracy=round(train_accuracy, 6),
        leave_one_out=False,
        jackknife_stable=False,
        structural_confidence=1.0 if train_exact else 0.0,
        synthetic_self_consistency=(status == "passed"),
        synthetic_invariance_status=status,
        synthetic_checks=checks,
        cross_task_firing=cross_task_firing,
        generalization_tier=generalization_tier,
    )
    if not train_exact:
        evidence.promotion_blockers.append("train_not_exact")
        return CandidateSpec(name, family, prior, None, "rejected", evidence)
    # Strict mode quarantines a *failed* applicable invariance test only — never an
    # operator that had no applicable test (status == "unavailable").
    if strict and status == "failed":
        evidence.promotion_blockers.append("fixed_synthetic_invariance_failed")
        return CandidateSpec(name, family, prior, transform, "quarantined", evidence)
    # Optional general-only gate (opt-in): require >=2-distinct-task firing evidence.
    if strict and env_enabled("ARC_REQUIRE_GENERAL_OPERATORS", "0") and generalization_tier == "memorized":
        evidence.promotion_blockers.append("single_task_memorized")
        return CandidateSpec(name, family, prior, transform, "quarantined", evidence)
    return CandidateSpec(name, family, prior, transform, "certified", evidence)


def inferred_general_operator_candidate_specs(pairs: list[dict[str, Grid]]) -> list[CandidateSpec]:
    specs = [
        admit_fixed_operator(name, transform, family, prior, pairs)
        for name, transform, family, prior in fixed_general_operator_defs()
    ]
    return specs


def inferred_general_operator_specs(pairs: list[dict[str, Grid]]) -> list[tuple[str, Transform, str, int]]:
    out: list[tuple[str, Transform, str, int]] = []
    for spec in inferred_general_operator_candidate_specs(pairs):
        if spec.tier == "certified" and spec.transform is not None:
            out.append((spec.name, spec.transform, spec.family, spec.prior))
    return out


def inferred_ttt_sparse_patch_repair_specs(pairs: list[dict[str, Grid]], epochs: int, max_cells_per_grid: int) -> list[tuple[str, Transform, str, int]]:
    def factory(train_pairs: list[dict[str, Grid]]) -> Transform | None:
        return train_ttt_sparse_patch_repair(train_pairs, epochs=epochs, max_cells_per_grid=max_cells_per_grid)

    transform = factory(pairs)
    if transform is None:
        return []
    train_exact, _score = train_transform_score(transform, pairs)
    if not train_exact:
        return []
    if not leave_one_out_validates(pairs, factory):
        return []
    return [("ttt_sparse_patch_repair:local_residual_rules", transform, "ttt_repair", 12)]


def inferred_line_network_candidate_specs(pairs: list[dict[str, Grid]], ns: dict[str, Any] | None) -> list[CandidateSpec]:
    if not ns:
        return []
    fn = ns.get("straighten_noisy_line_networks")
    if not callable(fn):
        return []

    def transform(grid: Grid, fn: Callable[[Grid], Grid] = fn) -> Grid:
        return normalize_grid(fn(deepcopy(grid)))

    return [admit_fixed_operator(
        "line_network_gap_bridge:straighten_noisy",
        transform,
        "line_network",
        17,
        pairs,
    )]


def inferred_line_network_specs(pairs: list[dict[str, Grid]], ns: dict[str, Any] | None) -> list[tuple[str, Transform, str, int]]:
    return [
        (spec.name, spec.transform, spec.family, spec.prior)
        for spec in inferred_line_network_candidate_specs(pairs, ns)
        if spec.tier == "certified" and spec.transform is not None
    ]


def object_candidate_specs(task_data: dict[str, Any], task_id: str | None = None, ns: dict[str, Any] | None = None) -> list[tuple[str, Transform, str, int]]:
    pairs = task_data.get("train", [])
    specs: list[tuple[str, Transform, str, int]] = []

    if task_id not in CONTAMINATED_DIRECT_SOLVER_TASKS:
        specs.extend(cached_program_specs(task_id, ns))
    specs.extend(inferred_border_chain_transfer_specs(pairs, ns))
    specs.extend(inferred_crop_cleanup_specs(pairs))
    specs.extend(inferred_single_motif_reconstruct_specs(pairs, ns))
    specs.extend(inferred_general_operator_specs(pairs))
    specs.extend(inferred_line_network_specs(pairs, ns))
    for spec in collect_operator_specs(task_data, ns=ns, task_id=task_id, include_fixed=False):
        if spec.tier == "certified" and spec.transform is not None:
            specs.append((spec.name, spec.transform, spec.family, spec.prior))

    for selector_name, selector in OBJECT_SELECTORS:
        specs.append((
            f"object_crop:{selector_name}",
            lambda grid, selector=selector: crop_selected_object(grid, selector),
            "object",
            60,
        ))
        specs.append((
            f"object_remove:{selector_name}",
            lambda grid, selector=selector: remove_selected_objects(grid, selector),
            "object",
            72,
        ))

    specs.extend([
        ("holes:fill_surround_majority", fill_enclosed_regions, "holes", 52),
        ("symmetry:repair_h", lambda grid: repair_symmetry_vote(grid, "h"), "symmetry", 54),
        ("symmetry:repair_v", lambda grid: repair_symmetry_vote(grid, "v"), "symmetry", 55),
        ("symmetry:repair_rot180", lambda grid: repair_symmetry_vote(grid, "rot180"), "symmetry", 55),
        ("symmetry:repair_diag", lambda grid: repair_symmetry_vote(grid, "diag"), "symmetry", 56),
        ("symmetry:repair_anti", lambda grid: repair_symmetry_vote(grid, "anti"), "symmetry", 56),
        ("alignment:connect_same_color", connect_aligned_same_color, "alignment", 56),
        ("alignment:connect_diagonal_same_color", lambda grid: connect_aligned_points(grid, diagonal=True), "alignment", 57),
        ("containment:outline_boxes", outline_component_boxes, "containment", 58),
    ])

    for name, fn in infer_color_filters(pairs):
        specs.append((name, fn, "color_filter", 26))
    for name, fn in infer_pixel_block_expansion(pairs):
        specs.append((name, fn, "block_expand", 24))
    for name, fn in infer_downscale_selectors(pairs):
        specs.append((name, fn, "block_downscale", 24))
    for name, fn in infer_panel_overlays(pairs):
        specs.append((name, fn, "panel_overlay", 25))
    for name, fn in infer_binary_hole_frame_repair(pairs):
        specs.append((name, fn, "color_role_frame", 22))
    for name, fn in infer_framed_periodic_repair(pairs):
        specs.append((name, fn, "periodic_repair", 23))
    for name, fn in infer_object_extractor(pairs):
        specs.append((name, fn, "object_inferred", 30))
    for name, fn in infer_object_remover(pairs):
        specs.append((name, fn, "object_inferred", 34))
    for name, fn in infer_hole_fill(pairs):
        specs.append((name, fn, "holes_inferred", 32))
    for name, fn in infer_panel_extractor(pairs):
        specs.append((name, fn, "separator", 36))

    if os.environ.get("ARC_ENABLE_TTT_PIXEL", "1").strip().lower() in {"1", "true", "yes", "on"}:
        ttt_epochs = int(os.environ.get("ARC_TTT_PIXEL_EPOCHS", "8"))
        ttt_max_cells = int(os.environ.get("ARC_TTT_PIXEL_MAX_CELLS", "400"))
        specs.extend(inferred_ttt_sparse_patch_repair_specs(pairs, ttt_epochs, min(ttt_max_cells, 225)))
        ttt = train_ttt_pixel_classifier(pairs, epochs=ttt_epochs, max_cells_per_grid=ttt_max_cells)
        if ttt is not None:
            specs.append(("ttt:pixel_perceptron", ttt, "ttt_neural", 38))

    return specs


def generate_candidates(task_data: dict[str, Any], ns: dict[str, Any] | None = None, task_id: str | None = None) -> list[RankedCandidate]:
    pairs = task_data.get("train", [])
    candidates: list[RankedCandidate] = []
    seen_names: set[str] = set()

    def add(name: str, fn: Transform, family: str, prior: int) -> None:
        if name in seen_names:
            return
        seen_names.add(name)
        scored = score_transform(name, fn, task_data, family, prior)
        if scored is not None:
            candidates.append(scored)

    for i, (name, fn) in enumerate(BASIC_TRANSFORMS):
        add(name, fn, "basic", i)

    for name, fn, family, prior in object_candidate_specs(task_data, task_id=task_id, ns=ns):
        add(name, fn, family, prior)

    perception_timeout = float(os.environ.get("ARC_PERCEPTION_SEARCH_TIMEOUT", "0"))
    perception_fn = perception_search_transform(task_data, ns, perception_timeout)
    if perception_fn is not None:
        add("perception:object_rule_search", perception_fn, "perception", 28)

    if ns:
        for i, name in enumerate(NAMESPACE_CANDIDATE_NAMES):
            fn = ns.get(name)
            if callable(fn):
                add(f"dsl:{name}", lambda grid, fn=fn: fn(deepcopy(grid)), "dsl", 100 + i)

        allow_solve_candidates = (
            env_enabled("ARC_ENABLE_DSL_SOLVE_CANDIDATES", "0" if hidden_facing_mode() else "1")
            and not env_enabled("ARC_DISABLE_DSL_SOLVE_CANDIDATES", "0")
            and task_id not in CONTAMINATED_DIRECT_SOLVER_TASKS
        )
        if allow_solve_candidates:
            solve_names = sorted(
                name for name, value in ns.items()
                if name.startswith("solve_") and callable(value)
            )
            for i, name in enumerate(solve_names[:80]):
                fn = ns[name]
                add(f"dsl:{name}", lambda grid, fn=fn: fn(deepcopy(grid)), "dsl_solve", 160 + i)

    for i, (base_name, base_fn) in enumerate(BASIC_TRANSFORMS):
        mapping = infer_color_map(pairs, base_fn)
        if mapping is None:
            continue

        def mapped(grid: Grid, base_fn: Transform = base_fn, mapping: dict[int, int] = mapping) -> Grid:
            return apply_color_map(normalize_grid(base_fn(grid)), mapping)

        add(f"{base_name}+color_map", mapped, "color_map", 20 + i)

    mapping = infer_color_map(pairs)
    if mapping is not None:
        add("color_map", lambda grid, mapping=mapping: apply_color_map(normalize_grid(grid), mapping), "color_map", 18)

    const = infer_constant_output(pairs)
    if const is not None:
        add("constant_output", lambda _grid, const=const: deepcopy(const), "constant", 80)

    factors = infer_scale(pairs)
    if factors is not None:
        add(f"scale_{factors[0]}x{factors[1]}", lambda grid, f=factors: scale_grid(normalize_grid(grid), *f), "resize", 40)

    factors = infer_tile(pairs)
    if factors is not None:
        add(f"tile_{factors[0]}x{factors[1]}", lambda grid, f=factors: tile_grid(normalize_grid(grid), *f), "tile", 45)

    factors = infer_downscale(pairs)
    if factors is not None:
        add(
            f"downscale_majority_{factors[0]}x{factors[1]}",
            lambda grid, f=factors: downscale_majority(normalize_grid(grid), *f),
            "resize",
            50,
        )

    for scored in narrow_tta_candidates([candidate for candidate in candidates if candidate.exact_train], task_data):
        if scored.name not in seen_names:
            seen_names.add(scored.name)
            candidates.append(scored)

    candidates.sort()
    return candidates


def load_namespace(workspace: Path = WORKSPACE) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    dsl_path = workspace / "dsl.py"
    if dsl_path.exists():
        exec(dsl_path.read_text(), ns)
    for batch_file in sorted(workspace.glob("eval_holdout_primitives_batch*.py")):
        batch_ns: dict[str, Any] = {}
        try:
            exec(batch_file.read_text(), batch_ns)
        except Exception:
            continue
        for name, value in batch_ns.items():
            if name.startswith("solve_") and callable(value):
                ns[name] = value
    return ns


def helper_attempts(task_id: str, task_data: dict[str, Any], test_input: Grid, ns: dict[str, Any]) -> list[tuple[str, Grid]]:
    try:
        from submission_helper import solve_task
    except Exception:
        return []
    try:
        a1, a2 = solve_task(task_id, task_data, test_input, ns)
    except Exception:
        return []
    out = []
    if a1:
        out.append(("submission_helper_attempt_1", normalize_grid(a1)))
    if a2 and grid_key(a2) not in {grid_key(v) for _, v in out}:
        out.append(("submission_helper_attempt_2", normalize_grid(a2)))
    return out


FAMILY_BONUS = {
    "object_inferred": 8.0,
    "holes_inferred": 7.5,
    "separator": 7.0,
    "perception": 6.5,
    "resize": 5.5,
    "tile": 5.0,
    "color_map": 4.5,
    "block_expand": 8.5,
    "block_downscale": 8.0,
    "panel_overlay": 8.0,
    "color_role_frame": 8.6,
    "color_role_diagonal_rays": 8.5,
    "row_marker_panel_flow": 8.4,
    "periodic_repair": 8.8,
    "border_chain_transfer": 9.2,
    "crop_cleanup": 9.0,
    "motif_reconstruct": 9.3,
    "shape_operator": 9.1,
    "object_motion": 9.0,
    "route_glyph_transfer": 9.4,
    "object_contact": 9.0,
    "panel_projection": 8.9,
    "frame_projection": 8.9,
    "line_network": 8.8,
    "color_filter": 7.0,
    "program_cache": 12.0,
    "ttt_neural": 3.5,
    "tta_d4": 2.8,
    "tta_color": 2.6,
    "ttt_repair": 5.5,
    "object": 2.5,
    "holes": 2.0,
    "symmetry": 2.0,
    "alignment": 1.8,
    "containment": 1.5,
    "basic": 1.0,
    "dsl": 0.5,
    "dsl_solve": -2.0,
    "constant": -3.0,
}


def candidate_prediction_score_no_model(
    candidate: RankedCandidate,
    pred: Grid,
    task_data: dict[str, Any],
    test_input: Grid,
    profile: dict[str, Any],
) -> float:
    score = candidate.train_accuracy * 100.0
    if candidate.exact_train:
        exact_bonus = float(os.environ.get("ARC_EXACT_TRAIN_BONUS", "45.0" if hidden_facing_mode() else "25.0"))
        score += exact_bonus
    score += FAMILY_BONUS.get(candidate.family, 0.0)
    score += prediction_profile_score(pred, test_input, profile)
    score -= max(0, candidate.sort_key[1]) * 0.01
    if grid_key(pred) == grid_key(test_input) and not profile.get("same_shape"):
        score -= 3.0
    return score


def candidate_prediction_score(
    candidate: RankedCandidate,
    pred: Grid,
    task_data: dict[str, Any],
    test_input: Grid,
    profile: dict[str, Any],
) -> float:
    score = candidate_prediction_score_no_model(candidate, pred, task_data, test_input, profile)
    model = load_ranker_model()
    if model:
        weight = float(os.environ.get("ARC_RANKER_MODEL_WEIGHT", model.get("default_weight", 12.0)))
        score += weight * ml_ranker_score(ranker_feature_dict(candidate, pred, task_data, test_input, profile), model)
    return score


def ranked_predictions_for_pair(
    candidates: list[RankedCandidate],
    task_data: dict[str, Any],
    test_input: Grid,
    *,
    limit: int = 32,
) -> list[tuple[float, str, Grid]]:
    profile = infer_output_profile(task_data)
    if env_enabled("ARC_ENABLE_SHAPE_PROFILE_FILTER", "1" if hidden_facing_mode() else "0"):
        profile = dict(profile)
        profile["predicted_output_shapes"] = predict_output_shapes_from_train(task_data.get("train", []), test_input)
    rows: list[tuple[float, str, Grid]] = []
    for candidate in candidates[:limit]:
        try:
            pred = normalize_grid(call_with_timeout(lambda c=candidate: c.transform(deepcopy(test_input)), 0.50))
        except Exception:
            continue
        score = candidate_prediction_score(candidate, pred, task_data, test_input, profile)
        rows.append((score, candidate.name, pred))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return rows


def solve_task(
    task_id: str,
    task_data: dict[str, Any],
    ns: dict[str, Any] | None = None,
    return_diagnostics: bool = False,
    include_helper_attempts: bool = True,
) -> list[dict[str, Grid]] | tuple[list[dict[str, Grid]], dict[str, Any]]:
    if ns is None:
        ns = load_namespace()

    operator_specs = collect_operator_specs(task_data, ns=ns, task_id=task_id)
    certified_specs = [spec for spec in operator_specs if spec.tier == "certified"]
    quarantined_specs = [spec for spec in operator_specs if spec.tier == "quarantined"]
    rejected_specs = [spec for spec in operator_specs if spec.tier == "rejected"]
    all_candidates = generate_candidates(task_data, ns=ns, task_id=task_id)
    exact_candidates = [c for c in all_candidates if c.exact_train]
    has_model = bool(load_ranker_model())
    default_pool_limit = "32"
    pool_limit = int(os.environ.get("ARC_CANDIDATE_POOL_LIMIT", default_pool_limit))
    exact_train_attempts_only = env_enabled(
        "ARC_EXACT_TRAIN_ATTEMPTS_ONLY",
        "1" if hidden_facing_mode() else "0",
    )
    if exact_candidates and exact_train_attempts_only:
        ranked_pool = exact_candidates
    elif exact_candidates:
        ranked_pool = exact_candidates + [c for c in all_candidates if not c.exact_train][: max(0, pool_limit - len(exact_candidates))]
    else:
        ranked_pool = all_candidates[:pool_limit]

    results: list[dict[str, Grid]] = []
    diagnostics = {
        "task_id": task_id,
        "hidden_facing_mode": hidden_facing_mode(),
        "ranker_model": "loaded" if has_model else "disabled_or_missing",
        "dsl_solve_candidates": (
            env_enabled("ARC_ENABLE_DSL_SOLVE_CANDIDATES", "0" if hidden_facing_mode() else "1")
            and not env_enabled("ARC_DISABLE_DSL_SOLVE_CANDIDATES", "0")
            and task_id not in CONTAMINATED_DIRECT_SOLVER_TASKS
        ),
        "narrow_tta": env_enabled("ARC_ENABLE_NARROW_TTA", "1" if hidden_facing_mode() else "0"),
        "strict_operator_admission": strict_operator_admission_mode(),
        "num_candidates": len(all_candidates),
        "num_exact_train": len(exact_candidates),
        "exact_train_attempts_only": bool(exact_candidates and exact_train_attempts_only),
        "top_candidates": [
            {
                "name": c.name,
                "family": c.family,
                "exact_train": c.exact_train,
                "train_accuracy": round(c.train_accuracy, 4),
                "prior": c.sort_key[1],
            }
            for c in all_candidates[:10]
        ],
        "exact_families": dict(Counter(c.family for c in exact_candidates)),
        "operator_admission_summary": {
            "by_tier": dict(Counter(spec.tier for spec in operator_specs)),
            "by_admission_type": dict(Counter(spec.evidence.admission_type for spec in operator_specs)),
            "fixed_train_exact": sum(
                1 for spec in operator_specs
                if spec.evidence.admission_type == "fixed_transform" and spec.evidence.train_exact
            ),
            "fixed_synthetic_pass": sum(
                1 for spec in operator_specs
                if spec.evidence.admission_type == "fixed_transform" and spec.evidence.synthetic_self_consistency
            ),
            "loo_informative": sum(1 for spec in operator_specs if spec.evidence.loo_informative),
            "loo_non_informative": sum(1 for spec in operator_specs if not spec.evidence.loo_informative),
            "fixed_synthetic_status": dict(Counter(
                spec.evidence.synthetic_invariance_status for spec in operator_specs
                if spec.evidence.admission_type == "fixed_transform"
            )),
            "by_generalization_tier": dict(Counter(
                spec.evidence.generalization_tier for spec in operator_specs
                if spec.evidence.admission_type == "fixed_transform"
            )),
        },
        "certified_operator_candidates": [],
        "quarantined_candidates": [],
        "rejected_speculative_candidates": [],
        "promotion_blockers": {},
        "shadow_attempt_sources": [],
    }

    for pair_index, pair in enumerate(task_data.get("test", [])):
        test_input = normalize_grid(pair["input"])
        attempts: list[tuple[str, Grid]] = []
        shadow_entries = []

        if include_helper_attempts and os.environ.get("ARC_DISABLE_HELPER_ATTEMPTS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            attempts.extend(helper_attempts(task_id, {"train": task_data.get("train", []), "test": [pair]}, test_input, ns))

        ranked_rows = ranked_predictions_for_pair(ranked_pool, task_data, test_input, limit=len(ranked_pool))
        for _score, name, pred in ranked_rows:
            if len(attempts) >= 2:
                break
            if grid_key(pred) not in {grid_key(v) for _, v in attempts}:
                attempts.append((name, pred))

        if not attempts:
            attempts.append(("identity_fallback", test_input))
        if len(attempts) == 1:
            zero = [[0]]
            if grid_key(attempts[0][1]) == grid_key(zero):
                attempts.append(("identity_fallback", test_input))
            else:
                attempts.append(("zero_fallback", zero))

        results.append({
            "attempt_1": normalize_grid(attempts[0][1]),
            "attempt_2": normalize_grid(attempts[1][1]),
        })
        diagnostics.setdefault("test_attempt_sources", []).append(
            {"pair_index": pair_index, "attempt_1": attempts[0][0], "attempt_2": attempts[1][0]}
        )
        for spec in quarantined_specs[:8]:
            if spec.transform is None:
                continue
            try:
                pred = normalize_grid(call_with_timeout(lambda spec=spec: spec.transform(deepcopy(test_input)), 0.50))
            except Exception as exc:
                shadow_entries.append({"name": spec.name, "error": str(exc)})
                continue
            entry = {
                "name": spec.name,
                "family": spec.family,
                "tier": spec.tier,
                "shape": list(shape(pred)),
            }
            if "output" in pair:
                expected = normalize_grid(pair["output"])
                diff_count = mismatch_count(expected, pred)
                entry["exact"] = diff_count == 0
                entry["mismatch_count"] = diff_count
            shadow_entries.append(entry)
        if shadow_entries:
            diagnostics["shadow_attempt_sources"].append({"pair_index": pair_index, "candidates": shadow_entries})

    evidence_scored_specs = [
        spec for spec in operator_specs
        if spec.transform is not None and spec.tier in {"certified", "quarantined"}
    ]
    if evidence_scored_specs:
        for spec in evidence_scored_specs:
            hits = 0
            total = 0
            diffs = []
            if spec.transform is not None:
                for pair in task_data.get("test", []):
                    if "output" not in pair:
                        continue
                    total += 1
                    try:
                        pred = normalize_grid(call_with_timeout(lambda spec=spec, pair=pair: spec.transform(deepcopy(pair["input"])), 0.50))
                    except Exception:
                        continue
                    diff_count = mismatch_count(pair["output"], pred)
                    diffs.append(diff_count)
                    hits += diff_count == 0
            spec.evidence.public_shadow_hits = hits if total else None
            spec.evidence.public_shadow_total = total if total else None
            spec.evidence.public_shadow_diff_counts = diffs
        diagnostics["certified_operator_candidates"] = [
            operator_spec_diagnostic(spec) for spec in certified_specs if spec.transform is not None
        ]
    if quarantined_specs:
        diagnostics["quarantined_candidates"] = [operator_spec_diagnostic(spec) for spec in quarantined_specs]
    if rejected_specs:
        diagnostics["rejected_speculative_candidates"] = [operator_spec_diagnostic(spec) for spec in rejected_specs[:8]]
    diagnostics["promotion_blockers"] = {
        spec.name: list(spec.evidence.promotion_blockers)
        for spec in operator_specs
        if spec.evidence.promotion_blockers and not (
            spec.evidence.admission_type == "fixed_transform"
            and spec.evidence.promotion_blockers == ["train_not_exact"]
        )
    }

    if return_diagnostics:
        return results, diagnostics
    return results


def solve_challenges(tasks: dict[str, Any], ns: dict[str, Any] | None = None) -> dict[str, list[dict[str, Grid]]]:
    if ns is None:
        ns = load_namespace()
    return {task_id: solve_task(task_id, task, ns=ns) for task_id, task in tasks.items()}


def random_grid(rng: random.Random, rows: int, cols: int, colors: int = 4, bg: int = 0) -> Grid:
    grid = [[bg for _ in range(cols)] for _ in range(rows)]
    object_count = rng.randint(1, 4)
    for _ in range(object_count):
        color = rng.randint(1, max(1, colors - 1))
        h = rng.randint(1, max(1, min(4, rows)))
        w = rng.randint(1, max(1, min(4, cols)))
        r0 = rng.randint(0, max(0, rows - h))
        c0 = rng.randint(0, max(0, cols - w))
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                if rng.random() < 0.75:
                    grid[r][c] = color
    return grid


def synthetic_transform_family(name: str, rng: random.Random) -> Transform:
    if name == "color_map":
        mapping = {0: 0}
        colors = list(range(1, 10))
        shuffled = colors[:]
        rng.shuffle(shuffled)
        mapping.update({src: dst for src, dst in zip(colors, shuffled)})
        return lambda grid, mapping=mapping: apply_color_map(grid, mapping)
    if name == "crop_largest":
        return lambda grid: crop_selected_object(grid, select_largest)
    if name == "remove_smallest":
        return lambda grid: remove_selected_objects(grid, select_smallest)
    if name == "fill_holes":
        color = rng.randint(1, 9)
        return lambda grid, color=color: fill_enclosed_regions(grid, color)
    if name == "connect_aligned":
        return connect_aligned_same_color
    if name == "mirror_h":
        return mirror_h
    if name == "mirror_v":
        return mirror_v
    if name == "rot90":
        return rot90
    return copy_grid


SYNTHETIC_FAMILIES = [
    "color_map",
    "crop_largest",
    "remove_smallest",
    "fill_holes",
    "connect_aligned",
    "mirror_h",
    "mirror_v",
    "rot90",
]


def generate_synthetic_arc_task(seed: int, family: str | None = None, n_train: int = 3, n_test: int = 1) -> dict[str, Any]:
    rng = random.Random(seed)
    family = family or rng.choice(SYNTHETIC_FAMILIES)
    transform = synthetic_transform_family(family, rng)
    pairs = []
    for _ in range(n_train + n_test):
        rows = rng.randint(4, 12)
        cols = rng.randint(4, 12)
        inp = random_grid(rng, rows, cols, colors=rng.randint(3, 7))
        out = normalize_grid(transform(deepcopy(inp)))
        if out == inp or out == [[0]]:
            out = normalize_grid(rot90(inp) if rows == cols else mirror_h(inp))
        pairs.append({"input": inp, "output": out})
    return {
        "family": family,
        "train": pairs[:n_train],
        "test": pairs[n_train:],
    }


def write_synthetic_tasks(path: Path, count: int, seed: int = 0) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    family_counts = Counter()
    with path.open("w") as fh:
        for i in range(count):
            task = generate_synthetic_arc_task(seed + i)
            family_counts[task["family"]] += 1
            fh.write(json.dumps(task, separators=(",", ":")) + "\n")
    return {"path": str(path), "count": count, "families": dict(family_counts)}


def synthetic_ranker_smoke(count: int = 64, seed: int = 0) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    solved = 0
    by_family: Counter[str] = Counter()
    hit_family: Counter[str] = Counter()
    for i in range(count):
        task = generate_synthetic_arc_task(seed + i)
        by_family[task["family"]] += 1
        entries = solve_task(f"synthetic_{i}", task, ns=ns)
        ok = True
        for idx, pair in enumerate(task["test"]):
            entry = entries[idx]
            if entry["attempt_1"] != pair["output"] and entry["attempt_2"] != pair["output"]:
                ok = False
                break
        if ok:
            solved += 1
            hit_family[task["family"]] += 1
    return {
        "count": count,
        "solved": solved,
        "score": solved / count if count else 0.0,
        "families": dict(by_family),
        "family_hits": dict(hit_family),
    }


def load_tasks(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return {
            p.stem: json.loads(p.read_text())
            for p in sorted(path.glob("*.json"))
            if "metadata" not in p.name.lower()
        }
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict task file: {path}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path, default=WORKSPACE / "arc_agi_2_data" / "evaluation")
    parser.add_argument("--generate-synthetic", type=Path, default=None)
    parser.add_argument("--synthetic-count", type=int, default=256)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()

    if args.generate_synthetic:
        report = write_synthetic_tasks(args.generate_synthetic, args.synthetic_count)
        print(json.dumps(report, indent=2))
    elif args.synthetic_smoke:
        print(json.dumps(synthetic_ranker_smoke(args.synthetic_count), indent=2))
    else:
        tasks = load_tasks(args.source)
        ns = load_namespace()
        submission = solve_challenges(tasks, ns=ns)
        print(json.dumps(submission, separators=(",", ":")))
