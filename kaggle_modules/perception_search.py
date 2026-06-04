"""Perception-first object-level search for ARC tasks.

Inspired by arXiv 2604.02434 (Compositional Neuro-Symbolic Reasoning).
Segments objects, establishes correspondence, enumerates per-object transforms,
verifies cross-example consistency, emits executable code.

No LLM calls — pure enumeration (~2-10s per task).
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectDescriptor:
    idx: int
    coords: list[tuple[int, int]]
    color: int                        # dominant non-bg color
    bbox: tuple[int, int, int, int]   # (min_r, min_c, max_r, max_c)
    size: int
    centroid: tuple[float, float]
    shape_hash: int
    is_rectangle: bool
    touches_border: bool
    cropped: list[list[int]]


@dataclass
class TransformSpec:
    kind: str       # 'identity','mirror_h','mirror_v','rotate_cw','rotate_ccw',
                    # 'rotate_180','move','recolor','delete','fill_bbox','outline'
    params: dict = field(default_factory=dict)

    def __eq__(self, other):
        return isinstance(other, TransformSpec) and self.kind == other.kind and self.params == other.params

    def __hash__(self):
        return hash((self.kind, tuple(sorted(self.params.items()))))


@dataclass
class ObjectRule:
    condition_key: str      # 'color','size_rank','shape_hash','quadrant','touches_border'
    condition_value: Any
    transform: TransformSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_bbox(coords):
    rs = [r for r, c in coords]
    cs = [c for r, c in coords]
    return (min(rs), min(cs), max(rs), max(cs))


def _crop(grid, coords, bg):
    min_r, min_c, max_r, max_c = _get_bbox(coords)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    out = [[bg] * w for _ in range(h)]
    for r, c in coords:
        out[r - min_r][c - min_c] = grid[r][c]
    return out


def _normalize(coords):
    if not coords:
        return []
    min_r = min(r for r, c in coords)
    min_c = min(c for r, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _mirror_h(grid):
    return [row[::-1] for row in grid]


def _mirror_v(grid):
    return grid[::-1]


def _rotate_cw(grid):
    if not grid or not grid[0]:
        return grid
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]


def _rotate_ccw(grid):
    if not grid or not grid[0]:
        return grid
    rows, cols = len(grid), len(grid[0])
    return [[grid[r][cols - 1 - c] for r in range(rows)] for c in range(cols - 1, -1, -1)]


def _rotate_180(grid):
    return [row[::-1] for row in grid[::-1]]


def _transpose(grid):
    if not grid or not grid[0]:
        return grid
    rows, cols = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(rows)] for c in range(cols)]


def _grids_equal(a, b):
    if len(a) != len(b):
        return False
    return all(list(ra) == list(rb) for ra, rb in zip(a, b))


def _detect_bg(grid):
    rows, cols = len(grid), len(grid[0])
    border = []
    for c in range(cols):
        border.append(grid[0][c])
        border.append(grid[rows - 1][c])
    for r in range(rows):
        border.append(grid[r][0])
        border.append(grid[r][cols - 1])
    if border:
        return Counter(border).most_common(1)[0][0]
    return 0


def _quadrant(centroid, rows, cols):
    mid_r, mid_c = rows / 2.0, cols / 2.0
    if centroid[0] < mid_r:
        return "top_left" if centroid[1] < mid_c else "top_right"
    return "bottom_left" if centroid[1] < mid_c else "bottom_right"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def extract_properties(grid, obj, bg, idx):
    """Extract object descriptor from coordinate list."""
    min_r, min_c, max_r, max_c = _get_bbox(obj)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    colors = [grid[r][c] for r, c in obj if grid[r][c] != bg]
    dominant = Counter(colors).most_common(1)[0][0] if colors else bg
    norm = _normalize(obj)
    centroid_r = sum(r for r, c in obj) / len(obj)
    centroid_c = sum(c for r, c in obj) / len(obj)
    # Rectangle check
    is_rect = len(obj) == h * w
    # Border check
    rows, cols = len(grid), len(grid[0])
    touches = any(r == 0 or r == rows - 1 or c == 0 or c == cols - 1 for r, c in obj)

    return ObjectDescriptor(
        idx=idx,
        coords=obj,
        color=dominant,
        bbox=(min_r, min_c, max_r, max_c),
        size=len(obj),
        centroid=(centroid_r, centroid_c),
        shape_hash=hash(tuple(norm)),
        is_rectangle=is_rect,
        touches_border=touches,
        cropped=_crop(grid, obj, bg),
    )


def segment_grid(grid, bg, mode="objects"):
    """Segment grid into ObjectDescriptors."""
    rows, cols = len(grid), len(grid[0])
    if rows == 0 or cols == 0:
        return []

    # Use simple BFS segmentation (reimplemented to avoid dsl.py import issues)
    diag = mode.endswith("_diag")
    multi_color = mode.startswith("shapes")

    visited = set()
    objects = []
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if diag:
        neighbors += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and (r, c) not in visited:
                # BFS
                q = [(r, c)]
                visited.add((r, c))
                component = [(r, c)]
                color = grid[r][c]
                head = 0
                while head < len(q):
                    cr, cc = q[head]
                    head += 1
                    for dr, dc in neighbors:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                            if multi_color:
                                if grid[nr][nc] != bg:
                                    visited.add((nr, nc))
                                    q.append((nr, nc))
                                    component.append((nr, nc))
                            else:
                                if grid[nr][nc] == color:
                                    visited.add((nr, nc))
                                    q.append((nr, nc))
                                    component.append((nr, nc))
                objects.append(component)

    return [extract_properties(grid, obj, bg, i) for i, obj in enumerate(objects)]


# ---------------------------------------------------------------------------
# Object correspondence
# ---------------------------------------------------------------------------

def match_objects(in_descs, out_descs, strategy="color"):
    """Match input objects to output objects. Returns list of (in_idx, out_idx) or None."""
    if not in_descs or not out_descs:
        return None

    if strategy == "color":
        return _match_by_color(in_descs, out_descs)
    elif strategy == "shape":
        return _match_by_shape(in_descs, out_descs)
    elif strategy == "proximity":
        return _match_by_proximity(in_descs, out_descs)
    elif strategy == "size_rank":
        return _match_by_size_rank(in_descs, out_descs)
    elif strategy == "scan_order":
        return _match_by_scan_order(in_descs, out_descs)
    return None


def _match_by_color(in_descs, out_descs):
    """Match by dominant color — works when each color appears in exactly one object."""
    in_by_color = {}
    for d in in_descs:
        if d.color in in_by_color:
            return None  # duplicate color
        in_by_color[d.color] = d.idx
    out_by_color = {}
    for d in out_descs:
        if d.color in out_by_color:
            return None
        out_by_color[d.color] = d.idx
    # Match colors present in both
    pairs = []
    for color in in_by_color:
        if color in out_by_color:
            pairs.append((in_by_color[color], out_by_color[color]))
    # Need at least 1 match
    return pairs if pairs else None


def _match_by_shape(in_descs, out_descs):
    """Match by shape hash (position-invariant)."""
    in_by_shape = {}
    for d in in_descs:
        in_by_shape.setdefault(d.shape_hash, []).append(d.idx)
    out_by_shape = {}
    for d in out_descs:
        out_by_shape.setdefault(d.shape_hash, []).append(d.idx)
    pairs = []
    for sh in in_by_shape:
        if sh in out_by_shape:
            ins = in_by_shape[sh]
            outs = out_by_shape[sh]
            if len(ins) == len(outs) == 1:
                pairs.append((ins[0], outs[0]))
            elif len(ins) == len(outs):
                # Multiple with same shape — match by proximity
                for i, ii in enumerate(ins):
                    if i < len(outs):
                        pairs.append((ii, outs[i]))
    return pairs if pairs else None


def _match_by_proximity(in_descs, out_descs):
    """Greedy nearest-centroid matching."""
    used_out = set()
    pairs = []
    for ind in in_descs:
        best_dist = float("inf")
        best_out = None
        for outd in out_descs:
            if outd.idx in used_out:
                continue
            dist = (ind.centroid[0] - outd.centroid[0]) ** 2 + (ind.centroid[1] - outd.centroid[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_out = outd.idx
        if best_out is not None:
            pairs.append((ind.idx, best_out))
            used_out.add(best_out)
    return pairs if pairs else None


def _match_by_size_rank(in_descs, out_descs):
    """Match by size rank (largest=0, etc.)."""
    in_sorted = sorted(in_descs, key=lambda d: -d.size)
    out_sorted = sorted(out_descs, key=lambda d: -d.size)
    n = min(len(in_sorted), len(out_sorted))
    return [(in_sorted[i].idx, out_sorted[i].idx) for i in range(n)] if n > 0 else None


def _match_by_scan_order(in_descs, out_descs):
    """Match by scan order (top-to-bottom, left-to-right)."""
    in_sorted = sorted(in_descs, key=lambda d: (d.centroid[0], d.centroid[1]))
    out_sorted = sorted(out_descs, key=lambda d: (d.centroid[0], d.centroid[1]))
    n = min(len(in_sorted), len(out_sorted))
    return [(in_sorted[i].idx, out_sorted[i].idx) for i in range(n)] if n > 0 else None


# ---------------------------------------------------------------------------
# Per-object transform detection
# ---------------------------------------------------------------------------

def find_object_transform(in_desc, out_desc, bg):
    """Determine what transform converts in_desc's crop to out_desc's crop."""
    in_crop = in_desc.cropped
    out_crop = out_desc.cropped

    in_bbox = in_desc.bbox
    out_bbox = out_desc.bbox
    dr = out_bbox[0] - in_bbox[0]
    dc = out_bbox[1] - in_bbox[1]

    # 1. Identity (same position)
    if _grids_equal(in_crop, out_crop) and dr == 0 and dc == 0:
        return TransformSpec("identity")

    # 2. Move only
    if _grids_equal(in_crop, out_crop):
        return TransformSpec("move", {"dr": dr, "dc": dc})

    # Geometric transforms — try each, with and without move
    transforms = [
        ("mirror_h", _mirror_h),
        ("mirror_v", _mirror_v),
        ("rotate_cw", _rotate_cw),
        ("rotate_ccw", _rotate_ccw),
        ("rotate_180", _rotate_180),
        ("transpose", _transpose),
    ]

    for name, fn in transforms:
        try:
            transformed = fn(in_crop)
            if _grids_equal(transformed, out_crop):
                if dr == 0 and dc == 0:
                    return TransformSpec(name)
                return TransformSpec(name, {"dr": dr, "dc": dc})
        except Exception:
            continue

    # Recolor: same shape, different color
    in_norm = _normalize(in_desc.coords)
    out_norm = _normalize(out_desc.coords)
    if in_norm == out_norm:
        in_colors = set(c for r, c_coord in in_desc.coords for c in [in_desc.color])
        out_colors = set(c for r, c_coord in out_desc.coords for c in [out_desc.color])
        if in_colors != out_colors:
            return TransformSpec("recolor", {
                "from_color": in_desc.color,
                "to_color": out_desc.color,
                "dr": dr, "dc": dc,
            })

    # Fill bbox: output is solid rectangle
    oh, ow = len(out_crop), len(out_crop[0]) if out_crop else 0
    if oh > 0 and ow > 0:
        dominant_out = Counter(v for row in out_crop for v in row if v != bg)
        if dominant_out:
            fill_color = dominant_out.most_common(1)[0][0]
            solid = [[fill_color] * ow for _ in range(oh)]
            if _grids_equal(out_crop, solid):
                return TransformSpec("fill_bbox", {"color": fill_color, "dr": dr, "dc": dc})

    # Outline: only border cells
    if out_desc.size < in_desc.size and out_desc.shape_hash != in_desc.shape_hash:
        out_cells = set(out_desc.coords)
        in_cells = set(in_desc.coords)
        border_cells = set()
        for r, c in in_cells:
            for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if (nr, nc) not in in_cells:
                    border_cells.add((r, c))
                    break
        if border_cells == out_cells:
            return TransformSpec("outline", {"dr": dr, "dc": dc})

    return None


# ---------------------------------------------------------------------------
# Cross-example consistency verification
# ---------------------------------------------------------------------------

def _get_condition_value(desc, key, all_descs, grid):
    """Extract condition value for an object under a given key."""
    if key == "color":
        return desc.color
    elif key == "size_rank":
        sorted_by_size = sorted(all_descs, key=lambda d: -d.size)
        for i, d in enumerate(sorted_by_size):
            if d.idx == desc.idx:
                return i
        return -1
    elif key == "quadrant":
        rows, cols = len(grid), len(grid[0])
        return _quadrant(desc.centroid, rows, cols)
    elif key == "touches_border":
        return desc.touches_border
    elif key == "shape_hash":
        return desc.shape_hash
    return None


def verify_consistency(task_data, bg, seg_mode, match_strategy):
    """Check if a consistent per-object rule set holds across all training pairs.

    Returns list of ObjectRules or None.
    """
    pairs = task_data.get("train", [])
    if len(pairs) < 2:
        return None  # need ≥2 examples for consistency check

    condition_keys = ["color", "size_rank", "quadrant", "touches_border", "shape_hash"]

    for cond_key in condition_keys:
        # Collect rules from each pair
        all_rules_per_pair = []
        valid = True

        for pair in pairs:
            in_grid = pair["input"]
            out_grid = pair["output"]
            pair_bg = _detect_bg(in_grid)

            in_descs = segment_grid(in_grid, pair_bg, seg_mode)
            out_descs = segment_grid(out_grid, pair_bg, seg_mode)

            if not in_descs:
                valid = False
                break

            matching = match_objects(in_descs, out_descs, match_strategy)
            if not matching:
                valid = False
                break

            # Find transform for each matched pair
            matched_in_ids = set()
            pair_rules = {}  # condition_value -> TransformSpec
            for in_idx, out_idx in matching:
                in_d = next((d for d in in_descs if d.idx == in_idx), None)
                out_d = next((d for d in out_descs if d.idx == out_idx), None)
                if not in_d or not out_d:
                    valid = False
                    break

                matched_in_ids.add(in_idx)
                transform = find_object_transform(in_d, out_d, pair_bg)
                if transform is None:
                    valid = False
                    break

                cond_val = _get_condition_value(in_d, cond_key, in_descs, in_grid)
                if cond_val is None:
                    valid = False
                    break

                # For move transforms, normalize dr/dc relative to object properties
                # so the rule generalizes (e.g., "move down by object_height")
                norm_transform = _normalize_transform(transform, in_d, out_d)
                if cond_val in pair_rules and pair_rules[cond_val] != norm_transform:
                    valid = False  # same condition maps to different transforms
                    break
                pair_rules[cond_val] = norm_transform

            # Include unmatched input objects as "identity" — ensures condition
            # key doesn't collide between transformed and non-transformed objects
            if valid:
                for in_d in in_descs:
                    if in_d.idx not in matched_in_ids:
                        cv = _get_condition_value(in_d, cond_key, in_descs, in_grid)
                        identity = TransformSpec("identity")
                        if cv in pair_rules and pair_rules[cv] != identity:
                            valid = False  # condition collides with a transform rule
                            break
                        pair_rules[cv] = identity

            if not valid:
                break
            all_rules_per_pair.append(pair_rules)

        if not valid or not all_rules_per_pair:
            continue

        # Check consistency: same condition_value -> same transform across all pairs
        ref_rules = all_rules_per_pair[0]
        consistent = True
        for other_rules in all_rules_per_pair[1:]:
            if set(ref_rules.keys()) != set(other_rules.keys()):
                consistent = False
                break
            for cv in ref_rules:
                if cv not in other_rules or ref_rules[cv] != other_rules[cv]:
                    consistent = False
                    break
            if not consistent:
                break

        if consistent and ref_rules:
            return [
                ObjectRule(condition_key=cond_key, condition_value=cv, transform=ts)
                for cv, ts in ref_rules.items()
            ]

    return None


def _normalize_transform(transform, in_desc, out_desc):
    """Normalize transforms for cross-example consistency checking.

    - Recolor: strip from_color (varies per example), keep to_color only.
    - Move: strip absolute dr/dc, mark as move_relative.
    - Geometric: if no position change, return as-is; with position change,
      keep geometric kind only (drop dr/dc).
    """
    params = dict(transform.params)
    dr = params.get("dr", 0)
    dc = params.get("dc", 0)

    # Recolor: normalize to just to_color (from_color varies per example)
    if transform.kind == "recolor":
        to_color = params.get("to_color", 0)
        return TransformSpec("recolor", {"to_color": to_color})

    # No position change — return stripped of dr/dc
    if dr == 0 and dc == 0:
        params.pop("dr", None)
        params.pop("dc", None)
        return TransformSpec(transform.kind, params)

    # Move only — mark as relative (pattern inferred later)
    if transform.kind == "move":
        return TransformSpec("move_relative")

    # Geometric + move — keep geometric kind, drop positional
    return TransformSpec(transform.kind)


# ---------------------------------------------------------------------------
# Relative-move pattern inference
# ---------------------------------------------------------------------------

# Each pattern: (name, predicted_dr_dc(min_r, min_c, max_r, max_c, rows, cols))
_MOVE_PATTERNS = [
    ("gravity_down",  lambda r0, c0, r1, c1, R, C: (R - 1 - r1, 0)),
    ("gravity_up",    lambda r0, c0, r1, c1, R, C: (-r0, 0)),
    ("gravity_left",  lambda r0, c0, r1, c1, R, C: (0, -c0)),
    ("gravity_right", lambda r0, c0, r1, c1, R, C: (0, C - 1 - c1)),
    ("to_center",     lambda r0, c0, r1, c1, R, C: (
        R // 2 - (r1 - r0 + 1) // 2 - r0,
        C // 2 - (r1 - r0 + 1) // 2 - c0  # intentionally approximate
    )),
    ("to_origin",     lambda r0, c0, r1, c1, R, C: (-r0, -c0)),
]


def _resolve_relative_moves(rules, task_data, bg, seg_mode, match_strategy):
    """For rules with move_relative, infer the actual move pattern.

    Re-runs segmentation/matching to recover per-example dr/dc, then checks
    which pattern template predicts all examples correctly.

    Returns updated rules (move_relative replaced with specific pattern),
    or None if no consistent pattern found for any move_relative rule.
    """
    # Check if any rule needs resolution
    move_rules = [(i, r) for i, r in enumerate(rules) if r.transform.kind == "move_relative"]
    if not move_rules:
        return rules  # nothing to resolve

    pairs = task_data.get("train", [])
    cond_key = rules[0].condition_key

    # Collect per-example dr/dc for each condition_value that has move_relative
    move_data = {}  # condition_value -> list of (dr, dc)
    for pair in pairs:
        in_grid = pair["input"]
        out_grid = pair["output"]
        pair_bg = _detect_bg(in_grid)
        rows, cols = len(in_grid), len(in_grid[0])

        in_descs = segment_grid(in_grid, pair_bg, seg_mode)
        out_descs = segment_grid(out_grid, pair_bg, seg_mode)
        if not in_descs:
            return None

        matching = match_objects(in_descs, out_descs, match_strategy)
        if not matching:
            return None

        for in_idx, out_idx in matching:
            in_d = next((d for d in in_descs if d.idx == in_idx), None)
            out_d = next((d for d in out_descs if d.idx == out_idx), None)
            if not in_d or not out_d:
                return None

            cond_val = _get_condition_value(in_d, cond_key, in_descs, in_grid)

            # Only care about condition values that map to move_relative
            if not any(r.condition_value == cond_val and r.transform.kind == "move_relative"
                       for _, r in move_rules):
                continue

            dr = out_d.bbox[0] - in_d.bbox[0]
            dc = out_d.bbox[1] - in_d.bbox[1]
            move_data.setdefault(cond_val, []).append(
                (dr, dc, in_d.bbox[0], in_d.bbox[1], in_d.bbox[2], in_d.bbox[3], rows, cols)
            )

    # For each move_relative rule, find a matching pattern
    updated = list(rules)
    for rule_idx, rule in move_rules:
        cv = rule.condition_value
        examples = move_data.get(cv, [])
        if not examples:
            return None

        found_pattern = None
        for pat_name, pat_fn in _MOVE_PATTERNS:
            matches_all = True
            for actual_dr, actual_dc, r0, c0, r1, c1, R, C in examples:
                pred_dr, pred_dc = pat_fn(r0, c0, r1, c1, R, C)
                if pred_dr != actual_dr or pred_dc != actual_dc:
                    matches_all = False
                    break
            if matches_all:
                found_pattern = pat_name
                break

        if found_pattern is None:
            return None  # can't resolve this move → reject whole ruleset

        updated[rule_idx] = ObjectRule(
            condition_key=rule.condition_key,
            condition_value=rule.condition_value,
            transform=TransformSpec(found_pattern),
        )

    return updated


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def generate_code(rules, seg_mode, default_bg=0):
    """Emit executable transform() code from rules."""
    cond_key = rules[0].condition_key

    # Determine segmentation call
    diag = "True" if seg_mode.endswith("_diag") else "False"
    if seg_mode.startswith("shapes"):
        seg_call = f"get_shapes(grid, background=bg, diag={diag})"
    else:
        seg_call = f"get_objects(grid, background=bg, diag={diag})"

    lines = [
        "def transform(grid):",
        "    bg = detect_background_color(grid)",
        f"    objs = {seg_call}",
        "    rows, cols = len(grid), len(grid[0])",
        "    out = [[bg] * cols for _ in range(rows)]",
        "    for obj in objs:",
    ]

    # Extract condition value
    if cond_key == "color":
        lines.append("        color = grid[obj[0][0]][obj[0][1]]")
        cond_expr = "color"
    elif cond_key == "size_rank":
        # Need to sort first — pre-compute ranks
        lines = [
            "def transform(grid):",
            "    bg = detect_background_color(grid)",
            f"    objs = {seg_call}",
            "    rows, cols = len(grid), len(grid[0])",
            "    out = [[bg] * cols for _ in range(rows)]",
            "    objs_sorted = sorted(range(len(objs)), key=lambda i: -len(objs[i]))",
            "    rank_map = {objs_sorted[i]: i for i in range(len(objs_sorted))}",
            "    for oi, obj in enumerate(objs):",
            "        rank = rank_map.get(oi, -1)",
        ]
        cond_expr = "rank"
    elif cond_key == "quadrant":
        lines += [
            "        cr = sum(r for r, c in obj) / len(obj)",
            "        cc = sum(c for r, c in obj) / len(obj)",
            "        quad = ('top_left' if cr < rows/2 else 'bottom_left') if cc < cols/2 else ('top_right' if cr < rows/2 else 'bottom_right')",
        ]
        cond_expr = "quad"
    elif cond_key == "touches_border":
        lines.append("        touches = any(r == 0 or r == rows-1 or c == 0 or c == cols-1 for r, c in obj)")
        cond_expr = "touches"
    else:  # shape_hash or fallback
        lines.append("        norm = tuple(sorted((r - min(rr for rr, _ in obj), c - min(cc for _, cc in obj)) for r, c in obj))")
        cond_expr = "hash(norm)"

    # Shared crop logic
    lines += [
        "        rs = [r for r, c in obj]",
        "        cs = [c for r, c in obj]",
        "        min_r, min_c = min(rs), min(cs)",
        "        max_r, max_c = max(rs), max(cs)",
        "        h, w = max_r - min_r + 1, max_c - min_c + 1",
        "        cropped = [[bg]*w for _ in range(h)]",
        "        for r, c in obj:",
        "            cropped[r - min_r][c - min_c] = grid[r][c]",
    ]

    # Generate conditional branches
    first = True
    for rule in rules:
        cv = rule.condition_value
        ts = rule.transform
        if first:
            lines.append(f"        if {cond_expr} == {cv!r}:")
            first = False
        else:
            lines.append(f"        elif {cond_expr} == {cv!r}:")

        lines += _gen_transform_block(ts, indent=12)

    # Default: identity
    lines.append("        else:")
    lines.append("            transformed = cropped")
    lines.append("            paste_r, paste_c = min_r, min_c")

    # Paste
    lines += [
        "        for r, row in enumerate(transformed):",
        "            for c, val in enumerate(row):",
        "                rr, cc = paste_r + r, paste_c + c",
        "                if 0 <= rr < rows and 0 <= cc < cols and val != bg:",
        "                    out[rr][cc] = val",
        "    return out",
    ]
    return "\n".join(lines)


def _gen_transform_block(ts, indent=12):
    """Generate the transform + paste_r/paste_c lines for one rule."""
    pad = " " * indent
    dr = ts.params.get("dr", 0)
    dc = ts.params.get("dc", 0)

    if ts.kind == "identity":
        return [f"{pad}transformed = cropped", f"{pad}paste_r, paste_c = min_r, min_c"]
    elif ts.kind == "move":
        return [f"{pad}transformed = cropped", f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "mirror_h":
        return [f"{pad}transformed = [row[::-1] for row in cropped]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "mirror_v":
        return [f"{pad}transformed = cropped[::-1]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "rotate_cw":
        return [f"{pad}transformed = [[cropped[len(cropped)-1-r][c] for r in range(len(cropped))] for c in range(len(cropped[0]))]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "rotate_ccw":
        return [f"{pad}transformed = [[cropped[r][len(cropped[0])-1-c] for r in range(len(cropped))] for c in range(len(cropped[0])-1, -1, -1)]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "rotate_180":
        return [f"{pad}transformed = [row[::-1] for row in cropped[::-1]]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "transpose":
        return [f"{pad}transformed = [[cropped[r][c] for r in range(len(cropped))] for c in range(len(cropped[0]))]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "recolor":
        to_c = ts.params.get("to_color", 0)
        if "from_color" in ts.params:
            from_c = ts.params["from_color"]
            return [f"{pad}transformed = [[{to_c} if v == {from_c} else v for v in row] for row in cropped]",
                    f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
        # Normalized: replace all non-bg pixels with to_color
        return [f"{pad}transformed = [[{to_c} if v != bg else bg for v in row] for row in cropped]",
                f"{pad}paste_r, paste_c = min_r, min_c"]
    elif ts.kind == "fill_bbox":
        fill_c = ts.params.get("color", 1)
        return [f"{pad}transformed = [[{fill_c}]*w for _ in range(h)]",
                f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}"]
    elif ts.kind == "outline":
        return [
            f"{pad}obj_set = set(obj)",
            f"{pad}border_cells = set()",
            f"{pad}for r, c in obj:",
            f"{pad}    for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:",
            f"{pad}        if (nr, nc) not in obj_set:",
            f"{pad}            border_cells.add((r, c))",
            f"{pad}            break",
            f"{pad}transformed = [[bg]*w for _ in range(h)]",
            f"{pad}for r, c in border_cells:",
            f"{pad}    transformed[r-min_r][c-min_c] = grid[r][c]",
            f"{pad}paste_r, paste_c = min_r + {dr}, min_c + {dc}",
        ]
    elif ts.kind == "move_relative":
        # Unresolved — shouldn't reach codegen (filtered by _resolve_relative_moves)
        return [f"{pad}transformed = cropped", f"{pad}paste_r, paste_c = min_r, min_c"]
    elif ts.kind == "gravity_down":
        return [f"{pad}transformed = cropped",
                f"{pad}paste_r, paste_c = rows - 1 - (max_r - min_r), min_c"]
    elif ts.kind == "gravity_up":
        return [f"{pad}transformed = cropped",
                f"{pad}paste_r, paste_c = 0, min_c"]
    elif ts.kind == "gravity_left":
        return [f"{pad}transformed = cropped",
                f"{pad}paste_r, paste_c = min_r, 0"]
    elif ts.kind == "gravity_right":
        return [f"{pad}transformed = cropped",
                f"{pad}paste_r, paste_c = min_r, cols - 1 - (max_c - min_c)"]
    elif ts.kind == "to_center":
        return [f"{pad}transformed = cropped",
                f"{pad}paste_r = rows // 2 - h // 2",
                f"{pad}paste_c = cols // 2 - w // 2"]
    elif ts.kind == "to_origin":
        return [f"{pad}transformed = cropped",
                f"{pad}paste_r, paste_c = 0, 0"]
    elif ts.kind == "delete":
        return [f"{pad}continue  # object deleted"]
    else:
        return [f"{pad}transformed = cropped", f"{pad}paste_r, paste_c = min_r, min_c"]


# ---------------------------------------------------------------------------
# End-to-end verification
# ---------------------------------------------------------------------------

def _verify_code(code, task_data, dsl_namespace):
    """Execute generated code and verify against all training pairs."""
    try:
        exec_ns = dict(dsl_namespace)
        exec(code, exec_ns)
        transform = exec_ns.get("transform")
        if not transform:
            return False
        for pair in task_data.get("train", []):
            result = transform(pair["input"])
            expected = pair["output"]
            if [list(r) for r in result] != [list(r) for r in expected]:
                return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def perception_search(task_data, dsl_namespace, timeout=15.0):
    """Perception-first object-level search.

    Segments objects, establishes correspondence, enumerates per-object
    transforms, verifies consistency, emits code.

    Returns "def transform(grid): ..." code string, or None.
    """
    start = time.monotonic()
    deadline = start + timeout
    pairs = task_data.get("train", [])
    if len(pairs) < 2:
        return None

    # Detect background from first pair
    bg = _detect_bg(pairs[0]["input"])

    seg_modes = ["objects", "shapes", "objects_diag", "shapes_diag"]
    match_strategies = ["color", "shape", "proximity", "size_rank", "scan_order"]

    for seg_mode in seg_modes:
        if time.monotonic() > deadline:
            break
        for match_strat in match_strategies:
            if time.monotonic() > deadline:
                break

            try:
                rules = verify_consistency(task_data, bg, seg_mode, match_strat)
                if rules:
                    # Resolve any move_relative rules to specific patterns
                    rules = _resolve_relative_moves(
                        rules, task_data, bg, seg_mode, match_strat
                    )
                    if not rules:
                        continue

                    code = generate_code(rules, seg_mode, bg)
                    if _verify_code(code, task_data, dsl_namespace):
                        elapsed = time.monotonic() - start
                        rule_summary = ", ".join(
                            f"{r.condition_value}->{r.transform.kind}"
                            for r in rules[:4]
                        )
                        print(f"  [perception] SOLVED: seg={seg_mode} match={match_strat} "
                              f"rules=[{rule_summary}] ({elapsed:.1f}s)")
                        return code
            except Exception:
                continue

    elapsed = time.monotonic() - start
    print(f"  [perception] no match ({elapsed:.1f}s, {len(seg_modes)}×{len(match_strategies)} combos)")
    return None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    dsl_path = WORKSPACE / "dsl.py"
    dsl_ns = {}
    exec(compile(dsl_path.read_text(), str(dsl_path), "exec"), dsl_ns)

    if len(sys.argv) > 1:
        task_path = Path(sys.argv[1])
        task_data = json.loads(task_path.read_text())
        print(f"[perception] Searching for {task_path.stem}...")
        result = perception_search(task_data, dsl_ns, timeout=15.0)
        if result:
            print(f"[perception] FOUND:\n{result}")
        else:
            print("[perception] No solution found")
    else:
        # Run on a few training tasks
        data_dir = WORKSPACE / "arc_agi_2_data" / "training"
        if data_dir.exists():
            tasks = sorted(data_dir.glob("*.json"))[:20]
            solved = 0
            for tp in tasks:
                td = json.loads(tp.read_text())
                code = perception_search(td, dsl_ns, timeout=10.0)
                if code:
                    solved += 1
            print(f"[perception] Solved {solved}/{len(tasks)} tasks")
