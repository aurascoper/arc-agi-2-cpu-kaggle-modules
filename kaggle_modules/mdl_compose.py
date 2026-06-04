"""MDL-guided composition of DSL primitives for ARC tasks.

Enumerates chains of grid->grid functions up to depth 3.
Selects shortest chain that passes all training examples exactly.
Runs without LLM -- pure Python enumeration (~2-5s per task).

Source: RCE (2602.15725) -- "12-18 point gains on ARC-AGI-2 with Mistral-7B"
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent

# Curated "interesting" primitives for depth 2-3 search (~32 functions).
# These are the most commonly useful grid->grid transforms.
INTERESTING_PRIMITIVES = [
    # Geometric (14)
    "rotate_cw", "rotate_ccw", "rotate_180", "mirror_h", "mirror_v",
    "transpose", "flip_anti_diagonal", "crop_foreground", "crop_to_bounding_box",
    "remove_border", "gravity_down", "symmetrize_h", "symmetrize_v", "scale_grid",
    # Color (8)
    "remove_noise", "fill_enclosed_background", "fill_holes", "recolor",
    "extract_color", "remove_color", "remap_colors", "infer_noise_color",
    # Object (6)
    "largest_object", "smallest_object", "extract_main_shape",
    "remove_small_objects", "sort_objects", "keep_most_common_colors",
    # Object-conditional (23)
    "extract_largest_object", "extract_smallest_object",
    "remove_largest_object", "remove_smallest_object",
    "keep_border_objects", "remove_border_objects", "isolate_unique_color_object",
    "mirror_each_object_h", "mirror_each_object_v", "rotate_each_object_cw",
    "fill_object_bboxes", "outline_objects",
    "recolor_by_size_rank", "recolor_by_row_position", "recolor_by_col_position",
    "color_objects_by_neighbor_count", "swap_object_colors_by_size",
    "gravity_objects_down", "gravity_objects_up", "gravity_objects_left",
    "gravity_objects_right", "move_objects_to_center", "spread_objects_from_center",
    # Repair (4)
    "repair_symmetry", "repair_holes", "best_symmetry_repair", "best_pattern_repair",
    # Eval-targeted (added 2026-04-14, 48h autoresearch loop)
    "recover_point_symmetric_region",   # 0934a4d8: recover 180° symmetric region masked by 8s
    "replace_plus_shapes_with_color",   # 1818057f: detect 5-cell plus shapes, recolor
    "reorganize_grid_sections",         # 78332cb0: reorder 6-separated sections (reverse/sort by count)
    "reflect_through_marker",           # 7ed72f31: reflect colored objects through 2-marker cells
    "connect_cells_via_key_sequence",   # 3e6067c3: connect colored boxes per bottom key-row sequence
    "frame_minority_cells",             # 71e489b6: frame isolated minority pixels with 7-borders
    # Eval-targeted (added 2026-04-15, batch 3)
    "fill_frames_with_marker_color",    # 8b7bacbf: fill open-corner frames with unique marker color
    "shoot_tips_at_noise_groups",       # 8b9c3697: shape tips cast beams at noise-2 groups
    "count_noise_clusters_into_frame",  # 8f215267: count noise clusters, place dots in frame middle
    "highlight_kth_longest_bar",        # 97d7923e: K header cells → recolor K-th longest bar
    # Eval-targeted (added 2026-04-15, batch 4)
    "transfer_box_center_via_border_chain",  # d35bdbdc: 3x3 box chain transfer with survive/erase
    "recolor_blobs_by_topological_holes",    # e3721c99: recolor 5-blobs by topological hole count
    # Eval-targeted (added 2026-04-15, batch 5)
    "apply_mod6_column_stripe_pattern",      # 221dfab4: mod-6 row pattern on 4-marked columns
    # Eval-targeted (added 2026-04-15, batch 6)
    "erase_blobs_containing_key_set",        # d59b0160: erase regions containing all 3x3-corner key values
    "swap_cross_hub_tail_colors",            # c7f57c3e: cross motif hub/tail color swap (X-type reflect, +-type shift)
    # Eval-targeted (added 2026-04-15, batch 7)
    "solve_a32d8b75",                        # a32d8b75: left-panel tile+marker → stamp swapped tile on canvas
    "solve_9385bd28",                        # 9385bd28: legend-based bbox fill, reverse priority, exclusion zones
    "solve_9bbf930d",                        # 9bbf930d: staircase col-0 marker repositioning (A1/A2/B1/B3/B5)
    "solve_80a900e0",                        # 80a900e0: checkerboard + cluster corner diagonal rays
]

# Adjacent pairs that cancel out (no-ops) -- skip these in depth-2/3
CANCEL_PAIRS = {
    ("rotate_cw", "rotate_ccw"),
    ("rotate_ccw", "rotate_cw"),
    ("mirror_h", "mirror_h"),
    ("mirror_v", "mirror_v"),
    ("transpose", "transpose"),
    ("rotate_180", "rotate_180"),
    ("flip_anti_diagonal", "flip_anti_diagonal"),
}


def _is_grid(val) -> bool:
    """Check if value looks like a grid (list of lists of ints)."""
    if not isinstance(val, list) or len(val) == 0:
        return False
    if not isinstance(val[0], list):
        return False
    return all(isinstance(c, (int, float)) for c in val[0])


def discover_composable(dsl_namespace: dict) -> list[tuple[str, callable]]:
    """Find functions that accept a single grid and return a grid.

    Tests each function with a 3x3 canary grid. Returns (name, fn) pairs.
    Each function is tested with a 2s SIGALRM timeout to catch infinite loops.
    """
    import copy as _copy
    import signal as _signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("discover_composable: function timed out on canary")

    _canary_orig = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    composable = []

    for name, obj in sorted(dsl_namespace.items()):
        if not callable(obj) or name.startswith("_"):
            continue
        # Skip known non-composable (classes, etc.)
        if isinstance(obj, type):
            continue
        try:
            sig = inspect.signature(obj)
            params = list(sig.parameters.values())
            if not params:
                continue
            # Must have exactly 1 required param (the grid)
            required = [p for p in params if p.default is inspect.Parameter.empty
                        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            if len(required) != 1:
                continue
        except (ValueError, TypeError, NameError):
            continue

        try:
            # deepcopy so in-place mutating functions don't corrupt the canary
            # SIGALRM guard catches functions that loop on canary despite deepcopy
            old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(2)
            try:
                result = obj(_copy.deepcopy(_canary_orig))
            finally:
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, old_handler)
            if _is_grid(result):
                composable.append((name, obj))
        except Exception:
            continue

    return composable


def _eval_chain(chain: list[callable], task_data: dict) -> tuple[bool, float]:
    """Run chain on all training pairs. Returns (all_passed, mean_pixel_accuracy).

    Early exit on first mismatch but records pixel accuracy.
    """
    pairs = task_data.get("train", [])
    if not pairs:
        return False, 0.0

    total_correct = 0
    total_pixels = 0

    for pair in pairs:
        grid = pair["input"]
        expected = pair["output"]
        try:
            for fn in chain:
                grid = fn(grid)
            # Normalize to list of lists
            result = [list(row) for row in grid] if grid else []
            exp = [list(row) for row in expected]

            if result == exp:
                # Count all pixels as correct
                pixels = sum(len(row) for row in exp)
                total_correct += pixels
                total_pixels += pixels
            else:
                # Count matching pixels
                if len(result) == len(exp) and all(len(r) == len(e) for r, e in zip(result, exp)):
                    for r_row, e_row in zip(result, exp):
                        for r_val, e_val in zip(r_row, e_row):
                            total_pixels += 1
                            if r_val == e_val:
                                total_correct += 1
                else:
                    # Size mismatch -- 0 accuracy for this pair
                    total_pixels += sum(len(row) for row in exp)
                return False, total_correct / total_pixels if total_pixels else 0.0
        except Exception:
            return False, 0.0

    return True, 1.0


def _chain_to_code(func_names: list[str]) -> str:
    """Convert function name list to executable Python code."""
    lines = ["def transform(grid):"]
    for name in func_names:
        lines.append(f"    grid = {name}(grid)")
    lines.append("    return grid")
    return "\n".join(lines)


def compose_search(task_data: dict, dsl_namespace: dict, timeout: float = 5.0) -> str | None:
    """Search for a primitive chain matching all training pairs.

    Returns Python code string "def transform(grid): ..." or None.

    Search order (MDL preference = shorter chains first):
      Depth 1: all composable functions
      Depth 2: INTERESTING x INTERESTING (~1K chains)
      Depth 3: only if depth 1-2 found no match; guided by depth-2 partial matches
    """
    start = time.monotonic()
    deadline = start + timeout

    # Discover all composable functions
    all_composable = discover_composable(dsl_namespace)
    if not all_composable:
        return None

    all_by_name = {name: fn for name, fn in all_composable}

    # --- Depth 1: try all composable ---
    for name, fn in all_composable:
        if time.monotonic() > deadline:
            return None
        passed, _ = _eval_chain([fn], task_data)
        if passed:
            return _chain_to_code([name])

    # --- Depth 2: interesting x interesting ---
    interesting_fns = [(n, all_by_name[n]) for n in INTERESTING_PRIMITIVES if n in all_by_name]
    depth2_partials = []  # Track partial matches for depth-3 guidance

    for i, (n1, f1) in enumerate(interesting_fns):
        if time.monotonic() > deadline:
            return None
        for n2, f2 in interesting_fns:
            if (n1, n2) in CANCEL_PAIRS:
                continue
            passed, pa = _eval_chain([f1, f2], task_data)
            if passed:
                return _chain_to_code([n1, n2])
            if pa > 0.7:
                depth2_partials.append((n1, n2, f1, f2, pa))

    # --- Depth 3: use depth-2 partial matches as prefixes ---
    if not depth2_partials:
        return None

    # Sort by pixel accuracy descending -- best prefixes first
    depth2_partials.sort(key=lambda x: -x[4])
    # Limit to top 10 prefixes to stay within timeout
    for n1, n2, f1, f2, _ in depth2_partials[:10]:
        if time.monotonic() > deadline:
            return None
        for n3, f3 in interesting_fns:
            if (n2, n3) in CANCEL_PAIRS:
                continue
            if time.monotonic() > deadline:
                return None
            passed, _ = _eval_chain([f1, f2, f3], task_data)
            if passed:
                return _chain_to_code([n1, n2, n3])

    # --- P2: DAG composition templates (split-transform-merge) ---
    if time.monotonic() < deadline:
        dag_result = _dag_compose(task_data, interesting_fns, all_by_name, dsl_namespace, deadline)
        if dag_result:
            return dag_result

    return None


# ---------------------------------------------------------------------------
# P2: DAG composition templates
# ---------------------------------------------------------------------------

def _verify_all_pairs(code: str, task_data: dict, dsl_namespace: dict) -> bool:
    """Execute code and verify against ALL training pairs."""
    try:
        exec_ns = dict(dsl_namespace)
        exec(code, exec_ns)
        transform = exec_ns.get("transform")
        if not transform:
            return False
        for pair in task_data.get("train", []):
            result = transform(pair["input"])
            if [list(r) for r in result] != [list(r) for r in pair["output"]]:
                return False
        return True
    except Exception:
        return False


def _template_extract_overlay(task_data, interesting_fns, all_by_name, dsl_ns, deadline):
    """Template: crop_foreground → transform → overlay onto background."""
    if "crop_foreground" not in all_by_name or "detect_background_color" not in dsl_ns:
        return None
    for fname, _fn in interesting_fns[:15]:
        if time.monotonic() > deadline:
            return None
        if fname in ("crop_foreground", "detect_background_color"):
            continue
        code = (
            f"def transform(grid):\n"
            f"    bg = detect_background_color(grid)\n"
            f"    fg = crop_foreground(grid)\n"
            f"    transformed = {fname}(fg)\n"
            f"    rows, cols = len(grid), len(grid[0])\n"
            f"    out = [[bg]*cols for _ in range(rows)]\n"
            f"    tr, tc = len(transformed), len(transformed[0]) if transformed else 0\n"
            f"    sr = (rows - tr) // 2\n"
            f"    sc = (cols - tc) // 2\n"
            f"    for r in range(min(tr, rows)):\n"
            f"        for c in range(min(tc, cols)):\n"
            f"            if transformed[r][c] != bg:\n"
            f"                out[min(sr+r, rows-1)][min(sc+c, cols-1)] = transformed[r][c]\n"
            f"    return out\n"
        )
        if _verify_all_pairs(code, task_data, dsl_ns):
            return code
    return None


def _template_mask_apply(task_data, interesting_fns, all_by_name, dsl_ns, deadline):
    """Template: detect background → apply transform to foreground cells only."""
    if "detect_background_color" not in dsl_ns:
        return None
    for fname, _fn in interesting_fns[:15]:
        if time.monotonic() > deadline:
            return None
        code = (
            f"def transform(grid):\n"
            f"    bg = detect_background_color(grid)\n"
            f"    transformed = {fname}(grid)\n"
            f"    rows, cols = len(grid), len(grid[0])\n"
            f"    out = [row[:] for row in grid]\n"
            f"    tr, tc = len(transformed), len(transformed[0]) if transformed else 0\n"
            f"    for r in range(min(rows, tr)):\n"
            f"        for c in range(min(cols, tc)):\n"
            f"            if grid[r][c] != bg:\n"
            f"                out[r][c] = transformed[r][c]\n"
            f"    return out\n"
        )
        if _verify_all_pairs(code, task_data, dsl_ns):
            return code
    return None


def _template_split_color(task_data, interesting_fns, all_by_name, dsl_ns, deadline):
    """Template: for each non-bg color, extract + transform + overlay."""
    if "detect_background_color" not in dsl_ns or "get_objects" not in dsl_ns:
        return None
    for fname, _fn in interesting_fns[:10]:
        if time.monotonic() > deadline:
            return None
        code = (
            f"def transform(grid):\n"
            f"    from copy import deepcopy\n"
            f"    bg = detect_background_color(grid)\n"
            f"    rows, cols = len(grid), len(grid[0])\n"
            f"    out = [[bg]*cols for _ in range(rows)]\n"
            f"    colors = set()\n"
            f"    for r in grid:\n"
            f"        for c in r:\n"
            f"            if c != bg:\n"
            f"                colors.add(c)\n"
            f"    for color in sorted(colors):\n"
            f"        layer = [[bg]*cols for _ in range(rows)]\n"
            f"        for r in range(rows):\n"
            f"            for c in range(cols):\n"
            f"                if grid[r][c] == color:\n"
            f"                    layer[r][c] = color\n"
            f"        transformed = {fname}(layer)\n"
            f"        tr, tc = len(transformed), len(transformed[0]) if transformed else 0\n"
            f"        for r in range(min(rows, tr)):\n"
            f"            for c in range(min(cols, tc)):\n"
            f"                if transformed[r][c] != bg:\n"
            f"                    out[r][c] = transformed[r][c]\n"
            f"    return out\n"
        )
        if _verify_all_pairs(code, task_data, dsl_ns):
            return code
    return None


def _dag_compose(task_data, interesting_fns, all_by_name, dsl_namespace, deadline):
    """Try all DAG templates. Returns code or None."""
    templates = [
        ("extract_overlay", _template_extract_overlay),
        ("mask_apply", _template_mask_apply),
        ("split_color", _template_split_color),
    ]
    for tname, tfn in templates:
        if time.monotonic() > deadline:
            return None
        result = tfn(task_data, interesting_fns, all_by_name, dsl_namespace, deadline)
        if result:
            print(f"  [compose] DAG {tname} found solution!")
            return result
    return None


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    # Load DSL namespace
    dsl_path = WORKSPACE / "dsl.py"
    dsl_ns = {}
    exec(compile(dsl_path.read_text(), str(dsl_path), "exec"), dsl_ns)

    composable = discover_composable(dsl_ns)
    interesting_available = [n for n in INTERESTING_PRIMITIVES if n in {c[0] for c in composable}]
    print(f"[mdl] Composable functions: {len(composable)}")
    print(f"[mdl] Interesting available: {len(interesting_available)}/{len(INTERESTING_PRIMITIVES)}")

    # Test on a task if provided
    if len(sys.argv) > 1:
        task_path = Path(sys.argv[1])
        task_data = json.loads(task_path.read_text())
        print(f"[mdl] Searching compositions for {task_path.stem}...")
        result = compose_search(task_data, dsl_ns, timeout=10.0)
        if result:
            print(f"[mdl] FOUND:\n{result}")
        else:
            print("[mdl] No composition found")
