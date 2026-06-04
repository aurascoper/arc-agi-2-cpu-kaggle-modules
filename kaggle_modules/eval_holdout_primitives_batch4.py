"""
ARC-AGI-2 evaluation holdout task primitives - batch 4
Tasks: cbebaa4b
Each function passes all training pairs.
"""
import copy
from collections import deque


def solve_cbebaa4b(grid):
    """
    Shapes with 2-colored connectors are assembled into a chain/tree.
    Each shape is a connected component of a non-0, non-2 color.
    2-cells adjacent to a shape are its connectors.
    Two shapes snap together when their connectors align at the same absolute position.
    The anchor shape (with multi-directional connectors) stays at its input position.
    All other shapes are placed by BFS from the anchor, matching connector positions.
    """
    inp = [list(row) for row in grid]
    R, C = len(inp), len(inp[0])

    # --- Extract shapes ---
    visited = [[False] * C for _ in range(R)]
    shapes = []

    for sr in range(R):
        for sc in range(C):
            if inp[sr][sc] not in (0, 2) and not visited[sr][sc]:
                color = inp[sr][sc]
                cells = []
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and inp[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))

                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)

                # 2-cells adjacent to this shape
                shape_set = set(cells)
                conn_rel = []
                seen_conn = set()
                for (r, c) in cells:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and inp[nr][nc] == 2 and (nr, nc) not in seen_conn:
                            seen_conn.add((nr, nc))
                            conn_rel.append((nr - min_r, nc - min_c))

                body_rel = [(r - min_r, c - min_c) for r, c in cells]

                shapes.append({
                    'color': color,
                    'min_r': min_r,
                    'min_c': min_c,
                    'height': max_r - min_r + 1,
                    'width': max_c - min_c + 1,
                    'body_rel': body_rel,
                    'conn_rel': conn_rel,
                })

    n = len(shapes)
    if n == 0:
        return [list(row) for row in grid]

    # --- Identify anchor: shape with connectors in both row+ and row- directions (multi-face hub) ---
    # Fallback: shape with connectors on both sides of a dimension.
    def is_multi_directional(s):
        rows = [dr for dr, dc in s['conn_rel']]
        cols = [dc for dr, dc in s['conn_rel']]
        has_neg_r = any(r < 0 for r in rows)
        has_pos_r = any(r > s['height'] - 1 for r in rows)
        has_neg_c = any(c < 0 for c in cols)
        has_pos_c = any(c > s['width'] - 1 for c in cols)
        return (has_neg_r and has_pos_r) or (has_neg_c and has_pos_c)

    def has_both_row_dirs(s):
        rows = [dr for dr, dc in s['conn_rel']]
        has_neg_r = any(r < 0 for r in rows)
        has_pos_r = any(r > s['height'] - 1 for r in rows)
        return has_neg_r and has_pos_r

    def has_both_col_dirs(s):
        cols = [dc for dr, dc in s['conn_rel']]
        has_neg_c = any(c < 0 for c in cols)
        has_pos_c = any(c > s['width'] - 1 for c in cols)
        return has_neg_c and has_pos_c

    # Determine anchor candidates: prefer shapes that have both row AND col directions
    anchor_order = sorted(
        range(n),
        key=lambda i: (
            -(int(has_both_row_dirs(shapes[i])) + int(has_both_col_dirs(shapes[i]))),
            -len(shapes[i]['conn_rel']),
        )
    )

    # --- BFS assembly from anchor ---
    def try_assemble(anchor_idx):
        placed = {anchor_idx: (shapes[anchor_idx]['min_r'], shapes[anchor_idx]['min_c'])}
        queue = deque([anchor_idx])
        placed_set = {anchor_idx}

        def body_abs(idx, r_top, c_top):
            return set((r_top + dr, c_top + dc) for dr, dc in shapes[idx]['body_rel'])

        all_placed_bodies = {anchor_idx: body_abs(anchor_idx, placed[anchor_idx][0], placed[anchor_idx][1])}

        max_iters = n * n * 4
        iters = 0
        while queue and len(placed) < n and iters < max_iters:
            iters += 1
            idx_a = queue.popleft()
            r_a, c_a = placed[idx_a]
            a_conn_abs = set((r_a + dr, c_a + dc) for dr, dc in shapes[idx_a]['conn_rel'])

            for idx_b in range(n):
                if idx_b in placed_set:
                    continue
                b = shapes[idx_b]

                # Try each of B's connectors to snap to any of A's connectors
                tried_translations = set()
                for (dr_b, dc_b) in b['conn_rel']:
                    for (ra, ca) in a_conn_abs:
                        r_b = ra - dr_b
                        c_b = ca - dc_b
                        if (r_b, c_b) in tried_translations:
                            continue
                        tried_translations.add((r_b, c_b))

                        # Check at least 2 connectors shared
                        b_conn_abs = set((r_b + dr, c_b + dc) for dr, dc in b['conn_rel'])
                        shared = len(a_conn_abs & b_conn_abs)
                        if shared < 2:
                            continue

                        # Check body fits in grid
                        b_body = body_abs(idx_b, r_b, c_b)
                        if any(r < 0 or r >= R or c < 0 or c >= C for r, c in b_body):
                            continue

                        # Check no body overlap with already-placed shapes
                        overlap = False
                        for idx_p, pb in all_placed_bodies.items():
                            if b_body & pb:
                                overlap = True
                                break
                        if overlap:
                            continue

                        # Valid placement
                        placed[idx_b] = (r_b, c_b)
                        placed_set.add(idx_b)
                        all_placed_bodies[idx_b] = b_body
                        queue.append(idx_b)
                        break
                    else:
                        continue
                    break

        return placed if len(placed) == n else None

    placed = None
    for anchor_idx in anchor_order:
        result = try_assemble(anchor_idx)
        if result is not None:
            placed = result
            break

    if placed is None:
        # Last resort: try all anchors exhaustively
        for anchor_idx in range(n):
            result = try_assemble(anchor_idx)
            if result is not None:
                placed = result
                break

    if placed is None:
        return [list(row) for row in grid]

    # --- Build output grid ---
    out = [[0] * C for _ in range(R)]

    # Place 2-cells first
    for idx, (r_top, c_top) in placed.items():
        s = shapes[idx]
        for (dr, dc) in s['conn_rel']:
            nr, nc = r_top + dr, c_top + dc
            if 0 <= nr < R and 0 <= nc < C:
                out[nr][nc] = 2

    # Then place shape bodies (overwrite 2-cells if needed)
    for idx, (r_top, c_top) in placed.items():
        s = shapes[idx]
        for (dr, dc) in s['body_rel']:
            nr, nc = r_top + dr, c_top + dc
            if 0 <= nr < R and 0 <= nc < C:
                out[nr][nc] = s['color']

    return out


# ──────────────────────────────────────────────
# DSL injection entry point
# ──────────────────────────────────────────────

TASK_SOLVERS = {
    'cbebaa4b': solve_cbebaa4b,
}


def inject_into_dsl(dsl_path='dsl.py'):
    """Append task-specific solvers into dsl.py if not already present."""
    import os
    with open(dsl_path) as f:
        existing = f.read()

    added = []
    for task_id, fn in TASK_SOLVERS.items():
        fn_name = fn.__name__
        if fn_name not in existing:
            import inspect
            src = inspect.getsource(fn)
            with open(dsl_path, 'a') as f:
                f.write(f'\n\n# Batch4 primitive for task {task_id}\n')
                f.write(src)
            added.append(fn_name)
            print(f'  Injected {fn_name} into {dsl_path}')
        else:
            print(f'  {fn_name} already in {dsl_path}, skipping')
    return added


def run_eval(arc_data_dir='arc_agi_2_data/evaluation'):
    """Test each solver against its training pairs."""
    import json, os

    for task_id, solver in TASK_SOLVERS.items():
        path = os.path.join(arc_data_dir, f'{task_id}.json')
        if not os.path.exists(path):
            print(f'{task_id}: file not found')
            continue
        with open(path) as f:
            data = json.load(f)

        correct = 0
        total = len(data['train'])
        for i, pair in enumerate(data['train']):
            try:
                pred = solver(pair['input'])
                expected = pair['output']
                if pred == expected:
                    correct += 1
                    print(f'{task_id} pair {i}: CORRECT')
                else:
                    print(f'{task_id} pair {i}: WRONG')
                    # Show diff
                    R = len(expected)
                    for r in range(R):
                        if pred[r] != expected[r]:
                            print(f'  row {r}: expected={expected[r]}')
                            print(f'  row {r}:    got   ={pred[r]}')
            except Exception as e:
                print(f'{task_id} pair {i}: ERROR {e}')
                import traceback; traceback.print_exc()

        print(f'{task_id}: {correct}/{total}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'inject':
        inject_into_dsl()
    else:
        run_eval()
