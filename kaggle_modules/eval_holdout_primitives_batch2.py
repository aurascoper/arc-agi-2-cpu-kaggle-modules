"""
ARC-AGI-2 evaluation holdout task primitives - batch 2
Tasks: 9385bd28, 97d7923e, 8b9c3697
Each function scores >= 3/4 training pairs.
"""
import copy
from collections import Counter, deque


def solve_9385bd28(grid):
    """
    Legend-based fill: column-pair legend maps shape_color -> fill_color.
    2-component shapes do ray-cast fill within their bounding box.
    1-component shapes convert their own cells to fill_color.
    Conflicts resolved by bounding box area (smaller bbox = higher priority).
    """
    g = copy.deepcopy(grid)
    rows = len(g); cols = len(g[0])
    flat = [v for row in g for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    legend = {}
    legend_col_start = None
    for start_c in range(min(cols - 1, 5)):
        cand = {}
        valid = True
        for r in range(rows):
            v0 = g[r][start_c]; v1 = g[r][start_c + 1]
            if v0 != bg and v1 != bg:
                if v0 in cand and cand[v0] != v1:
                    cand = {}; valid = False; break
                cand[v0] = v1
        if valid and cand and any(k != v for k, v in cand.items()):
            legend = cand; legend_col_start = start_c; break

    if legend_col_start is None:
        return g
    legend_cols = {legend_col_start, legend_col_start + 1}
    result = copy.deepcopy(g)

    def get_components(positions):
        pos_set = set(positions); visited = set(); components = []
        for start in positions:
            if start in visited:
                continue
            comp = []; q = deque([start]); visited.add(start)
            while q:
                r, c = q.popleft(); comp.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + dr, c + dc)
                    if nb in pos_set and nb not in visited:
                        visited.add(nb); q.append(nb)
            components.append(comp)
        return components

    def ray_cast_fills(positions, g, bg, min_r, max_r, min_c, max_c):
        to_fill = set()
        for r, c in positions:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                while min_r <= nr <= max_r and min_c <= nc <= max_c:
                    if g[nr][nc] != bg:
                        break
                    to_fill.add((nr, nc)); nr += dr; nc += dc
        return to_fill

    def find_interior_holes(positions, g, bg, min_r, max_r, min_c, max_c, ray_fills):
        shape_set = set(positions)
        exterior = set(); queue = deque()
        for r in range(min_r, max_r + 1):
            for c in [min_c, max_c]:
                cell = (r, c)
                if g[r][c] == bg and cell not in ray_fills and cell not in exterior:
                    exterior.add(cell); queue.append(cell)
        for c in range(min_c, max_c + 1):
            for r in [min_r, max_r]:
                cell = (r, c)
                if g[r][c] == bg and cell not in ray_fills and cell not in exterior:
                    exterior.add(cell); queue.append(cell)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc; cell = (nr, nc)
                if (min_r <= nr <= max_r and min_c <= nc <= max_c and cell not in exterior and
                        cell not in ray_fills and cell not in shape_set and g[nr][nc] == bg):
                    exterior.add(cell); queue.append(cell)
        return [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)
                if g[r][c] == bg and (r, c) not in ray_fills and (r, c) not in exterior
                and (r, c) not in shape_set]

    shape_fills = {}; shape_bbox_areas = {}
    for shape_color, fill_color in legend.items():
        positions = [(r, c) for r in range(rows) for c in range(cols)
                     if g[r][c] == shape_color and c not in legend_cols]
        if not positions:
            continue
        components = get_components(positions)
        if len(components) >= 2:
            min_r = min(r for r, c in positions); max_r = max(r for r, c in positions)
            min_c = min(c for r, c in positions); max_c = max(c for r, c in positions)
            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            to_fill = ray_cast_fills(positions, g, bg, min_r, max_r, min_c, max_c)
            has_blockers = any(g[r][c] != bg and g[r][c] != shape_color
                               for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1))
            if not has_blockers:
                holes = find_interior_holes(positions, g, bg, min_r, max_r, min_c, max_c, to_fill)
                to_fill.update(holes)
            shape_fills[shape_color] = (fill_color, to_fill, area)
            shape_bbox_areas[shape_color] = area
        else:
            shape_fills[shape_color] = (fill_color, set(positions), 0)
            shape_bbox_areas[shape_color] = 0

    # Resolve conflicts: for each cell, use fill from shape with SMALLEST bbox area
    cell_fills = {}
    for shape_color, (fill_color, to_fill, area) in shape_fills.items():
        for r, c in to_fill:
            cell = (r, c)
            if cell not in cell_fills or area < cell_fills[cell][1]:
                cell_fills[cell] = (fill_color, area)

    for (r, c), (fill_color, area) in cell_fills.items():
        if g[r][c] == bg or g[r][c] in legend.keys() or area == 0:
            result[r][c] = fill_color

    return result


def solve_97d7923e(grid):
    """
    Vertical bars with cap-body-cap structure. A reference column (or cells above a bar)
    gives a count N. The bar ranked N-th by body length (descending) for the matching cap
    color has its body cells changed to the cap color.
    """
    g = copy.deepcopy(grid)
    rows = len(g); cols = len(g[0])
    bg = 0

    bars = {}
    for c in range(cols):
        col_nonzero = [(r, g[r][c]) for r in range(rows) if g[r][c] != bg]
        if not col_nonzero:
            continue
        if col_nonzero[-1][0] != rows - 1:
            continue
        bottom_color = col_nonzero[-1][1]
        top_cap_idx = None
        for i, (r, v) in enumerate(col_nonzero):
            if v == bottom_color and r < rows - 1:
                top_cap_idx = i; break
        if top_cap_idx is None:
            continue
        body = [(r, v) for r, v in col_nonzero[top_cap_idx + 1:-1]]
        body_rows = [r for r, v in body]
        body_count = len(body)
        if bottom_color not in bars:
            bars[bottom_color] = []
        bars[bottom_color].append((body_count, c, col_nonzero[top_cap_idx][0], body_rows))

    references = {}
    for c in range(cols):
        col_nonzero = [(r, g[r][c]) for r in range(rows) if g[r][c] != bg]
        if not col_nonzero:
            continue
        has_bar = (col_nonzero[-1][0] == rows - 1)
        if has_bar:
            bottom_color = col_nonzero[-1][1]
            top_cap_idx = None
            for i, (r, v) in enumerate(col_nonzero):
                if v == bottom_color and r < rows - 1:
                    top_cap_idx = i; break
            if top_cap_idx is not None:
                for r, v in col_nonzero[:top_cap_idx]:
                    if v not in references:
                        references[v] = 0
                    references[v] += 1
        else:
            for r, v in col_nonzero:
                if v not in references:
                    references[v] = 0
                references[v] += 1

    result = copy.deepcopy(g)
    for cap_color, ref_count in references.items():
        if cap_color not in bars:
            continue
        sorted_bars = sorted(bars[cap_color], key=lambda x: x[0], reverse=True)
        if ref_count < 1 or ref_count > len(sorted_bars):
            continue
        body_count, c, top_cap_row, body_rows = sorted_bars[ref_count - 1]
        for r in body_rows:
            result[r][c] = cap_color

    return result


def _find_shape_interior_openings(shape_comp, all_shape_cells, rows, cols):
    """Returns dict mapping slide direction -> True if valid entry face for this shape."""
    min_r = min(r for r, c in shape_comp); max_r = max(r for r, c in shape_comp)
    min_c = min(c for r, c in shape_comp); max_c = max(c for r, c in shape_comp)
    openings = {}
    for face_dr, face_dc, boundary_cells in [
        (0, -1, [(r, max_c) for r in range(min_r, max_r + 1)]),
        (0, 1, [(r, min_c) for r in range(min_r, max_r + 1)]),
        (-1, 0, [(max_r, c) for c in range(min_c, max_c + 1)]),
        (1, 0, [(min_r, c) for c in range(min_c, max_c + 1)]),
    ]:
        for br, bc in boundary_cells:
            if (br, bc) in all_shape_cells:
                continue
            nr, nc = br + face_dr, bc + face_dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in all_shape_cells:
                if (face_dr, face_dc) not in openings:
                    openings[(face_dr, face_dc)] = []
                openings[(face_dr, face_dc)].append((br, bc))
                break
    return openings


def solve_8b9c3697(grid):
    """
    2-colored clusters slide into shape openings. Each cluster moves in one direction
    until blocked by a shape wall. Trail cells become 0. Clusters that don't fit any
    shape opening are removed. Valid entry: cluster must stop inside shape bounding box
    aligned with the shape's open face direction.
    """
    g = copy.deepcopy(grid)
    rows = len(g); cols = len(g[0])
    bg = g[0][0]

    flat = [v for row in g for v in row]
    color_counts = Counter(v for v in flat if v != bg and v != 2)
    if not color_counts:
        return g
    shape_color = color_counts.most_common(1)[0][0]
    shape_cells = set((r, c) for r in range(rows) for c in range(cols) if g[r][c] == shape_color)

    visited_s = set(); shape_comps = []
    for start in sorted(shape_cells):
        if start in visited_s:
            continue
        comp = set(); q = deque([start]); visited_s.add(start)
        while q:
            r, c = q.popleft(); comp.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in shape_cells and nb not in visited_s:
                    visited_s.add(nb); q.append(nb)
        shape_comps.append(comp)

    shape_info = []
    for comp in shape_comps:
        min_r = min(r for r, c in comp); max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp); max_c = max(c for r, c in comp)
        openings = _find_shape_interior_openings(comp, shape_cells, rows, cols)
        shape_info.append((min_r, max_r, min_c, max_c, openings))

    two_cells = set((r, c) for r in range(rows) for c in range(cols) if g[r][c] == 2)
    visited_2 = set(); clusters = []
    for start in sorted(two_cells):
        if start in visited_2:
            continue
        comp = []
        q = deque([start]); visited_2.add(start)
        while q:
            r, c = q.popleft(); comp.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb in two_cells and nb not in visited_2:
                    visited_2.add(nb); q.append(nb)
        clusters.append(comp)

    result = copy.deepcopy(g)
    for r, c in two_cells:
        result[r][c] = bg

    for cluster in clusters:
        min_r = min(r for r, c in cluster); min_c = min(c for r, c in cluster)
        max_r = max(r for r, c in cluster); max_c = max(c for r, c in cluster)
        rel = [(r - min_r, c - min_c) for r, c in cluster]
        h = max_r - min_r + 1; w = max_c - min_c + 1

        valid_directions = set()
        for (sb_min_r, sb_max_r, sb_min_c, sb_max_c, openings) in shape_info:
            valid_directions.update(openings.keys())

        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if direction not in valid_directions:
                continue
            dr, dc = direction
            final_pos = None
            for step in range(1, rows + cols):
                new_min_r = min_r + dr * step; new_min_c = min_c + dc * step
                new_cells = [(new_min_r + rr, new_min_c + rc) for rr, rc in rel]
                new_max_r = new_min_r + h - 1; new_max_c = new_min_c + w - 1
                if any(r < 0 or r >= rows or c < 0 or c >= cols for r, c in new_cells):
                    break
                if any((r, c) in shape_cells for r, c in new_cells):
                    break
                if any(g[r][c] != bg and g[r][c] != 2 for r, c in new_cells):
                    break
                next_cells = [(new_min_r + dr + rr, new_min_c + dc + rc) for rr, rc in rel]
                hits_shape_ahead = any((r, c) in shape_cells for r, c in next_cells
                                       if 0 <= r < rows and 0 <= c < cols)
                if hits_shape_ahead:
                    inside_valid_shape = False
                    for (sb_min_r, sb_max_r, sb_min_c, sb_max_c, openings) in shape_info:
                        if direction not in openings:
                            continue
                        if (sb_min_r <= new_min_r and new_max_r <= sb_max_r and
                                sb_min_c <= new_min_c and new_max_c <= sb_max_c):
                            inside_valid_shape = True; break
                    if inside_valid_shape and all(g[r][c] == bg for r, c in new_cells):
                        final_pos = (new_min_r, new_min_c, new_cells, step)
                    break

            if final_pos is None:
                continue

            new_min_r, new_min_c, final_cells, final_step = final_pos
            for s in range(0, final_step):
                trail_min_r = min_r + dr * s; trail_min_c = min_c + dc * s
                for tr, tc in [(trail_min_r + rr, trail_min_c + rc) for rr, rc in rel]:
                    if 0 <= tr < rows and 0 <= tc < cols:
                        result[tr][tc] = 0
            for r, c in final_cells:
                result[r][c] = 2
            break

    return result


if __name__ == '__main__':
    import json
    tasks = [
        ('9385bd28', solve_9385bd28),
        ('97d7923e', solve_97d7923e),
        ('8b9c3697', solve_8b9c3697),
    ]
    for task_id, solve_fn in tasks:
        with open(f'arc_agi_2_data/evaluation/{task_id}.json') as f:
            data = json.load(f)
        n = len(data['train'])
        passed = sum(1 for pair in data['train'] if solve_fn(pair['input']) == pair['output'])
        print(f'{task_id}: {passed}/{n}')
