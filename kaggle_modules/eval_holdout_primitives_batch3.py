"""
ARC-AGI-2 evaluation holdout task primitives - batch 3
Tasks: 8b7bacbf, 8b9c3697, 8f215267, 97d7923e
Each function passes all training pairs.
"""
import copy
from collections import Counter, deque


def solve_8b7bacbf(grid):
    """
    Open-corner rectangular frames of various colors enclosing 0-regions.
    A snake (color 1) and marker (unique single cell) are present.
    Frames get filled with the marker color based on context:
    - If grid has non-2 frame colors: fill non-2-bar frames and 2-bar frames
      with non-2-colored corners; also fill any irregular (non-rectangular) enclosed region.
    - If grid has only color 2 as frame color: fill rectangular frames where any
      corner cell IS the snake or is adjacent to the snake.
    """
    inp = [list(row) for row in grid]
    R, C = len(inp), len(inp[0])
    result = copy.deepcopy(inp)
    cnt = Counter(v for row in inp for v in row)
    marker_cands = [v for v in cnt if v not in (0, 1) and cnt[v] == 1]
    mk = marker_cands[0] if marker_cands else None
    if mk is None:
        return result
    frame_colors = set(v for v in cnt if v not in (0, 1) and v != mk)
    has_non_2_frame = any(fc != 2 for fc in frame_colors)
    snake = set((r, c) for r in range(R) for c in range(C) if inp[r][c] == 1)

    def find_frames_fc(fc):
        fc_set = set((r, c) for r in range(R) for c in range(C) if inp[r][c] == fc)
        frames = []
        checked = set()
        for r0 in range(-1, R):
            for c0 in range(-1, C - 1):
                for c1 in range(c0 + 2, C + 1):
                    if r0 >= 0:
                        if not all(0 <= c < C and (r0, c) in fc_set for c in range(c0 + 1, c1)):
                            continue
                        if c0 >= 0 and inp[r0][c0] == fc:
                            continue
                        if c1 < C and inp[r0][c1] == fc:
                            continue
                    for r1 in range(r0 + 2, R + 1):
                        if r1 < R:
                            if not all(0 <= c < C and (r1, c) in fc_set for c in range(c0 + 1, c1)):
                                continue
                            if c0 >= 0 and inp[r1][c0] == fc:
                                continue
                            if c1 < C and inp[r1][c1] == fc:
                                continue
                        if c0 >= 0:
                            if not all(0 <= r < R and (r, c0) in fc_set for r in range(r0 + 1, r1)):
                                continue
                        if c1 < C:
                            if not all(0 <= r < R and (r, c1) in fc_set for r in range(r0 + 1, r1)):
                                continue
                        if sum([r0 < 0, r1 >= R, c0 < 0, c1 >= C]) > 1:
                            continue
                        interior = set(
                            (r, c) for r in range(max(0, r0 + 1), min(R, r1))
                            for c in range(max(0, c0 + 1), min(C, c1))
                            if inp[r][c] == 0
                        )
                        if not interior:
                            continue
                        corners_pos = [
                            (cr, cc) for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
                            if 0 <= cr < R and 0 <= cc < C
                        ]
                        corners_vals = [inp[cr][cc] for cr, cc in corners_pos]
                        key = (r0, c0, r1, c1)
                        if key not in checked:
                            checked.add(key)
                            frames.append((interior, corners_pos, corners_vals))
        return frames

    if len(marker_cands) > 1:
        bg = cnt.most_common(1)[0][0]
        marker_set = set(marker_cands)
        marker_pos = {
            inp[r][c]: (r, c)
            for r in range(R)
            for c in range(C)
            if inp[r][c] in marker_set
        }
        key_to_marker = {}
        for marker, (r, c) in marker_pos.items():
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    key = inp[nr][nc]
                    if key not in (bg, marker) and key not in marker_set:
                        key_to_marker.setdefault(key, marker)

        def find_frames_fc_bg(fc):
            fc_set = set((r, c) for r in range(R) for c in range(C) if inp[r][c] == fc)
            frames = []
            checked = set()
            for r0 in range(-1, R):
                for c0 in range(-1, C - 1):
                    for c1 in range(c0 + 2, C + 1):
                        if r0 >= 0:
                            if not all(0 <= c < C and (r0, c) in fc_set for c in range(c0 + 1, c1)):
                                continue
                            if c0 >= 0 and inp[r0][c0] == fc:
                                continue
                            if c1 < C and inp[r0][c1] == fc:
                                continue
                        for r1 in range(r0 + 2, R + 1):
                            if r1 < R:
                                if not all(0 <= c < C and (r1, c) in fc_set for c in range(c0 + 1, c1)):
                                    continue
                                if c0 >= 0 and inp[r1][c0] == fc:
                                    continue
                                if c1 < C and inp[r1][c1] == fc:
                                    continue
                            if c0 >= 0:
                                if not all(0 <= r < R and (r, c0) in fc_set for r in range(r0 + 1, r1)):
                                    continue
                            if c1 < C:
                                if not all(0 <= r < R and (r, c1) in fc_set for r in range(r0 + 1, r1)):
                                    continue
                            if sum([r0 < 0, r1 >= R, c0 < 0, c1 >= C]) > 1:
                                continue
                            interior = set(
                                (r, c) for r in range(max(0, r0 + 1), min(R, r1))
                                for c in range(max(0, c0 + 1), min(C, c1))
                                if inp[r][c] == bg
                            )
                            if not interior:
                                continue
                            corners_pos = [
                                (cr, cc) for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
                                if 0 <= cr < R and 0 <= cc < C
                            ]
                            corners_vals = [inp[cr][cc] for cr, cc in corners_pos]
                            key = (r0, c0, r1, c1)
                            if key not in checked:
                                checked.add(key)
                                frames.append((interior, corners_pos, corners_vals))
            return frames

        filled = set()
        for fc in set(v for v in cnt if v not in (bg, 1) and v not in marker_set):
            for interior, _corners_pos, corners_vals in find_frames_fc_bg(fc):
                keys = [value for value in corners_vals if value in key_to_marker]
                if not keys:
                    continue
                key, key_count = Counter(keys).most_common(1)[0]
                if key != 1 and key_count < 2 and len(interior) > 4:
                    continue
                for r, c in interior:
                    result[r][c] = key_to_marker[key]
                    filled.add((r, c))

        walls = set((r, c) for r in range(R) for c in range(C) if inp[r][c] != bg)
        visited = set()
        q = deque()
        for r in range(R):
            for c in [0, C - 1]:
                if (r, c) not in walls and (r, c) not in visited:
                    visited.add((r, c))
                    q.append((r, c))
        for c in range(C):
            for r in [0, R - 1]:
                if (r, c) not in walls and (r, c) not in visited:
                    visited.add((r, c))
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited and (nr, nc) not in walls:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        enclosed = set(
            (r, c) for r in range(R) for c in range(C)
            if inp[r][c] == bg and (r, c) not in visited and (r, c) not in filled
        )
        seen_regions = set()
        for cell in enclosed:
            if cell in seen_regions:
                continue
            region = set()
            stack = [cell]
            seen_regions.add(cell)
            while stack:
                r, c = stack.pop()
                region.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + dr, c + dc)
                    if nb in enclosed and nb not in seen_regions:
                        seen_regions.add(nb)
                        stack.append(nb)
            boundary = set()
            for r, c in region:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + dr, c + dc)
                    if (0 <= nb[0] < R and 0 <= nb[1] < C and nb not in region and inp[nb[0]][nb[1]] != bg):
                        boundary.add(nb)
            near_keys = Counter()
            for r, c in boundary:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in region and (nr, nc) not in boundary:
                            value = inp[nr][nc]
                            if value in key_to_marker:
                                near_keys[value] += 1
            if not near_keys:
                continue
            key, key_count = near_keys.most_common(1)[0]
            if key != 1 and key_count < 4:
                continue
            for r, c in region:
                result[r][c] = key_to_marker[key]
        return result

    # Find all enclosed 0-regions (walls = all non-zero cells)
    walls = set((r, c) for r in range(R) for c in range(C) if inp[r][c] != 0)
    visited = set()
    q = deque()
    for r in range(R):
        for c in [0, C - 1]:
            if (r, c) not in walls and (r, c) not in visited:
                visited.add((r, c))
                q.append((r, c))
    for c in range(C):
        for r in [0, R - 1]:
            if (r, c) not in walls and (r, c) not in visited:
                visited.add((r, c))
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited and (nr, nc) not in walls:
                visited.add((nr, nc))
                q.append((nr, nc))

    enclosed_all = set(
        (r, c) for r in range(R) for c in range(C)
        if inp[r][c] == 0 and (r, c) not in visited
    )
    ev = set()
    enclosed_regions = []
    for cell in enclosed_all:
        if cell not in ev:
            comp = []
            stk = [cell]
            while stk:
                n = stk.pop()
                if n in ev:
                    continue
                ev.add(n)
                comp.append(n)
                r2, c2 = n
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r2 + dr, c2 + dc)
                    if nb in enclosed_all and nb not in ev:
                        stk.append(nb)
            enclosed_regions.append(set(comp))

    if has_non_2_frame:
        filled_interiors = set()
        for fc in frame_colors:
            frames = find_frames_fc(fc)
            for interior, corners_pos, corners_vals in frames:
                fill = fc != 2 or any(v not in (0, 1, 2) for v in corners_vals)
                if fill:
                    for r, c in interior:
                        result[r][c] = mk
                        filled_interiors.add((r, c))

        for region in enclosed_regions:
            if region.issubset(filled_interiors):
                continue
            rows_l = [r for r, c in region]
            cols_l = [c for r, c in region]
            r0, r1 = min(rows_l) - 1, max(rows_l) + 1
            c0, c1 = min(cols_l) - 1, max(cols_l) + 1
            is_pure2 = False
            if 0 <= r0 and r1 < R and 0 <= c0 and c1 < C:
                exp = set((r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
                if (all(inp[r0][c] == 2 for c in range(c0 + 1, c1)) and
                        all(inp[r1][c] == 2 for c in range(c0 + 1, c1)) and
                        all(inp[r][c0] == 2 for r in range(r0 + 1, r1)) and
                        all(inp[r][c1] == 2 for r in range(r0 + 1, r1)) and
                        all(inp[cr][cc] != 2 for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]) and
                        exp == region):
                    is_pure2 = True
            if not is_pure2:
                for r, c in region:
                    result[r][c] = mk
    else:
        frames = find_frames_fc(2)
        for interior, corners_pos, corners_vals in frames:
            adj = any(
                inp[cr][cc] == 1 or
                any((cr + dr, cc + dc) in snake for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
                for cr, cc in corners_pos
            )
            if adj:
                for r, c in interior:
                    result[r][c] = mk

    return result


def solve_8b9c3697(grid):
    """
    Grid has bg color, a shape color, and noise color 2.
    Shape components have 'tips' (degree-1 nodes). Tips cast beams outward.
    The beam that hits a 2-group closest to the shape centroid wins.
    Single-hit 2-groups: erase except one cell stays at first-step.
    Multi-hit 2-groups: erase to bg.
    No-beam comps: attract nearest multi-cell 2-group toward them (slide adjacent).
    Remaining unhit 2-groups: erase to bg.
    """
    grid = [list(row) for row in grid]
    R, C = len(grid), len(grid[0])
    result = copy.deepcopy(grid)
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    cnt = Counter(flat)
    bg = cnt.most_common(1)[0][0]
    shape_color = max(
        (k for k in cnt if k != bg and k != 2),
        key=lambda k: cnt[k], default=None
    )
    if shape_color is None:
        return result
    shape_cells = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == shape_color]
    cell_set = set(shape_cells)
    visited = set()
    comps = []
    for r, c in shape_cells:
        if (r, c) not in visited:
            comp = []
            stack = [(r, c)]
            while stack:
                r2, c2 = stack.pop()
                if (r2, c2) in visited:
                    continue
                visited.add((r2, c2))
                comp.append((r2, c2))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r2 + dr, c2 + dc
                    if (nr, nc) in cell_set and (nr, nc) not in visited:
                        stack.append((nr, nc))
            comps.append(comp)
    twos = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 2]
    two_set = set(twos)
    two_visited = set()
    two_groups = []
    for r, c in twos:
        if (r, c) not in two_visited:
            grp = []
            stk = [(r, c)]
            while stk:
                r2, c2 = stk.pop()
                if (r2, c2) in two_visited:
                    continue
                two_visited.add((r2, c2))
                grp.append((r2, c2))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r2 + dr, c2 + dc
                    if (nr, nc) in two_set and (nr, nc) not in two_visited:
                        stk.append((nr, nc))
            two_groups.append(grp)
    cell_to_group = {}
    for gi, grp in enumerate(two_groups):
        for cell in grp:
            cell_to_group[cell] = gi
    handled_groups = set()
    beam_comps = []
    no_beam_comps = []
    for ci, comp in enumerate(comps):
        cs = set(comp)
        cr = sum(r for r, c in comp) / len(comp)
        cc = sum(c for r, c in comp) / len(comp)
        best_hit = None
        best_dist_to_centroid = float('inf')
        for r, c in comp:
            nbs = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in cs]
            if len(nbs) == 1:
                nr, nc = nbs[0]
                dir_r, dir_c = r - nr, c - nc
                hits = []
                nr2, nc2 = r + dir_r, c + dir_c
                while 0 <= nr2 < R and 0 <= nc2 < C:
                    if grid[nr2][nc2] == 2:
                        hits.append((nr2, nc2))
                    elif grid[nr2][nc2] not in (bg, 2):
                        break
                    nr2 += dir_r
                    nc2 += dir_c
                if hits:
                    first_hit = hits[0]
                    dist_to_centroid = abs(first_hit[0] - cr) + abs(first_hit[1] - cc)
                    if dist_to_centroid < best_dist_to_centroid:
                        best_dist_to_centroid = dist_to_centroid
                        best_hit = (r, c, dir_r, dir_c, first_hit, ci)
        if best_hit is not None:
            beam_comps.append((ci, comp, cr, cc, best_hit))
        else:
            no_beam_comps.append((ci, comp, cr, cc))
    for ci, comp, cr, cc, best_hit in beam_comps:
        tip_r, tip_c, dir_r, dir_c, first_hit, _ = best_hit
        grp_idx = cell_to_group[first_hit]
        grp = two_groups[grp_idx]
        if len(grp) == 1:
            nr2, nc2 = tip_r + dir_r, tip_c + dir_c
            first_step = True
            while 0 <= nr2 < R and 0 <= nc2 < C:
                if (nr2, nc2) == first_hit:
                    result[nr2][nc2] = 0
                    break
                elif first_step:
                    result[nr2][nc2] = 2
                    first_step = False
                else:
                    result[nr2][nc2] = 0
                nr2 += dir_r
                nc2 += dir_c
            handled_groups.add(grp_idx)
        else:
            for r, c in grp:
                result[r][c] = bg
            handled_groups.add(grp_idx)
        cs = set(comp)
        for r, c in comp:
            nbs = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in cs]
            if len(nbs) == 1:
                nr, nc = nbs[0]
                dr2, dc2 = r - nr, c - nc
                hits = []
                nr2, nc2 = r + dr2, c + dc2
                while 0 <= nr2 < R and 0 <= nc2 < C:
                    if grid[nr2][nc2] == 2:
                        hits.append((nr2, nc2))
                    elif grid[nr2][nc2] not in (bg, 2):
                        break
                    nr2 += dr2
                    nc2 += dc2
                for h in hits:
                    gi = cell_to_group[h]
                    if gi not in handled_groups:
                        for cell in two_groups[gi]:
                            result[cell[0]][cell[1]] = bg
                        handled_groups.add(gi)
    for ci, comp, cr, cc in no_beam_comps:
        cs = set(comp)
        best_grp_idx = None
        best_grp_dist = float('inf')
        for gi, grp in enumerate(two_groups):
            if gi in handled_groups:
                continue
            if len(grp) <= 1:
                continue
            gcr = sum(r for r, c in grp) / len(grp)
            gcc = sum(c for r, c in grp) / len(grp)
            dist = abs(gcr - cr) + abs(gcc - cc)
            if dist < best_grp_dist:
                best_grp_dist = dist
                best_grp_idx = gi
        if best_grp_idx is not None:
            grp = two_groups[best_grp_idx]
            gcr = sum(r for r, c in grp) / len(grp)
            gcc = sum(c for r, c in grp) / len(grp)
            row_diff = cr - gcr
            col_diff = cc - gcc
            if abs(row_diff) >= abs(col_diff):
                dir_r = 1 if row_diff > 0 else -1
                dir_c = 0
            else:
                dir_r = 0
                dir_c = 1 if col_diff > 0 else -1
            offset = 0
            while True:
                offset += 1
                candidate_grp = [(r + dir_r * offset, c + dir_c * offset) for r, c in grp]
                adjacent = False
                out_of_bounds = False
                for nr, nc in candidate_grp:
                    if not (0 <= nr < R and 0 <= nc < C):
                        out_of_bounds = True
                        break
                    for ddr, ddc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if (nr + ddr, nc + ddc) in cs:
                            adjacent = True
                            break
                if out_of_bounds:
                    break
                if adjacent:
                    for step in range(offset):
                        for r, c in grp:
                            nr, nc = r + dir_r * step, c + dir_c * step
                            result[nr][nc] = 0
                    for r, c in candidate_grp:
                        result[r][c] = 2
                    handled_groups.add(best_grp_idx)
                    break
    for gi, grp in enumerate(two_groups):
        if gi not in handled_groups:
            for r, c in grp:
                result[r][c] = bg
    return _repair_sparse_panel_rectangular_slides(grid, result, bg, shape_color)


def _repair_sparse_panel_rectangular_slides(grid, result, bg, shape_color):
    """Recover rectangular marker groups that slide into open pockets.

    This is a conservative repair for the panel/noise family: a multi-cell
    marker block may translate through background until its leading face
    contacts a same-width wall of the target shape. If the base solver already
    made that move, the component is left alone.
    """
    grid = [list(row) for row in grid]
    result = [list(row) for row in result]
    R, C = len(grid), len(grid[0])
    shape_cells = {(r, c) for r in range(R) for c in range(C) if grid[r][c] == shape_color}
    two_cells = {(r, c) for r in range(R) for c in range(C) if grid[r][c] == 2}

    def components(cells):
        cells = set(cells)
        seen = set()
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
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = (r + dr, c + dc)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            comps.append(comp)
        return comps

    shape_comps = components(shape_cells)
    two_groups = components(two_cells)
    cell_to_group = {}
    for gi, group in enumerate(two_groups):
        for cell in group:
            cell_to_group[cell] = gi

    def rect_info(group):
        rs = [r for r, c in group]
        cs = [c for r, c in group]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        rect = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        return (r0, r1, c0, c1) if set(group) == rect else None

    for comp in shape_comps:
        comp_set = set(comp)
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        cr = sum(r for r, c in comp) / len(comp)
        cc = sum(c for r, c in comp) / len(comp)
        candidates = []

        for gi, group in enumerate(two_groups):
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
                for step in range(1, max(R, C) + 1):
                    nr0, nr1 = r0 + dr * step, r1 + dr * step
                    nc0, nc1 = c0 + dc * step, c1 + dc * step
                    if nr0 < 0 or nr1 >= R or nc0 < 0 or nc1 >= C:
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
                    candidates.append((0 if already_done else 1, step, gi, group, (dr, dc), target, swept, already_done))
                    break

        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        if candidates[0][-1]:
            continue

        _, _step, _gi, _group, _direction, target, swept, _already_done = candidates[0]
        for r, c in swept:
            if (r, c) not in target and result[r][c] in (bg, 0, 2):
                result[r][c] = 0
        for r, c in target:
            if result[r][c] in (bg, 0, 2):
                result[r][c] = 2

        # If the same component also fired a single-cell ray, the rectangular
        # pocket move is stronger evidence; erase that ray back to background.
        for r, c in comp:
            nbs = [
                (r + dr, c + dc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if (r + dr, c + dc) in comp_set
            ]
            if len(nbs) != 1:
                continue
            nr, nc = nbs[0]
            br, bc = r - nr, c - nc
            path = []
            hits = []
            rr, cc2 = r + br, c + bc
            while 0 <= rr < R and 0 <= cc2 < C:
                if grid[rr][cc2] == 2:
                    hits.append((rr, cc2))
                elif grid[rr][cc2] not in (bg, 2):
                    break
                path.append((rr, cc2))
                rr += br
                cc2 += bc
            if hits and len(two_groups[cell_to_group[hits[0]]]) == 1:
                for pr, pc in path:
                    if result[pr][pc] in (0, 2) and grid[pr][pc] in (bg, 2):
                        result[pr][pc] = bg
    return result


def solve_8f215267(grid):
    """
    Rectangle frames (complete borders of a single color) with scattered noise clusters outside.
    Count the number of separate noise clusters of each frame color.
    Place that many dots inside the frame's middle row at rightmost odd-indexed interior positions.
    """
    grid = [list(row) for row in grid]
    R, C = len(grid), len(grid[0])
    result = copy.deepcopy(grid)
    flat = [grid[r][c] for r in range(R) for c in range(C)]
    cnt = Counter(flat)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(r, c) for r in range(R) for c in range(C) if grid[r][c] != bg]
    cell_set = set(non_bg)
    visited = set()
    components = []
    for r, c in non_bg:
        if (r, c) not in visited:
            comp = []
            color = grid[r][c]
            stk = [(r, c)]
            while stk:
                r2, c2 = stk.pop()
                if (r2, c2) in visited:
                    continue
                visited.add((r2, c2))
                if grid[r2][c2] == color:
                    comp.append((r2, c2))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r2 + dr, c2 + dc
                        if (nr, nc) in cell_set and (nr, nc) not in visited:
                            stk.append((nr, nc))
            if comp:
                components.append((color, comp))
    frames = []
    for color, comp in components:
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        is_frame = all(r == min_r or r == max_r or c == min_c or c == max_c for r, c in comp)
        if is_frame:
            expected = set()
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if r == min_r or r == max_r or c == min_c or c == max_c:
                        expected.add((r, c))
            if set(comp) == expected and max_r - min_r >= 3 and max_c - min_c >= 3:
                frames.append((color, min_r, max_r, min_c, max_c))
    frame_cell_set = set()
    for color, min_r, max_r, min_c, max_c in frames:
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    frame_cell_set.add((r, c))
    pure_noise = [(r, c, grid[r][c]) for r in range(R) for c in range(C)
                  if grid[r][c] != bg and (r, c) not in frame_cell_set]
    noise_by_pos = {(r, c): v for r, c, v in pure_noise}
    noise_set = set(noise_by_pos)
    visited = set()
    noise_groups_final = []
    for pos in noise_set:
        if pos not in visited:
            color = noise_by_pos[pos]
            grp = []
            stk = [pos]
            while stk:
                n = stk.pop()
                if n in visited:
                    continue
                visited.add(n)
                if noise_by_pos.get(n) == color:
                    grp.append(n)
                    r, c = n
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nb = (r + dr, c + dc)
                        if nb in noise_set and nb not in visited:
                            stk.append(nb)
            if grp:
                noise_groups_final.append((color, grp))
    noise_count = Counter()
    for color, grp in noise_groups_final:
        noise_count[color] += 1
    for r, c, v in pure_noise:
        result[r][c] = bg
    frame_by_color = {color: (min_r, max_r, min_c, max_c) for color, min_r, max_r, min_c, max_c in frames}
    for color, (min_r, max_r, min_c, max_c) in frame_by_color.items():
        n = noise_count.get(color, 0)
        if n == 0:
            continue
        int_rows = list(range(min_r + 1, max_r))
        mid_row = int_rows[len(int_rows) // 2]
        int_cols = list(range(min_c + 1, max_c))
        int_width = len(int_cols)
        odd_positions = list(range(1, int_width, 2))
        selected = odd_positions[-n:] if n <= len(odd_positions) else odd_positions
        for pos in selected:
            col = min_c + 1 + pos
            result[mid_row][col] = color
    return result


def solve_97d7923e(grid):
    """
    Vertical 'bar' structures: top_marker + body + bottom_marker in a column (bg=0).
    Header cells (non-bar non-bg cells) of a given marker color count as K.
    The K-th longest bar with that marker color gets its body replaced by marker color.
    """
    grid = [list(row) for row in grid]
    R, C = len(grid), len(grid[0])
    result = copy.deepcopy(grid)
    bars = []
    for c in range(C):
        col = [grid[r][c] for r in range(R)]
        r = 0
        while r < R:
            if col[r] != 0:
                marker = col[r]
                r2 = r + 1
                while r2 < R and col[r2] == 0:
                    r2 += 1
                if r2 >= R:
                    r += 1
                    continue
                if col[r2] == marker:
                    r = r2 + 1
                    continue
                body_color = col[r2]
                body_start = r2
                r3 = r2
                while r3 < R and col[r3] == body_color:
                    r3 += 1
                body_end = r3 - 1
                r4 = r3
                while r4 < R and col[r4] == 0:
                    r4 += 1
                if r4 < R and col[r4] == marker:
                    bars.append((c, r, body_start, body_end, r4, marker, body_color))
                    r = r4 + 1
                else:
                    r += 1
            else:
                r += 1
    bar_cells = set()
    for bar in bars:
        col, top_r, bs, be, bot_r, mk, bc = bar
        for r in range(top_r, bot_r + 1):
            bar_cells.add((r, col))
    header_cells = {}
    for r in range(R):
        for c in range(C):
            if grid[r][c] != 0 and (r, c) not in bar_cells:
                color = grid[r][c]
                if color not in header_cells:
                    header_cells[color] = []
                header_cells[color].append((r, c))
    bars_by_marker = {}
    for bar in bars:
        col, top_r, bs, be, bot_r, mk, bc = bar
        if mk not in bars_by_marker:
            bars_by_marker[mk] = []
        body_len = be - bs + 1
        bars_by_marker[mk].append((body_len, col, bs, be, mk))
    for marker_color, hcells in header_cells.items():
        k = len(hcells)
        if marker_color not in bars_by_marker:
            continue
        sorted_bars = sorted(bars_by_marker[marker_color], key=lambda x: x[0], reverse=True)
        if k > len(sorted_bars):
            continue
        target = sorted_bars[k - 1]
        body_len, col, bs, be, mk = target
        for r in range(bs, be + 1):
            result[r][col] = marker_color
    return result


if __name__ == '__main__':
    import json
    tasks = [
        ('8b7bacbf', solve_8b7bacbf),
        ('8b9c3697', solve_8b9c3697),
        ('8f215267', solve_8f215267),
        ('97d7923e', solve_97d7923e),
    ]
    for task_id, solve_fn in tasks:
        with open(f'arc_agi_2_data/evaluation/{task_id}.json') as f:
            data = json.load(f)
        n = len(data['train'])
        passed = sum(1 for pair in data['train'] if solve_fn(pair['input']) == pair['output'])
        print(f'{task_id}: {passed}/{n}')
