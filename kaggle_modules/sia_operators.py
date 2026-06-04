"""Quarantined ARC operator library — formalized from the Generator-Quality Transfer Audit (2026-06-04).

NOT integrated into the solver / Kaggle / notebook. Self-contained (no import of refine.py/common.py)
to avoid harness churn. Two audited reusable operators + the shared utility substrate.

Convention (matches the density-pilot candidate interface so it can later plug into execute_filter):
    op(train) -> transform | None
where train is [{"input": grid, "output": grid}, ...], transform(grid)->grid, and op returns None when
the operator does not fit the train pairs. All task-specific parameters are LEARNED from train inside op
(family re-derivation), so the same operator solves multiple tasks with different inferred params.
"""
from collections import Counter

# ---------- utility substrate (the primitives every audited program re-derived) ----------
def norm(g):
    return [list(row) for row in g]

def dims(g):
    return (len(g), len(g[0]) if g else 0)

def eq(a, b):
    return [list(r) for r in a] == [list(r) for r in b]

def bg(g):
    """Background color: 0 if present, else the most frequent value."""
    c = Counter(v for row in g for v in row)
    if 0 in c:
        return 0
    return c.most_common(1)[0][0] if c else 0

def connected_components(g, exclude_color, connectivity=4):
    """4- or 8-connected components of cells != exclude_color. Returns list of [(r,c), ...]."""
    H, W = dims(g)
    seen = [[False] * W for _ in range(H)]
    nbrs = ([(1, 0), (-1, 0), (0, 1), (0, -1)] if connectivity == 4
            else [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)])
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r][c] != exclude_color and not seen[r][c]:
                stack = [(r, c)]; seen[r][c] = True; cells = []
                while stack:
                    cr, cc = stack.pop(); cells.append((cr, cc))
                    for dr, dc in nbrs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and g[nr][nc] != exclude_color:
                            seen[nr][nc] = True; stack.append((nr, nc))
                comps.append(cells)
    return comps

# ---------- D4 dihedral group (for symmetry_tile_expand) ----------
def _identity(g):  return norm(g)
def _rot90(g):     return [list(row) for row in zip(*g[::-1])]        # clockwise
def _rot180(g):    return [row[::-1] for row in g[::-1]]
def _rot270(g):    return [list(row) for row in zip(*g)][::-1]        # counter-clockwise
def _flipH(g):     return [row[::-1] for row in g]
def _flipV(g):     return [row[:] for row in g[::-1]]
def _transpose(g): return [list(row) for row in zip(*g)]
def _antiT(g):     return [list(row) for row in zip(*[r[::-1] for r in g[::-1]])]
D4 = {"id": _identity, "rot90": _rot90, "rot180": _rot180, "rot270": _rot270,
      "flipH": _flipH, "flipV": _flipV, "transpose": _transpose, "antiT": _antiT}
_DIM_SWAP = {"rot90", "rot270", "transpose", "antiT"}  # require square input to fit a same-size quadrant

# ---------- Operator 1: symmetry_tile_expand (learned 2x2 D4 quadrant tiling) ----------
def op_symmetry_tile_expand(train):
    """N x M -> 2N x 2M, each quadrant a D4 transform of the input. The per-quadrant assignment is LEARNED
    from train (the kaleidoscope/rotate_quadrants/quad_rotation family is just different assignments)."""
    if not train:
        return None
    for p in train:
        ih, iw = dims(p["input"]); oh, ow = dims(p["output"])
        if ih == 0 or oh != 2 * ih or ow != 2 * iw:
            return None
    def quadrant(out, qr, qc, ih, iw):
        return [out[qr * ih + r][qc * iw: qc * iw + iw] for r in range(ih)]
    assign = {}
    for (qr, qc) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        chosen = None
        for name, fn in D4.items():
            ok = True
            for p in train:
                ih, iw = dims(p["input"])
                if name in _DIM_SWAP and ih != iw:
                    ok = False; break
                t = fn(norm(p["input"]))
                if dims(t) != (ih, iw) or not eq(t, quadrant(p["output"], qr, qc, ih, iw)):
                    ok = False; break
            if ok:
                chosen = name; break
        if chosen is None:
            return None
        assign[(qr, qc)] = chosen

    def transform(g):
        g = norm(g); ih, iw = dims(g)
        out = [[0] * (2 * iw) for _ in range(2 * ih)]
        for (qr, qc), name in assign.items():
            block = D4[name](g)
            for r in range(ih):
                for c in range(iw):
                    out[qr * ih + r][qc * iw + c] = block[r][c]
        return out
    transform.params = {"assign": {f"{k[0]}{k[1]}": v for k, v in assign.items()}}
    return transform

# ---------- Operator 2: region_fill_by_border_signature (learned divider-partition recolor) ----------
def _divider_candidates(g, b):
    """Colors (!= bg) forming at least one full row or full column — candidate divider lines."""
    H, W = dims(g); out = set()
    for color in set(v for row in g for v in row if v != b):
        if (any(all(g[r][c] == color for c in range(W)) for r in range(H))
                or any(all(g[r][c] == color for r in range(H)) for c in range(W))):
            out.add(color)
    return out

def _border_sig(cells, H, W):
    return (any(r == 0 for r, c in cells), any(r == H - 1 for r, c in cells),
            any(c == 0 for r, c in cells), any(c == W - 1 for r, c in cells))

def _make_region_fill_transform(divider, border_map, enclosed):
    def transform(g):
        g = norm(g); H, W = dims(g); out = [row[:] for row in g]
        for cells in connected_components(g, divider):
            sig = _border_sig(cells, H, W)
            if not any(sig) and enclosed is not None:
                for r, c in cells: out[r][c] = enclosed
            elif sig in border_map:
                for r, c in cells: out[r][c] = border_map[sig]
        return out
    transform.params = {"divider": divider, "border_map": {str(k): v for k, v in border_map.items()},
                        "enclosed": enclosed}
    return transform

def op_region_fill_by_border_signature(train):
    """Partition by a divider line (a color forming a full row/col in EVERY train input); recolor each
    region by a rule LEARNED from train: border-touch signature (top,bottom,left,right) -> color, plus a
    color for fully-enclosed regions. Deterministic: intersect divider candidates across all train pairs,
    then return the first (sorted) candidate whose learned map is train-exact (self-validating)."""
    if not train:
        return None
    for p in train:
        if dims(p["output"]) != dims(p["input"]):
            return None  # region-fill is a same-shape recolor; does not apply to shape-changing tasks
    candidates = set()
    for p in train:
        inp = norm(p["input"])
        candidates |= _divider_candidates(inp, bg(inp))  # union: a divider need only form a full line in SOME pair
    if not candidates:
        return None
    for divider in sorted(candidates):
        border_map = {}; enclosed = None; consistent = True
        for p in train:
            inp = norm(p["input"]); out = norm(p["output"]); b = bg(inp); H, W = dims(inp)
            for cells in connected_components(inp, divider):
                r0, c0 = cells[0]; oc = out[r0][c0]
                if oc == b:
                    continue  # region stays background; nothing learned
                sig = _border_sig(cells, H, W)
                if any(sig):
                    if sig in border_map and border_map[sig] != oc:
                        consistent = False
                    border_map[sig] = oc
                else:
                    enclosed = oc
        if not consistent or (not border_map and enclosed is None):
            continue
        t = _make_region_fill_transform(divider, border_map, enclosed)
        if all(eq(t(p["input"]), p["output"]) for p in train):
            return t
    return None

OPERATORS = {
    "symmetry_tile_expand": op_symmetry_tile_expand,
    "region_fill_by_border_signature": op_region_fill_by_border_signature,
}
