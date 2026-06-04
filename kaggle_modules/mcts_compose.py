"""MCTS over DSL function compositions for ARC tasks.

UCB1-guided tree search over chains of grid→grid functions.
Complements mdl_compose.py (brute-force depth 1-3) by exploring
depth 4+ with intelligent selection, and ExeDec (LLM-predicted
intermediates) by discovering chains without LLM guidance.

Key design: each node is a partial composition [f1, f2, ...].
Expansion appends one composable function. Rollout appends 0-2
random functions then evaluates pixel accuracy. UCB1 guides
exploration vs exploitation.

Reference: MCTS for program synthesis (Simmons-Edler 2018, 1806.02932)
"""

from __future__ import annotations

import copy
import gc
import math
import os
import random
import signal
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent

# MCTS hyperparameters
UCB_C = 1.41           # Exploration constant (sqrt(2) is standard)
MAX_DEPTH = 6          # Maximum composition chain length (was 5, bumped for primary solver role)
DEFAULT_BUDGET = 300   # Number of MCTS iterations (was 80, bumped for primary solver role)

# Per-function timeout (seconds) — catches infinite-loop DSL functions
_FN_TIMEOUT = 2

# Functions that have timed out — skip them for the rest of the session
_blacklist: set[str] = set()

# ─────────────────────────────────────────────────────────────────────────
# Memory hygiene (Fix A) — env-gated, ANALYZE/MCTS-only.
# Default OFF; behavior is byte-identical to pre-patch when disabled.
# Verified at 5s burst sampling: spike fires within ~10s of MCTS launching
# its 300-iter loop, peak 4.7-5.4 GB, recovers to 200-700M plateau, then
# a second spike crosses jetsam threshold ~3 min later. In-loop cleanup +
# end-of-search cleanup target both the peak and the plateau.
# ─────────────────────────────────────────────────────────────────────────
_MEMORY_HYGIENE_ENABLED = (
    os.environ.get("ARC_MCTS_MEMORY_HYGIENE", "").lower() == "true"
)
_MCTS_CLEAR_EVERY = max(1, int(os.environ.get("ARC_MCTS_CLEAR_EVERY", "5")))


def _hygiene_cleanup() -> None:
    """gc.collect + mlx.core.clear_cache. Defensive: tolerates missing mlx,
    never raises. Cheap when ARC_MCTS_MEMORY_HYGIENE!=true (early return)."""
    if not _MEMORY_HYGIENE_ENABLED:
        return
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import mlx.core as _mx
        _mx.clear_cache()
    except Exception:
        pass


class _FnTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FnTimeout()


def _safe_call(fn, grid, fn_name: str = ""):
    """Call fn(grid) with a hard SIGALRM timeout. Blacklists repeat offenders."""
    if fn_name and fn_name in _blacklist:
        raise _FnTimeout(f"{fn_name} blacklisted")
    # signal.alarm only works on main thread
    on_main = threading.current_thread() is threading.main_thread()
    if on_main:
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(_FN_TIMEOUT)
    try:
        result = fn(grid)
    except _FnTimeout:
        if fn_name:
            _blacklist.add(fn_name)
            print(f"  [mcts] BLACKLISTED {fn_name} (timeout {_FN_TIMEOUT}s)")
            # Hygiene on timeout: the timed-out path may have allocated
            # tensors that won't be released until next GC. Free them now.
            _hygiene_cleanup()
        raise
    finally:
        if on_main:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    return result
DEFAULT_TIMEOUT = 45.0 # Seconds (was 15, bumped for primary solver role)
ROLLOUT_EXTRA = 2      # Max extra random functions appended during rollout


@dataclass
class MCTSNode:
    """Node in the MCTS composition tree."""
    chain: list[str]           # Function names applied so far
    parent: 'MCTSNode | None' = None
    children: dict[str, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    total_reward: float = 0.0
    _untried: list[str] | None = None  # Lazy init

    @property
    def depth(self) -> int:
        return len(self.chain)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb1(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.mean_reward
        explore = UCB_C * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore


# Cancel pairs — skip adjacent no-ops (imported concept from mdl_compose)
CANCEL_PAIRS = {
    ("rotate_cw", "rotate_ccw"),
    ("rotate_ccw", "rotate_cw"),
    ("mirror_h", "mirror_h"),
    ("mirror_v", "mirror_v"),
    ("transpose", "transpose"),
    ("rotate_180", "rotate_180"),
    ("flip_anti_diagonal", "flip_anti_diagonal"),
}


def _eval_chain_accuracy(chain_fns: list[tuple[str, callable]], task_data: dict) -> float:
    """Evaluate a function chain on all training pairs. Returns mean pixel accuracy.

    chain_fns: list of (name, fn) tuples for timeout tracking.
    Deep-copies inputs to avoid mutation from DSL functions.
    """
    pairs = task_data.get("train", [])
    if not pairs:
        return 0.0

    total_correct = 0
    total_pixels = 0

    for pair in pairs:
        grid = copy.deepcopy(pair["input"])
        expected = pair["output"]
        try:
            for name, fn in chain_fns:
                grid = _safe_call(fn, grid, fn_name=name)
            result = [list(row) for row in grid] if grid else []
            exp = [list(row) for row in expected]

            if not result or not exp:
                total_pixels += sum(len(row) for row in exp) if exp else 1
                continue

            if len(result) == len(exp) and all(len(r) == len(e) for r, e in zip(result, exp)):
                for r_row, e_row in zip(result, exp):
                    for r_val, e_val in zip(r_row, e_row):
                        total_pixels += 1
                        if r_val == e_val:
                            total_correct += 1
            else:
                total_pixels += sum(len(row) for row in exp)
        except (_FnTimeout, Exception):
            total_pixels += sum(len(row) for row in expected) if expected else 1

    return total_correct / total_pixels if total_pixels > 0 else 0.0


def _chain_passes(chain_fns: list[tuple[str, callable]], task_data: dict) -> bool:
    """Check if chain solves ALL training pairs exactly.

    chain_fns: list of (name, fn) tuples for timeout tracking.
    """
    for pair in task_data.get("train", []):
        grid = copy.deepcopy(pair["input"])
        try:
            for name, fn in chain_fns:
                grid = _safe_call(fn, grid, fn_name=name)
            result = [list(row) for row in grid] if grid else []
            expected = [list(row) for row in pair["output"]]
            if result != expected:
                return False
        except (_FnTimeout, Exception):
            return False
    return True


def _chain_to_code(func_names: list[str]) -> str:
    """Convert function name list to executable Python code."""
    lines = ["def transform(grid):"]
    for name in func_names:
        lines.append(f"    grid = {name}(grid)")
    lines.append("    return grid")
    return "\n".join(lines)


def _is_cancel(prev_name: str, next_name: str) -> bool:
    """Check if appending next_name after prev_name is a no-op."""
    return (prev_name, next_name) in CANCEL_PAIRS


def _local_refine(
    chain: list[str],
    chain_accuracy: float,
    fn_names: list[str],
    fn_lookup: dict[str, callable],
    task_data: dict,
    deadline: float,
) -> str | None:
    """Local refinement of a near-miss chain (>0.9 accuracy).

    Tries three strategies:
    1. Swap: replace each position with every other function
    2. Append: add one function to the end
    3. Remove: drop each function from the chain

    Returns code string if exact match found, else None.
    """
    print(f"  [mcts] Refining near-miss: {' -> '.join(chain[:4])} acc={chain_accuracy:.3f}")
    attempts = 0

    # Strategy 1: Swap each position
    for i in range(len(chain)):
        if time.monotonic() > deadline:
            break
        for replacement in fn_names:
            if replacement == chain[i]:
                continue
            if i > 0 and _is_cancel(chain[i - 1], replacement):
                continue
            if i < len(chain) - 1 and _is_cancel(replacement, chain[i + 1]):
                continue
            candidate = chain[:i] + [replacement] + chain[i + 1:]
            named = []
            valid = True
            for n in candidate:
                fn = fn_lookup.get(n)
                if fn is None or n in _blacklist:
                    valid = False
                    break
                named.append((n, fn))
            if valid and _chain_passes(named, task_data):
                print(f"  [mcts] REFINED (swap pos {i}): {' -> '.join(candidate)} "
                      f"({attempts} attempts)")
                return _chain_to_code(candidate)
            attempts += 1

    # Strategy 2: Append one function
    if len(chain) < MAX_DEPTH + 1 and time.monotonic() < deadline:
        last = chain[-1] if chain else None
        for extra in fn_names:
            if time.monotonic() > deadline:
                break
            if last and _is_cancel(last, extra):
                continue
            candidate = chain + [extra]
            named = []
            valid = True
            for n in candidate:
                fn = fn_lookup.get(n)
                if fn is None or n in _blacklist:
                    valid = False
                    break
                named.append((n, fn))
            if valid and _chain_passes(named, task_data):
                print(f"  [mcts] REFINED (append {extra}): {' -> '.join(candidate)} "
                      f"({attempts} attempts)")
                return _chain_to_code(candidate)
            attempts += 1

    # Strategy 3: Remove each function (shorter chain = better MDL)
    if len(chain) > 1 and time.monotonic() < deadline:
        for i in range(len(chain)):
            candidate = chain[:i] + chain[i + 1:]
            if not candidate:
                continue
            # Check cancel pairs at the join point
            if i > 0 and i < len(chain) and len(candidate) > i - 1:
                pass  # no cancel check needed after removal
            named = []
            valid = True
            for n in candidate:
                fn = fn_lookup.get(n)
                if fn is None or n in _blacklist:
                    valid = False
                    break
                named.append((n, fn))
            if valid and _chain_passes(named, task_data):
                print(f"  [mcts] REFINED (remove pos {i}): {' -> '.join(candidate)} "
                      f"({attempts} attempts)")
                return _chain_to_code(candidate)
            attempts += 1

    if attempts > 0:
        print(f"  [mcts] Refinement exhausted ({attempts} attempts, no exact match)")
    return None


def mcts_compose_search(
    task_data: dict,
    dsl_namespace: dict,
    budget: int = DEFAULT_BUDGET,
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    """MCTS search over function compositions.

    Returns "def transform(grid): ..." code string, or None.

    Args:
        task_data: ARC task dict with "train" key
        dsl_namespace: DSL namespace with callable functions
        budget: number of MCTS iterations
        timeout: wall-clock time limit in seconds
    """
    start = time.monotonic()
    deadline = start + timeout

    # Discover composable functions, excluding blacklisted
    from mdl_compose import discover_composable
    composable = discover_composable(dsl_namespace)
    if not composable:
        return None

    fn_names = [name for name, _ in composable if name not in _blacklist]
    fn_lookup = {name: fn for name, fn in composable if name not in _blacklist}
    if not fn_names:
        return None

    # Tier 2: restrict pool to top-60 by pixel-delta relevance on training pairs
    if len(fn_names) > 60:
        def _score_relevance(fn, fn_name, pairs, n=None):
            score = 0
            for pair in (pairs if n is None else pairs[:n]):
                try:
                    result = _safe_call(fn, copy.deepcopy(pair['input']), fn_name=fn_name)
                    out = pair['output']
                    inp = pair['input']
                    if not result or len(result) != len(out):
                        continue
                    if any(len(result[r]) != len(out[r]) for r in range(len(out))):
                        continue
                    for r, row in enumerate(result):
                        for c, val in enumerate(row):
                            if r < len(out) and c < len(out[r]):
                                # Reward changed-correctly pixels (2x weight)
                                if val == out[r][c] and val != inp[r][c]:
                                    score += 2
                                # Also reward dimension-matching + output-matching pixels
                                elif val == out[r][c]:
                                    score += 1
                except Exception:
                    pass
            return score
        train_pairs = task_data.get('train', [])
        scored = [(n, _score_relevance(fn_lookup[n], n, train_pairs)) for n in fn_names]
        scored.sort(key=lambda x: -x[1])
        # Keep top-60 by score; always include any with score>0 up to 60
        fn_names = [n for n, _ in scored[:60]]
        fn_lookup = {n: fn_lookup[n] for n in fn_names}

    # Root node = empty chain (identity transform)
    root = MCTSNode(chain=[])

    best_solution = None
    best_accuracy = 0.0

    for iteration in range(budget):
        if time.monotonic() > deadline:
            break

        # In-loop memory hygiene every N iterations (Fix A, env-gated).
        # The 5s burst trace showed transient 2-5GB allocations during
        # MCTS execution that aren't freed until the search reports.
        # Periodic gc + mlx.clear_cache keeps the peak from accumulating.
        if _MEMORY_HYGIENE_ENABLED and iteration > 0 and (iteration % _MCTS_CLEAR_EVERY == 0):
            _hygiene_cleanup()

        # --- SELECT: traverse tree using UCB1 ---
        node = root
        while node.children and node.depth < MAX_DEPTH:
            # If there are untried children, expand instead
            if node._untried is None:
                last = node.chain[-1] if node.chain else None
                node._untried = [
                    n for n in fn_names
                    if last is None or not _is_cancel(last, n)
                ]
            if node._untried:
                break  # Go to EXPAND
            # All children tried — select best UCB1
            best_ucb = -1.0
            best_child = None
            for child in node.children.values():
                ucb = child.ucb1(node.visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            if best_child is None:
                break
            node = best_child

        # --- EXPAND: add one untried child ---
        # Lazy-init _untried here too — SELECT only inits it for nodes that
        # already have children, so the root (and any freshly-visited leaf)
        # never gets initialized there.
        if node._untried is None:
            last = node.chain[-1] if node.chain else None
            node._untried = [
                n for n in fn_names
                if last is None or not _is_cancel(last, n)
            ]
        if node._untried and node.depth < MAX_DEPTH:
            # Pick random untried function
            fn_name = random.choice(node._untried)
            node._untried.remove(fn_name)
            child_chain = node.chain + [fn_name]
            child = MCTSNode(chain=child_chain, parent=node)
            node.children[fn_name] = child
            node = child

        # --- ROLLOUT: random playout from current node ---
        rollout_chain = list(node.chain)
        for _ in range(ROLLOUT_EXTRA):
            if len(rollout_chain) >= MAX_DEPTH:
                break
            last = rollout_chain[-1] if rollout_chain else None
            candidates = [
                n for n in fn_names
                if last is None or not _is_cancel(last, n)
            ]
            if not candidates:
                break
            rollout_chain.append(random.choice(candidates))

        # Evaluate rollout chain — build (name, fn) tuples
        chain_named = []
        valid = True
        for name in rollout_chain:
            if name in _blacklist:
                valid = False
                break
            fn = fn_lookup.get(name)
            if fn is None:
                valid = False
                break
            chain_named.append((name, fn))

        if valid and chain_named:
            # Check exact match first (fast path)
            if _chain_passes(chain_named, task_data):
                code = _chain_to_code(rollout_chain)
                print(f"  [mcts] SOLVED: {' -> '.join(rollout_chain)} "
                      f"(depth {len(rollout_chain)}, iter {iteration})")
                return code

            reward = _eval_chain_accuracy(chain_named, task_data)

            if reward > best_accuracy:
                best_accuracy = reward
                best_solution = rollout_chain
        else:
            reward = 0.0

        # Also check the node's own chain (without rollout extension)
        if len(node.chain) > 0:
            node_named = [(n, fn_lookup[n]) for n in node.chain
                          if n in fn_lookup and n not in _blacklist]
            if len(node_named) == len(node.chain):
                if _chain_passes(node_named, task_data):
                    code = _chain_to_code(node.chain)
                    print(f"  [mcts] SOLVED: {' -> '.join(node.chain)} "
                          f"(depth {len(node.chain)}, iter {iteration})")
                    return code
                node_reward = _eval_chain_accuracy(node_named, task_data)
                reward = max(reward, node_reward)

        # --- BACKPROPAGATE ---
        backprop_node = node
        while backprop_node is not None:
            backprop_node.visits += 1
            backprop_node.total_reward += reward
            backprop_node = backprop_node.parent

    # --- LOCAL REFINEMENT: when near-miss >0.9, try systematic variations ---
    if best_accuracy >= 0.9 and best_solution and time.monotonic() < deadline:
        refine_result = _local_refine(
            best_solution, best_accuracy, fn_names, fn_lookup,
            task_data, deadline,
        )
        if refine_result:
            return refine_result

    # End-of-search hygiene (Fix A). Plateau RSS sat at 200-700M for 3 min
    # after the spike — releasing here knocks that down before the next
    # round's HYPOTHESIZE/INVENT/ANALYZE cycle starts allocating again.
    _hygiene_cleanup()

    # No exact solution found
    if best_accuracy > 0 and best_solution:
        elapsed = time.monotonic() - start
        print(f"  [mcts] No exact match. Best: {' -> '.join(best_solution[:4])}{'...' if len(best_solution) > 4 else ''} "
              f"acc={best_accuracy:.3f} ({iteration+1} iters, {elapsed:.1f}s)")

    return None
