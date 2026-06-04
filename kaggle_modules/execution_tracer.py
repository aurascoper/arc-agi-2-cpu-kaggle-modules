#!/usr/bin/env python3
"""
execution_tracer.py -- Tree-structured execution tracer for DSL function calls.

Inspired by the ABPR paper (arXiv 2603.20334) on Algorithmic Program Debugging
for ARC-AGI-2. The key insight: when a synthesized transform() fails, flat error
logs are insufficient. A tree of intermediate computations -- showing what each
function received and returned -- lets the LLM do top-down bug localization:
find the deepest function whose output is wrong but whose sub-calls are all
correct.

Usage:
    from execution_tracer import ExecutionTracer

    tracer = ExecutionTracer()
    result, root = tracer.trace_transform(code, input_grid, dsl_namespace)
    if root:
        print(tracer.format_trace(root, expected_output=expected, actual_output=result))
"""

from __future__ import annotations

import functools
import threading
import time
import types
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraceNode:
    """One node in the execution trace tree."""
    function_name: str
    args_summary: str        # compact repr of arguments (truncated grids)
    result_summary: str      # compact repr of return value
    children: list[TraceNode] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None  # if this call raised an exception
    _result_obj: object = field(default=None, repr=False)  # P4: actual result for fault localization


# ---------------------------------------------------------------------------
# Grid representation helpers
# ---------------------------------------------------------------------------

def _is_grid(val: Any) -> bool:
    """Check if a value looks like a grid (list[list[int]])."""
    if not isinstance(val, list) or len(val) == 0:
        return False
    if not isinstance(val[0], list):
        return False
    return all(isinstance(row, list) for row in val)


def _grid_summary(grid: list[list[int]], max_repr: int = 200) -> str:
    """Compact representation of a grid.

    Small grids (<=5x5): full literal [[1,0],[0,1]]
    Larger grids: <8x12 grid, 4 colors: {0:80, 1:10, 2:5, 3:1}>
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    if rows <= 5 and cols <= 5:
        literal = repr(grid)
        if len(literal) <= max_repr:
            return literal

    # Compact summary for larger grids
    counts = Counter(cell for row in grid for cell in row)
    sorted_colors = sorted(counts.items(), key=lambda x: -x[1])
    color_str = ", ".join(f"{c}:{n}" for c, n in sorted_colors)
    return f"<{rows}x{cols} grid, {len(counts)} colors: {{{color_str}}}>"


def _compact_repr(val: Any, max_len: int = 200) -> str:
    """Compact representation of any value for trace display."""
    if val is None:
        return "None"
    if _is_grid(val):
        return _grid_summary(val, max_len)
    if isinstance(val, list):
        # List of objects / coords / grids
        if len(val) == 0:
            return "[]"
        first = val[0]
        if isinstance(first, tuple) and len(first) == 2:
            # Looks like coordinate list
            if len(val) <= 8:
                return repr(val)
            return f"[{len(val)} coords, e.g. {val[0]}..{val[-1]}]"
        if _is_grid(first):
            summaries = [_grid_summary(g, 60) for g in val[:4]]
            suffix = f", ...+{len(val)-4}" if len(val) > 4 else ""
            return f"[{', '.join(summaries)}{suffix}]"
        # Generic list
        r = repr(val)
        if len(r) <= max_len:
            return r
        return f"[{len(val)} items]"
    if isinstance(val, tuple):
        r = repr(val)
        if len(r) <= max_len:
            return r
        return f"({len(val)}-tuple)"
    if isinstance(val, dict):
        r = repr(val)
        if len(r) <= max_len:
            return r
        return f"{{{len(val)} keys}}"
    # numpy arrays
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            if val.ndim == 2:
                return f"<np {val.shape[0]}x{val.shape[1]} array>"
            return f"<np array shape={val.shape}>"
    except ImportError:
        pass
    r = repr(val)
    if len(r) > max_len:
        return r[:max_len - 3] + "..."
    return r


def _compact_args(args: tuple, kwargs: dict, max_len: int = 200) -> str:
    """Summarize function arguments compactly."""
    parts = [_compact_repr(a, max_len // max(len(args) + len(kwargs), 1)) for a in args]
    for k, v in kwargs.items():
        parts.append(f"{k}={_compact_repr(v, 60)}")
    joined = ", ".join(parts)
    if len(joined) > max_len:
        joined = joined[:max_len - 3] + "..."
    return joined


# ---------------------------------------------------------------------------
# Thread-local trace stack
# ---------------------------------------------------------------------------

_trace_local = threading.local()


def _get_stack() -> list[TraceNode]:
    if not hasattr(_trace_local, "stack"):
        _trace_local.stack = []
    return _trace_local.stack


def _get_depth() -> int:
    if not hasattr(_trace_local, "depth"):
        _trace_local.depth = 0
    return _trace_local.depth


def _set_depth(d: int):
    _trace_local.depth = d


# ---------------------------------------------------------------------------
# ExecutionTracer
# ---------------------------------------------------------------------------

class ExecutionTracer:
    """Instruments DSL function calls to build a trace tree."""

    def __init__(self, max_depth: int = 8, max_grid_repr: int = 200,
                 timeout: float = 3.0):
        """
        Args:
            max_depth: Maximum call-tree depth to trace. Beyond this, calls
                       execute normally but are not recorded.
            max_grid_repr: Character limit for grid literal representations.
            timeout: Hard timeout in seconds for the traced execution.
        """
        self.max_depth = max_depth
        self.max_grid_repr = max_grid_repr
        self.timeout = timeout

    # -- Instrumentation ---------------------------------------------------

    def _make_wrapper(self, fn, name: str):
        """Create a tracing wrapper for a single DSL function."""
        max_depth = self.max_depth
        max_repr = self.max_grid_repr

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            depth = _get_depth()
            if depth >= max_depth:
                # Past depth limit -- execute without tracing
                return fn(*args, **kwargs)

            node = TraceNode(
                function_name=name,
                args_summary=_compact_args(args, kwargs, max_repr),
                result_summary="<pending>",
            )

            stack = _get_stack()
            if stack:
                stack[-1].children.append(node)

            stack.append(node)
            _set_depth(depth + 1)
            t0 = time.perf_counter()

            try:
                result = fn(*args, **kwargs)
                node.result_summary = _compact_repr(result, max_repr)
                # P4: store actual result for fault localization (depth ≤ 3 only)
                if depth <= 3 and _is_grid(result):
                    node._result_obj = result
                return result
            except Exception as exc:
                node.error = f"{type(exc).__name__}: {exc}"
                node.result_summary = f"RAISED {type(exc).__name__}"
                raise
            finally:
                node.duration_ms = (time.perf_counter() - t0) * 1000
                stack.pop()
                _set_depth(depth)

        wrapper._tracer_original = fn
        return wrapper

    def _instrument_namespace(self, ns: dict) -> dict:
        """Wrap every callable in the namespace with tracing.

        Returns a new namespace dict with instrumented versions. Skips
        builtins, modules, and type objects to avoid interference.
        """
        instrumented = {}
        for name, obj in ns.items():
            if name.startswith("_"):
                instrumented[name] = obj
                continue
            if isinstance(obj, (type, types.ModuleType)):
                instrumented[name] = obj
                continue
            if callable(obj) and not isinstance(obj, type):
                instrumented[name] = self._make_wrapper(obj, name)
            else:
                instrumented[name] = obj
        return instrumented

    # -- Main entry point --------------------------------------------------

    def trace_transform(self, code: str, input_grid: list[list[int]],
                        dsl_namespace: dict) -> tuple[Any, TraceNode | None]:
        """Execute a transform() function with tracing enabled.

        The code must define a `transform(input_grid)` function that uses
        DSL helpers available in dsl_namespace.

        Returns (result, root_trace_node).
        result is the transform output (or None on error).
        root_trace_node is the full call tree.
        """
        # Instrument the DSL namespace
        traced_ns = self._instrument_namespace(dsl_namespace)

        # Build an execution namespace that includes traced DSL + builtins
        exec_ns = dict(traced_ns)
        exec_ns.setdefault("__builtins__", __builtins__)

        # Compile and exec the user code (defines transform())
        try:
            compiled = compile(code, "<transform>", "exec")
            exec(compiled, exec_ns)
        except Exception as e:
            error_node = TraceNode(
                function_name="<compile>",
                args_summary="",
                result_summary=f"RAISED {type(e).__name__}",
                error=f"{type(e).__name__}: {e}",
            )
            return None, error_node

        transform_fn = exec_ns.get("transform")
        if transform_fn is None:
            error_node = TraceNode(
                function_name="<missing>",
                args_summary="",
                result_summary="No transform() defined",
                error="Code does not define a transform() function",
            )
            return None, error_node

        # Wrap transform itself so it becomes the root node
        wrapped_transform = self._make_wrapper(transform_fn, "transform")

        # Execute with timeout in a separate thread
        result_holder: list[Any] = [None]
        error_holder: list[str | None] = [None]
        root_holder: list[TraceNode | None] = [None]

        def _run():
            # Reset thread-local state for this thread
            _trace_local.stack = []
            _trace_local.depth = 0

            # Create a sentinel root to capture the transform node
            sentinel = TraceNode(
                function_name="<sentinel>",
                args_summary="",
                result_summary="",
            )
            _get_stack().append(sentinel)

            try:
                result_holder[0] = wrapped_transform(input_grid)
            except Exception as exc:
                error_holder[0] = f"{type(exc).__name__}: {exc}"
            finally:
                # The transform node will be the first child of sentinel
                if sentinel.children:
                    root_holder[0] = sentinel.children[0]

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self.timeout)

        if t.is_alive():
            timeout_node = TraceNode(
                function_name="transform",
                args_summary=_compact_repr(input_grid, self.max_grid_repr),
                result_summary="TIMEOUT",
                error=f"Execution exceeded {self.timeout}s timeout",
            )
            return None, timeout_node

        root = root_holder[0]
        if root is None and error_holder[0]:
            root = TraceNode(
                function_name="transform",
                args_summary=_compact_repr(input_grid, self.max_grid_repr),
                result_summary=f"RAISED",
                error=error_holder[0],
            )

        return result_holder[0], root

    # -- Formatting --------------------------------------------------------

    def format_trace(self, root: TraceNode,
                     expected_output: list[list[int]] | None = None,
                     actual_output: list[list[int]] | None = None) -> str:
        """Format trace tree as indented text for LLM consumption.

        Includes APD-style bug localization instructions.
        """
        lines: list[str] = []
        lines.append("## EXECUTION TRACE (APD-style debugging)")
        lines.append("")

        # Input / expected / actual summary
        if root.args_summary:
            lines.append(f"Input: {root.args_summary}")
        if expected_output is not None:
            lines.append(f"Expected: {_compact_repr(expected_output, self.max_grid_repr)}")
        if actual_output is not None:
            exp_repr = _compact_repr(expected_output, self.max_grid_repr) if expected_output else "?"
            act_repr = _compact_repr(actual_output, self.max_grid_repr)
            match = (expected_output == actual_output) if expected_output is not None else None
            suffix = ""
            if match is True:
                suffix = " <- MATCH"
            elif match is False:
                suffix = " <- MISMATCH"
            lines.append(f"Actual: {act_repr}{suffix}")
        lines.append("")

        # Render the tree
        self._render_tree(root, lines, prefix="", is_last=True, is_root=True)

        # Bug localization instructions
        lines.append("")
        lines.append("## BUG LOCALIZATION INSTRUCTIONS")
        lines.append("Walk the trace top-down:")
        lines.append("1. Is the root output correct? Compare transform() result to expected.")
        lines.append("2. If wrong, check each child call: is its output plausible?")
        lines.append("3. Find the deepest call whose output is WRONG but whose")
        lines.append("   children all returned CORRECT results.")
        lines.append("4. That function (or the logic connecting its children) is the bug site.")
        lines.append("5. Fix that function's logic or its arguments, not higher-level code,")
        lines.append("   unless the arguments themselves are wrong.")

        return "\n".join(lines)

    # -- P4: Localized fault debugging -----------------------------------

    def localize_fault(self, root: TraceNode,
                       expected_output: list[list[int]] | None = None) -> str:
        """Walk trace tree to find deepest node whose output diverges from expected.

        Returns a human-readable fault description for injection into the
        diagnostic prompt. The localized fault helps the LLM target the specific
        sub-step that went wrong rather than rewriting the entire program.
        """
        if expected_output is None:
            return "Fault not localized — no expected output provided"

        def _pixel_accuracy(grid_a, grid_b) -> float:
            """Compute pixel-level accuracy between two grids."""
            if not grid_a or not grid_b:
                return 0.0
            rows_a, rows_b = len(grid_a), len(grid_b)
            cols_a = len(grid_a[0]) if rows_a else 0
            cols_b = len(grid_b[0]) if rows_b else 0
            if rows_a != rows_b or cols_a != cols_b:
                return 0.0
            total = rows_a * cols_a
            if total == 0:
                return 1.0
            match = sum(
                1 for r in range(rows_a) for c in range(cols_a)
                if grid_a[r][c] == grid_b[r][c]
            )
            return match / total

        def _to_list_grid(obj):
            """Normalize grid to list[list[int]]."""
            if obj is None:
                return None
            try:
                return [list(row) for row in obj]
            except (TypeError, ValueError):
                return None

        best_fault = None
        best_depth = -1

        def _walk(node: TraceNode, depth: int = 0):
            nonlocal best_fault, best_depth
            result = _to_list_grid(node._result_obj)
            if result is None:
                # Can't check — recurse into children
                for child in node.children:
                    _walk(child, depth + 1)
                return

            pa = _pixel_accuracy(result, expected_output)
            if pa >= 0.999:
                return  # This node is fine

            # This node's output is wrong. Check if children are all OK.
            children_ok = True
            for child in node.children:
                child_result = _to_list_grid(child._result_obj)
                if child_result is not None:
                    child_pa = _pixel_accuracy(child_result, expected_output)
                    if child_pa < 0.5:
                        children_ok = False
                        break

            if children_ok and depth > best_depth:
                r_shape = f"{len(result)}x{len(result[0])}" if result and result[0] else "?"
                e_shape = f"{len(expected_output)}x{len(expected_output[0])}" if expected_output and expected_output[0] else "?"
                best_fault = (
                    f"Fault at {node.function_name}({node.args_summary[:60]}): "
                    f"returned {r_shape} grid, PA={pa:.2f} vs expected {e_shape}. "
                    f"Children all returned plausible results."
                )
                best_depth = depth

            # Recurse deeper
            for child in node.children:
                _walk(child, depth + 1)

        _walk(root)
        return best_fault or "Fault not localized — output differs globally"

    def _render_tree(self, node: TraceNode, lines: list[str],
                     prefix: str, is_last: bool, is_root: bool = False):
        """Recursively render a trace node as a tree with box-drawing chars."""
        # Build the connector
        if is_root:
            connector = ""
            child_prefix = ""
        else:
            connector = prefix + ("└── " if is_last else "├── ")
            child_prefix = prefix + ("    " if is_last else "│   ")

        # Build the line: name(args) -> result [timing] [error]
        parts = [f"{node.function_name}({node.args_summary})"]
        parts.append(f" -> {node.result_summary}")
        if node.duration_ms >= 1.0:
            parts.append(f" [{node.duration_ms:.0f}ms]")
        elif node.duration_ms >= 0.1:
            parts.append(f" [{node.duration_ms:.1f}ms]")
        if node.error:
            parts.append(f"  !! {node.error}")

        lines.append(f"{connector}{''.join(parts)}")

        # Render children
        for i, child in enumerate(node.children):
            is_child_last = (i == len(node.children) - 1)
            self._render_tree(child, lines, child_prefix, is_child_last)


# ---------------------------------------------------------------------------
# Standalone helpers for integration
# ---------------------------------------------------------------------------

def load_dsl_and_trace(dsl_path: str, code: str, input_grid: list[list[int]],
                       expected_output: list[list[int]] | None = None,
                       timeout: float = 3.0) -> str:
    """Convenience: load DSL from file, trace code, return formatted string.

    Meant for quick integration into beam_search_local.py or evolve_qwen_arc.py.
    """
    import re
    from copy import deepcopy

    dsl_text = open(dsl_path).read()
    # Execute dsl.py to get HELPER_CODE_PREFIX
    ns: dict = {}
    exec(compile(dsl_text, dsl_path, "exec"), ns)
    helper_code = ns.get("HELPER_CODE_PREFIX", "")

    # Build DSL namespace
    import numpy as np
    dsl_ns: dict = {"__builtins__": __builtins__, "np": np, "deepcopy": deepcopy}
    exec(helper_code, dsl_ns)

    tracer = ExecutionTracer(timeout=timeout)
    result, root = tracer.trace_transform(code, input_grid, dsl_ns)
    if root is None:
        return "## EXECUTION TRACE\nNo trace produced."
    return tracer.format_trace(root, expected_output=expected_output,
                               actual_output=result)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Build a small mock DSL namespace --
    def find_objects(grid, background=0):
        """Find connected same-color components."""
        rows, cols = len(grid), len(grid[0])
        visited = set()
        objs = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != background and (r, c) not in visited:
                    color = grid[r][c]
                    component = []
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited:
                            continue
                        if not (0 <= cr < rows and 0 <= cc < cols):
                            continue
                        if grid[cr][cc] != color:
                            continue
                        visited.add((cr, cc))
                        component.append((cr, cc))
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            stack.append((cr + dr, cc + dc))
                    objs.append({"color": color, "cells": component})
        return objs

    def sort_by_size(objects):
        """Sort objects by number of cells, ascending."""
        return sorted(objects, key=lambda o: len(o["cells"]))

    def recolor(grid, color_map):
        """Replace colors in grid according to color_map dict."""
        return [[color_map.get(cell, cell) for cell in row] for row in grid]

    def copy_grid(grid):
        return [row[:] for row in grid]

    dsl_ns = {
        "__builtins__": __builtins__,
        "find_objects": find_objects,
        "sort_by_size": sort_by_size,
        "recolor": recolor,
        "copy_grid": copy_grid,
    }

    # -- A transform that has a deliberate bug in the color mapping --
    code = """
def transform(input_grid):
    objects = find_objects(input_grid, background=0)
    sorted_objs = sort_by_size(objects)
    # Bug: mapping 1->2 is correct, but we forgot to map 3->2
    color_map = {1: 2}
    result = recolor(input_grid, color_map)
    return result
"""

    input_grid = [
        [0, 1, 0, 0],
        [0, 1, 3, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
    ]

    expected_output = [
        [0, 2, 0, 0],
        [0, 2, 2, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
    ]

    tracer = ExecutionTracer()
    result, root = tracer.trace_transform(code, input_grid, dsl_ns)

    print("=" * 70)
    print("DEMO: Tracing a buggy transform")
    print("=" * 70)
    print()
    if root:
        print(tracer.format_trace(root, expected_output=expected_output,
                                  actual_output=result))
    else:
        print("No trace produced.")
    print()

    # -- Also demo an exception case --
    error_code = """
def transform(input_grid):
    objects = find_objects(input_grid, background=0)
    # Bug: accessing index that doesn't exist
    biggest = objects[99]
    return input_grid
"""

    print("=" * 70)
    print("DEMO: Tracing a transform that raises an exception")
    print("=" * 70)
    print()
    result2, root2 = tracer.trace_transform(error_code, input_grid, dsl_ns)
    if root2:
        print(tracer.format_trace(root2, expected_output=expected_output,
                                  actual_output=result2))
    print()

    # -- Demo with a larger grid --
    large_grid = [[((r + c) % 4) for c in range(10)] for r in range(10)]
    large_code = """
def transform(input_grid):
    g = copy_grid(input_grid)
    return recolor(g, {0: 5, 1: 5})
"""

    print("=" * 70)
    print("DEMO: Tracing with a larger grid (compact repr)")
    print("=" * 70)
    print()
    result3, root3 = tracer.trace_transform(large_code, large_grid, dsl_ns)
    if root3:
        print(tracer.format_trace(root3, actual_output=result3))
    print()
    print("All demos complete.")
