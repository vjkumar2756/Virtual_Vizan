"""
Depth First Search (DFS) Implementation.
Explores the deepest nodes first using a LIFO strategy (stack).
"""

import time
from typing import Any, List, Optional
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def depth_first_search(
    graph: SearchGraph,
    start: str,
    goal: str,
    max_depth: Optional[int] = None
) -> SearchResult:
    """
    Executes Depth First Search from start to goal.

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label
        max_depth: Optional depth cutoff limit

    Returns:
        SearchResult containing the path, explored order, cost, and execution telemetry.
    """
    t0 = time.perf_counter()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="Depth First Search (DFS)",
            path=[],
            explored_nodes=[],
            path_cost=0.0,
            nodes_expanded=0,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            message=f"Start node '{start}' does not exist in graph."
        )

    if goal not in graph.nodes():
        return SearchResult(
            algorithm="Depth First Search (DFS)",
            path=[],
            explored_nodes=[],
            path_cost=0.0,
            nodes_expanded=0,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            message=f"Goal node '{goal}' does not exist in graph."
        )

    # Edge case: start is already goal
    if start == goal:
        return SearchResult(
            algorithm="Depth First Search (DFS)",
            path=[start],
            explored_nodes=[start],
            path_cost=0.0,
            nodes_expanded=1,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            step_history=[{"step": 1, "current": start, "frontier": [], "explored": [start], "action": "Goal found immediately"}],
            success=True,
            message="Start node is the goal node."
        )

    # Stack holds tuples: (current_node, path_so_far, cost_so_far, current_depth)
    stack: List[tuple[str, List[str], float, int]] = [(start, [start], 0.0, 0)]
    visited = set()
    explored_order: List[str] = []
    step_history: List[dict] = []
    step_num = 0

    while stack:
        current, path, cost, depth = stack.pop()
        step_num += 1

        if current in visited:
            continue

        visited.add(current)
        explored_order.append(current)

        step_history.append({
            "step": step_num,
            "current": current,
            "depth": depth,
            "frontier": [item[0] for item in stack],
            "explored": list(explored_order),
            "action": f"Expanded node '{current}' at depth {depth}"
        })

        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Depth First Search (DFS)",
                path=path,
                explored_nodes=explored_order,
                path_cost=cost,
                nodes_expanded=len(explored_order),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=True,
                message=f"Goal '{goal}' reached successfully."
            )

        if max_depth is not None and depth >= max_depth:
            continue

        # Reverse neighbors to maintain standard left-to-right exploration order when popped
        neighbors = sorted(list(graph.get_neighbors(current).items()), key=lambda x: str(x[0]), reverse=True)
        for neighbor, weight in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor], cost + weight, depth + 1))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(
        algorithm="Depth First Search (DFS)",
        path=[],
        explored_nodes=explored_order,
        path_cost=0.0,
        nodes_expanded=len(explored_order),
        execution_time_ms=elapsed_ms,
        step_history=step_history,
        success=False,
        message=f"Goal '{goal}' is unreachable from start '{start}'."
    )

