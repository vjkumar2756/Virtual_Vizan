"""
Iterative Deepening Search (IDS) Implementation.
Repeatedly applies Depth-Limited Search with increasing depth bounds (0, 1, 2, ...).
Combines the low space complexity of DFS with the completeness/optimality of BFS.
"""

import time
from typing import Any, List, Optional, Set, Tuple
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def iterative_deepening_search(
    graph: SearchGraph,
    start: str,
    goal: str,
    max_search_depth: int = 50
) -> SearchResult:
    """
    Executes Iterative Deepening Search from start to goal.

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label
        max_search_depth: Upper bound on depth iteration to prevent infinite loops on infinite spaces

    Returns:
        SearchResult containing the path, aggregate explored nodes, cost, and telemetry.
    """
    t0 = time.perf_counter()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="Iterative Deepening Search (IDS)",
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
            algorithm="Iterative Deepening Search (IDS)",
            path=[],
            explored_nodes=[],
            path_cost=0.0,
            nodes_expanded=0,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            message=f"Goal node '{goal}' does not exist in graph."
        )

    if start == goal:
        return SearchResult(
            algorithm="Iterative Deepening Search (IDS)",
            path=[start],
            explored_nodes=[start],
            path_cost=0.0,
            nodes_expanded=1,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            step_history=[{"depth_limit": 0, "current": start, "action": "Goal found at root"}],
            success=True,
            message="Start node is the goal node."
        )

    all_explored_nodes: List[str] = []
    total_expanded = 0
    step_history: List[dict] = []

    def depth_limited_search(
        current: str,
        target: str,
        limit: int,
        path: List[str],
        cost: float,
        visited_on_branch: Set[str]
    ) -> Tuple[Optional[List[str]], Optional[float], bool]:
        """Recursive DLS. Returns (path, cost, remaining_nodes_cutoff)."""
        nonlocal total_expanded
        total_expanded += 1
        if current not in all_explored_nodes:
            all_explored_nodes.append(current)

        step_history.append({
            "depth_limit": limit,
            "current": current,
            "path": list(path),
            "action": f"DLS depth_limit={limit}: visiting '{current}'"
        })

        if current == target:
            return path, cost, False

        if limit <= 0:
            return None, None, True  # Cutoff reached

        any_cutoff = False
        neighbors = sorted(list(graph.get_neighbors(current).items()), key=lambda x: str(x[0]))

        for neighbor, weight in neighbors:
            if neighbor not in visited_on_branch:
                res_path, res_cost, cutoff_occurred = depth_limited_search(
                    neighbor,
                    target,
                    limit - 1,
                    path + [neighbor],
                    cost + weight,
                    visited_on_branch | {neighbor}
                )
                if res_path is not None:
                    return res_path, res_cost, False
                if cutoff_occurred:
                    any_cutoff = True

        return None, None, any_cutoff

    for depth in range(max_search_depth + 1):
        found_path, found_cost, cutoff = depth_limited_search(
            start, goal, depth, [start], 0.0, {start}
        )
        if found_path is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Iterative Deepening Search (IDS)",
                path=found_path,
                explored_nodes=all_explored_nodes,
                path_cost=found_cost if found_cost is not None else 0.0,
                nodes_expanded=total_expanded,
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=True,
                message=f"Goal '{goal}' reached at depth bound {depth}."
            )
        if not cutoff:
            # Entire graph was traversed within depth without encountering cutoff: goal unreachable
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(
        algorithm="Iterative Deepening Search (IDS)",
        path=[],
        explored_nodes=all_explored_nodes,
        path_cost=0.0,
        nodes_expanded=total_expanded,
        execution_time_ms=elapsed_ms,
        step_history=step_history,
        success=False,
        message=f"Goal '{goal}' is unreachable from start '{start}' within max depth limit."
    )

