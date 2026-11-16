"""
A* Search Algorithm Implementation.
Informed best-first search evaluating nodes using f(n) = g(n) + h(n).
Guarantees optimality when the heuristic is admissible and consistent.
"""

import heapq
import time
from typing import Any, Callable, Dict, List, Optional
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def a_star_search(
    graph: SearchGraph,
    start: str,
    goal: str,
    heuristic_fn: Optional[Callable[[str, str], float]] = None
) -> SearchResult:
    """
    Executes A* Search from start to goal.

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label
        heuristic_fn: Optional callable taking (node, goal) and returning estimated cost h(node)

    Returns:
        SearchResult containing the optimal path, explored nodes, path cost, and telemetry.
    """
    t0 = time.perf_counter()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="A* Search",
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
            algorithm="A* Search",
            path=[],
            explored_nodes=[],
            path_cost=0.0,
            nodes_expanded=0,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            message=f"Goal node '{goal}' does not exist in graph."
        )

    # Heuristic resolution
    if heuristic_fn is None:
        heuristic_fn = lambda u, v: graph.euclidean_heuristic(u, v)

    if start == goal:
        return SearchResult(
            algorithm="A* Search",
            path=[start],
            explored_nodes=[start],
            path_cost=0.0,
            nodes_expanded=1,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            step_history=[{"step": 1, "current": start, "g": 0, "h": 0, "f": 0, "action": "Goal found immediately"}],
            success=True,
            message="Start node is the goal node."
        )

    # Priority queue stores tuples: (f_score, tie_breaker, current_node, path, g_score)
    tie_breaker = 0
    h_start = heuristic_fn(start, goal)
    open_set: List[tuple[float, int, str, List[str], float]] = [
        (h_start, tie_breaker, start, [start], 0.0)
    ]

    # Best known g_score for each node
    g_scores: Dict[str, float] = {start: 0.0}
    closed_set = set()
    explored_order: List[str] = []
    step_history: List[dict] = []
    step_num = 0

    while open_set:
        f_val, _, current, path, g_val = heapq.heappop(open_set)
        step_num += 1

        if current in closed_set:
            continue

        closed_set.add(current)
        explored_order.append(current)

        step_history.append({
            "step": step_num,
            "current": current,
            "g": round(g_val, 2),
            "h": round(heuristic_fn(current, goal), 2),
            "f": round(f_val, 2),
            "frontier_size": len(open_set),
            "explored": list(explored_order),
            "action": f"Expanded '{current}' with f={f_val:.2f} (g={g_val:.2f}, h={f_val - g_val:.2f})"
        })

        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="A* Search",
                path=path,
                explored_nodes=explored_order,
                path_cost=g_val,
                nodes_expanded=len(explored_order),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=True,
                message=f"Goal '{goal}' reached with optimal cost {g_val:.2f}."
            )

        for neighbor, edge_weight in graph.get_neighbors(current).items():
            tentative_g = g_val + edge_weight

            if tentative_g < g_scores.get(neighbor, float("inf")):
                g_scores[neighbor] = tentative_g
                h_val = heuristic_fn(neighbor, goal)
                f_score = tentative_g + h_val
                tie_breaker += 1
                heapq.heappush(open_set, (f_score, tie_breaker, neighbor, path + [neighbor], tentative_g))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(
        algorithm="A* Search",
        path=[],
        explored_nodes=explored_order,
        path_cost=0.0,
        nodes_expanded=len(explored_order),
        execution_time_ms=elapsed_ms,
        step_history=step_history,
        success=False,
        message=f"Goal '{goal}' is unreachable from start '{start}'."
    )

