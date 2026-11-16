"""
Hill Climbing Search Implementation.
Local greedy search that iteratively moves to the neighboring state with the
steepest improvement in heuristic/objective value.
Detects local maxima, plateaus, and successful goal states.
"""

import time
from typing import Any, Callable, Dict, List, Optional
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def hill_climbing_search(
    graph: SearchGraph,
    start: str,
    goal: str,
    max_iterations: int = 100,
    heuristic_fn: Optional[Callable[[str, str], float]] = None
) -> SearchResult:
    """
    Executes Steepest-Ascent Hill Climbing from start to goal.

    Minimizes the heuristic distance h(node, goal), which corresponds to
    maximizing the objective -h(node, goal).

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label
        max_iterations: Maximum steps before halting
        heuristic_fn: Heuristic evaluating distance to goal

    Returns:
        SearchResult containing the trajectory, evaluated nodes, cost, and peak diagnosis.
    """
    t0 = time.perf_counter()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="Hill Climbing",
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
            algorithm="Hill Climbing",
            path=[],
            explored_nodes=[],
            path_cost=0.0,
            nodes_expanded=0,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            success=False,
            message=f"Goal node '{goal}' does not exist in graph."
        )

    if heuristic_fn is None:
        heuristic_fn = lambda u, v: graph.euclidean_heuristic(u, v)

    current = start
    path = [current]
    explored_nodes = [current]
    path_cost = 0.0
    step_history: List[dict] = []
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        current_h = heuristic_fn(current, goal)

        step_history.append({
            "step": iterations,
            "current": current,
            "h_val": round(current_h, 3),
            "action": f"At '{current}' with heuristic distance {current_h:.2f}"
        })

        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Hill Climbing",
                path=path,
                explored_nodes=explored_nodes,
                path_cost=path_cost,
                nodes_expanded=len(explored_nodes),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=True,
                message=f"Goal '{goal}' reached successfully in {iterations} iterations."
            )

        neighbors = graph.get_neighbors(current)
        if not neighbors:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Hill Climbing",
                path=path,
                explored_nodes=explored_nodes,
                path_cost=path_cost,
                nodes_expanded=len(explored_nodes),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=False,
                message=f"Terminated: Node '{current}' is a dead end with no outgoing edges."
            )

        # Find best neighbor (lowest heuristic distance to goal)
        best_neighbor = None
        best_h = current_h
        best_edge_cost = 0.0

        for neighbor, edge_weight in neighbors.items():
            if neighbor not in explored_nodes:
                explored_nodes.append(neighbor)
            nbr_h = heuristic_fn(neighbor, goal)
            if nbr_h < best_h:
                best_h = nbr_h
                best_neighbor = neighbor
                best_edge_cost = edge_weight

        # Check for local optimum or plateau
        if best_neighbor is None or best_h >= current_h:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Hill Climbing",
                path=path,
                explored_nodes=explored_nodes,
                path_cost=path_cost,
                nodes_expanded=len(explored_nodes),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=False,
                message=f"Stuck in local optimum / plateau at node '{current}' (h={current_h:.2f}). No strictly better neighbor exists."
            )

        current = best_neighbor
        path.append(current)
        path_cost += best_edge_cost

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(
        algorithm="Hill Climbing",
        path=path,
        explored_nodes=explored_nodes,
        path_cost=path_cost,
        nodes_expanded=len(explored_nodes),
        execution_time_ms=elapsed_ms,
        step_history=step_history,
        success=False,
        message=f"Maximum iteration limit ({max_iterations}) reached without attaining goal."
    )

