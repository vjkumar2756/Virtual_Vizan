"""
Breadth First Search (BFS) Implementation.
Explores the shallowest unexpanded nodes first using a FIFO queue.
"""

from collections import deque
import time
from typing import Any, List, Optional
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def breadth_first_search(
    graph: SearchGraph,
    start: str,
    goal: str
) -> SearchResult:
    """
    Executes Breadth First Search from start to goal.

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label

    Returns:
        SearchResult containing the optimal path (in hops), explored nodes, cost, and telemetry.
    """
    t0 = time.perf_counter()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="Breadth First Search (BFS)",
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
            algorithm="Breadth First Search (BFS)",
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
            algorithm="Breadth First Search (BFS)",
            path=[start],
            explored_nodes=[start],
            path_cost=0.0,
            nodes_expanded=1,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            step_history=[{"step": 1, "current": start, "frontier": [], "explored": [start], "action": "Goal found immediately"}],
            success=True,
            message="Start node is the goal node."
        )

    # Queue holds: (current_node, path, cost)
    queue = deque([(start, [start], 0.0)])
    visited = {start}
    explored_order: List[str] = []
    step_history: List[dict] = []
    step_num = 0

    while queue:
        current, path, cost = queue.popleft()
        step_num += 1
        explored_order.append(current)

        step_history.append({
            "step": step_num,
            "current": current,
            "frontier": [item[0] for item in queue],
            "explored": list(explored_order),
            "action": f"Popped '{current}' from queue"
        })

        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Breadth First Search (BFS)",
                path=path,
                explored_nodes=explored_order,
                path_cost=cost,
                nodes_expanded=len(explored_order),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=True,
                message=f"Goal '{goal}' reached successfully."
            )

        # Explore neighbors in alphabetical order
        neighbors = sorted(list(graph.get_neighbors(current).items()), key=lambda x: str(x[0]))
        for neighbor, weight in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor], cost + weight))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return SearchResult(
        algorithm="Breadth First Search (BFS)",
        path=[],
        explored_nodes=explored_order,
        path_cost=0.0,
        nodes_expanded=len(explored_order),
        execution_time_ms=elapsed_ms,
        step_history=step_history,
        success=False,
        message=f"Goal '{goal}' is unreachable from start '{start}'."
    )

