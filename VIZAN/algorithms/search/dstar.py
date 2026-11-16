"""
D* (Dynamic A* / D* Lite) Implementation.
Provides incremental heuristic path planning in dynamic environments where
edge costs change or obstacles appear during agent navigation.
Based on the D* Lite algorithm (Koenig & Likhachev).
"""

import heapq
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


class DStarLite:
    """
    D* Lite replanning algorithm.
    Searches backwards from goal to start so that changes in edge costs near the
    agent's current position can be efficiently re-propagated.
    """

    def __init__(
        self,
        graph: SearchGraph,
        start: str,
        goal: str,
        heuristic_fn: Optional[Callable[[str, str], float]] = None
    ):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.heuristic_fn = heuristic_fn or (lambda u, v: graph.euclidean_heuristic(u, v))

        self.k_m: float = 0.0
        self.rhs: Dict[str, float] = {}
        self.g: Dict[str, float] = {}
        self.open_set: List[Tuple[Tuple[float, float], int, str]] = []
        self.in_open: Dict[str, Tuple[float, float]] = {}
        self.tie_counter = 0

        self.nodes_expanded = 0
        self.explored_nodes: List[str] = []
        self.step_history: List[dict] = []

        # Initialize costs
        for node in self.graph.nodes():
            self.rhs[node] = float("inf")
            self.g[node] = float("inf")

        self.rhs[self.goal] = 0.0
        self._push(self.goal, self._calculate_key(self.goal))

    def _calculate_key(self, u: str) -> Tuple[float, float]:
        min_val = min(self.g[u], self.rhs[u])
        h_val = self.heuristic_fn(u, self.start)
        k1 = min_val + h_val + self.k_m
        k2 = min_val
        return (round(k1, 4), round(k2, 4))

    def _push(self, u: str, key: Tuple[float, float]):
        self.tie_counter += 1
        heapq.heappush(self.open_set, (key, self.tie_counter, u))
        self.in_open[u] = key

    def _update_vertex(self, u: str):
        if u != self.goal:
            min_rhs = float("inf")
            for neighbor, weight in self.graph.get_neighbors(u).items():
                cost = weight + self.g.get(neighbor, float("inf"))
                if cost < min_rhs:
                    min_rhs = cost
            self.rhs[u] = min_rhs

        # If currently in open set, remove invalid entry on pop
        if u in self.in_open:
            del self.in_open[u]

        if self.g[u] != self.rhs[u]:
            self._push(u, self._calculate_key(u))

    def compute_shortest_path(self) -> None:
        """Processes the priority queue until start is locally consistent."""
        while self.open_set:
            top_key, _, u = self.open_set[0]

            if u not in self.in_open or self.in_open[u] != top_key:
                heapq.heappop(self.open_set)
                continue

            start_key = self._calculate_key(self.start)
            if top_key >= start_key and self.rhs[self.start] == self.g[self.start]:
                break

            heapq.heappop(self.open_set)
            del self.in_open[u]

            self.nodes_expanded += 1
            if u not in self.explored_nodes:
                self.explored_nodes.append(u)

            k_old = top_key
            k_new = self._calculate_key(u)

            if k_old < k_new:
                self._push(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.step_history.append({
                    "action": f"Overconsistent node '{u}' updated g={self.g[u]:.2f}",
                    "node": u,
                    "g": self.g[u],
                    "rhs": self.rhs[u]
                })
                for pred, _ in self.graph.get_neighbors(u).items():
                    self._update_vertex(pred)
            else:
                self.g[u] = float("inf")
                self._update_vertex(u)
                for pred, _ in self.graph.get_neighbors(u).items():
                    self._update_vertex(pred)

    def extract_path(self) -> Tuple[List[str], float]:
        """Greedily traces the path from start to goal following minimum g-values."""
        if self.rhs[self.start] == float("inf"):
            return [], 0.0

        current = self.start
        path = [current]
        total_cost = 0.0
        max_hops = len(self.graph.nodes()) * 2

        while current != self.goal and len(path) < max_hops:
            neighbors = self.graph.get_neighbors(current)
            if not neighbors:
                break

            best_neighbor = None
            best_cost = float("inf")
            step_edge_weight = 0.0

            for nbr, weight in neighbors.items():
                c = weight + self.g.get(nbr, float("inf"))
                if c < best_cost:
                    best_cost = c
                    best_neighbor = nbr
                    step_edge_weight = weight

            if best_neighbor is None or best_neighbor in path:
                # Cycle or trapped
                break

            total_cost += step_edge_weight
            current = best_neighbor
            path.append(current)

        if path and path[-1] == self.goal:
            return path, total_cost
        return [], 0.0

    def update_edge_cost(self, u: str, v: str, new_weight: float):
        """Notifies D* Lite of an environmental edge cost change and replans."""
        self.k_m += self.heuristic_fn(self.start, self.start)
        # Update edge in graph
        self.graph.add_edge(u, v, new_weight)
        self._update_vertex(u)
        self._update_vertex(v)
        self.compute_shortest_path()


def d_star_search(
    graph: SearchGraph,
    start: str,
    goal: str,
    dynamic_changes: Optional[List[Tuple[str, str, float]]] = None
) -> SearchResult:
    """
    Executes D* Lite search from start to goal, with optional dynamic edge cost changes.

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label
        dynamic_changes: Optional list of (u, v, new_weight) applied during replanning test

    Returns:
        SearchResult containing the path, explored order, cost, and replanning notes.
    """
    t0 = time.perf_counter()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="D* Lite (Dynamic A*)",
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
            algorithm="D* Lite (Dynamic A*)",
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
            algorithm="D* Lite (Dynamic A*)",
            path=[start],
            explored_nodes=[start],
            path_cost=0.0,
            nodes_expanded=1,
            execution_time_ms=(time.perf_counter() - t0) * 1000,
            success=True,
            message="Start node is the goal node."
        )

    solver = DStarLite(graph, start, goal)
    solver.compute_shortest_path()
    initial_path, initial_cost = solver.extract_path()

    notes = [f"Initial planned path: {' -> '.join(initial_path)} (cost: {initial_cost:.2f})"]

    # Apply dynamic edge cost updates (replanning demonstration)
    if dynamic_changes:
        for u, v, new_w in dynamic_changes:
            solver.update_edge_cost(u, v, new_w)
            notes.append(f"Dynamic edge update: ({u}, {v}) -> cost {new_w}")

    final_path, final_cost = solver.extract_path()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    success = len(final_path) > 0 and final_path[-1] == goal
    msg = " | ".join(notes) if dynamic_changes else f"Optimal path found via D* Lite with cost {final_cost:.2f}"

    return SearchResult(
        algorithm="D* Lite (Dynamic A*)",
        path=final_path,
        explored_nodes=solver.explored_nodes,
        path_cost=final_cost,
        nodes_expanded=solver.nodes_expanded,
        execution_time_ms=elapsed_ms,
        step_history=solver.step_history,
        success=success,
        message=msg
    )

