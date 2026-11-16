"""
Simulated Annealing Algorithm Implementation.
Probabilistic local search metaheuristic that avoids local optima by accepting
worse transitions with a probability governed by the current temperature schedule:
P = exp(-Delta E / T).
"""

import math
import random
import time
from typing import Any, Callable, Dict, List, Optional
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def simulated_annealing_search(
    graph: SearchGraph,
    start: str,
    goal: str,
    initial_temperature: float = 100.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
    max_iterations: int = 1000,
    seed: Optional[int] = 42,
    heuristic_fn: Optional[Callable[[str, str], float]] = None
) -> SearchResult:
    """
    Executes Simulated Annealing from start to goal.

    Energy E is defined as the heuristic distance to goal: E(node) = h(node, goal).
    Moves with Delta E <= 0 are always accepted.
    Moves with Delta E > 0 are accepted with probability exp(-Delta E / T).

    Parameters:
        graph: SearchGraph instance
        start: Starting node label
        goal: Target node label
        initial_temperature: Initial temperature T0
        cooling_rate: Multiplicative factor alpha in [0.8, 0.99]
        min_temperature: Convergence temperature cutoff
        max_iterations: Safety iteration threshold
        seed: Random seed for deterministic reproducibility
        heuristic_fn: Distance estimation function

    Returns:
        SearchResult containing the annealed path, temperature history, and telemetry.
    """
    t0 = time.perf_counter()

    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    if start not in graph.nodes():
        return SearchResult(
            algorithm="Simulated Annealing",
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
            algorithm="Simulated Annealing",
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
    temperature = initial_temperature
    step_history: List[dict] = []
    iteration = 0

    while iteration < max_iterations and temperature > min_temperature:
        iteration += 1
        current_energy = heuristic_fn(current, goal)

        if current == goal:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return SearchResult(
                algorithm="Simulated Annealing",
                path=path,
                explored_nodes=explored_nodes,
                path_cost=path_cost,
                nodes_expanded=len(explored_nodes),
                execution_time_ms=elapsed_ms,
                step_history=step_history,
                success=True,
                message=f"Goal '{goal}' reached at T={temperature:.2f} in {iteration} steps."
            )

        neighbors = list(graph.get_neighbors(current).items())
        if not neighbors:
            break

        # Randomly choose a successor
        next_node, edge_weight = rng.choice(neighbors)
        if next_node not in explored_nodes:
            explored_nodes.append(next_node)

        next_energy = heuristic_fn(next_node, goal)
        delta_e = next_energy - current_energy

        accepted = False
        prob = 1.0

        if delta_e <= 0:
            accepted = True
        else:
            # Boltzmann acceptance probability
            prob = math.exp(-delta_e / max(temperature, 1e-6))
            if rng.random() < prob:
                accepted = True

        step_history.append({
            "step": iteration,
            "current": current,
            "next": next_node,
            "T": round(temperature, 3),
            "delta_E": round(delta_e, 3),
            "prob": round(prob, 4),
            "accepted": accepted,
            "action": f"T={temperature:.2f}: Candidate '{next_node}' (dE={delta_e:.2f}, P={prob:.3f}) -> {'ACCEPTED' if accepted else 'REJECTED'}"
        })

        if accepted:
            current = next_node
            path.append(current)
            path_cost += edge_weight

        # Anneal temperature
        temperature *= cooling_rate

    elapsed_ms = (time.perf_counter() - t0) * 1000
    success = (len(path) > 0 and path[-1] == goal)
    return SearchResult(
        algorithm="Simulated Annealing",
        path=path,
        explored_nodes=explored_nodes,
        path_cost=path_cost,
        nodes_expanded=len(explored_nodes),
        execution_time_ms=elapsed_ms,
        step_history=step_history,
        success=success,
        message=f"Halted at T={temperature:.3f}. Goal {'reached' if success else 'not reached'}."
    )

