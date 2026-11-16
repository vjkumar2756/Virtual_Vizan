"""
Search algorithms module for VIZAN.
Exports uninformed, informed, dynamic, and local search algorithms.
"""

from VIZAN.algorithms.search.base import (
    SearchGraph,
    SearchResult,
    create_romania_graph,
    create_tree_graph,
    create_grid_graph,
)
from VIZAN.algorithms.search.dfs import depth_first_search
from VIZAN.algorithms.search.bfs import breadth_first_search
from VIZAN.algorithms.search.ids import iterative_deepening_search
from VIZAN.algorithms.search.astar import a_star_search
from VIZAN.algorithms.search.dstar import d_star_search, DStarLite
from VIZAN.algorithms.search.hill_climbing import hill_climbing_search
from VIZAN.algorithms.search.simulated_annealing import simulated_annealing_search

__all__ = [
    "SearchGraph",
    "SearchResult",
    "create_romania_graph",
    "create_tree_graph",
    "create_grid_graph",
    "depth_first_search",
    "breadth_first_search",
    "iterative_deepening_search",
    "a_star_search",
    "d_star_search",
    "DStarLite",
    "hill_climbing_search",
    "simulated_annealing_search",
]

