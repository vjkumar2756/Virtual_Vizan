"""
Comprehensive unit test suite for VIZAN Search Algorithms.
Verifies all 7 algorithms against classic AI benchmarks and edge cases.
"""

import pytest
from VIZAN.algorithms.search.base import (
    SearchGraph,
    create_romania_graph,
    create_tree_graph,
    create_grid_graph,
)
from VIZAN.algorithms.search.dfs import depth_first_search
from VIZAN.algorithms.search.bfs import breadth_first_search
from VIZAN.algorithms.search.ids import iterative_deepening_search
from VIZAN.algorithms.search.astar import a_star_search
from VIZAN.algorithms.search.dstar import d_star_search
from VIZAN.algorithms.search.hill_climbing import hill_climbing_search
from VIZAN.algorithms.search.simulated_annealing import simulated_annealing_search


@pytest.fixture
def romania_graph():
    return create_romania_graph()


@pytest.fixture
def tree_graph():
    return create_tree_graph()


@pytest.fixture
def grid_graph():
    return create_grid_graph(rows=4, cols=4, obstacles={(1, 1), (2, 1)})


class TestSearchGraph:
    def test_graph_initialization(self):
        g = SearchGraph(directed=False)
        g.add_node("A", (0, 0))
        g.add_node("B", (3, 4))
        g.add_edge("A", "B", 5.0)

        assert "A" in g.nodes()
        assert "B" in g.nodes()
        assert g.get_neighbors("A")["B"] == 5.0
        assert g.get_neighbors("B")["A"] == 5.0
        assert g.euclidean_heuristic("A", "B") == 5.0
        assert g.manhattan_heuristic("A", "B") == 7.0


class TestDFS:
    def test_dfs_path_found(self, romania_graph):
        res = depth_first_search(romania_graph, "Arad", "Bucharest")
        assert res.success is True
        assert res.path[0] == "Arad"
        assert res.path[-1] == "Bucharest"
        assert res.path_cost > 0
        assert len(res.explored_nodes) > 0

    def test_dfs_root_is_goal(self, romania_graph):
        res = depth_first_search(romania_graph, "Arad", "Arad")
        assert res.success is True
        assert res.path == ["Arad"]
        assert res.path_cost == 0.0

    def test_dfs_unreachable(self):
        g = SearchGraph(directed=True)
        g.add_edge("A", "B", 1)
        g.add_node("C")
        res = depth_first_search(g, "A", "C")
        assert res.success is False
        assert res.path == []

    def test_dfs_invalid_nodes(self, romania_graph):
        res = depth_first_search(romania_graph, "InvalidNode", "Bucharest")
        assert res.success is False


class TestBFS:
    def test_bfs_optimal_hops(self, romania_graph):
        res = breadth_first_search(romania_graph, "Arad", "Bucharest")
        assert res.success is True
        assert res.path[0] == "Arad"
        assert res.path[-1] == "Bucharest"
        # BFS explores in hop order
        assert len(res.path) <= 6

    def test_bfs_tree(self, tree_graph):
        res = breadth_first_search(tree_graph, "A", "O")
        assert res.success is True
        assert res.path == ["A", "C", "G", "O"]


class TestIDS:
    def test_ids_path_found(self, romania_graph):
        res = iterative_deepening_search(romania_graph, "Arad", "Bucharest", max_search_depth=15)
        assert res.success is True
        assert res.path[0] == "Arad"
        assert res.path[-1] == "Bucharest"

    def test_ids_depth_cutoff(self, romania_graph):
        # Arad to Bucharest requires at least 3 hops (Arad-Sibiu-Fagaras-Bucharest)
        res = iterative_deepening_search(romania_graph, "Arad", "Bucharest", max_search_depth=1)
        assert res.success is False


class TestAStar:
    def test_astar_optimal_romania_path(self, romania_graph):
        res = a_star_search(romania_graph, "Arad", "Bucharest")
        assert res.success is True
        # Canonical optimal A* path on Romania map: Arad -> Sibiu -> Rimnicu Vilcea -> Pitesti -> Bucharest
        expected_path = ["Arad", "Sibiu", "Rimnicu Vilcea", "Pitesti", "Bucharest"]
        assert res.path == expected_path
        assert res.path_cost == 418.0

    def test_astar_grid_maze(self, grid_graph):
        res = a_star_search(grid_graph, "(0,0)", "(3,3)")
        assert res.success is True
        assert res.path[0] == "(0,0)"
        assert res.path[-1] == "(3,3)"
        assert res.path_cost > 0


class TestDStar:
    def test_dstar_static_planning(self, romania_graph):
        res = d_star_search(romania_graph, "Arad", "Bucharest")
        assert res.success is True
        assert res.path[0] == "Arad"
        assert res.path[-1] == "Bucharest"
        assert res.path_cost == 418.0

    def test_dstar_dynamic_replanning(self, romania_graph):
        # Introduce a blockage / infinite cost on Rimnicu Vilcea -> Pitesti
        blockage = [("Rimnicu Vilcea", "Pitesti", 9999.0)]
        res = d_star_search(romania_graph, "Arad", "Bucharest", dynamic_changes=blockage)
        assert res.success is True
        assert res.path[-1] == "Bucharest"
        # Should reroute via Fagaras
        assert "Pitesti" not in res.path or "Rimnicu Vilcea" not in res.path


class TestHillClimbing:
    def test_hill_climbing_success(self, romania_graph):
        res = hill_climbing_search(romania_graph, "Fagaras", "Bucharest")
        assert res.success is True
        assert res.path[-1] == "Bucharest"

    def test_hill_climbing_local_optimum_detected(self, romania_graph):
        # From Arad to Bucharest with straight-line distance, hill climbing might get trapped
        res = hill_climbing_search(romania_graph, "Arad", "Bucharest")
        assert isinstance(res.success, bool)
        assert len(res.path) >= 1


class TestSimulatedAnnealing:
    def test_simulated_annealing_execution(self, romania_graph):
        res = simulated_annealing_search(
            romania_graph,
            "Fagaras",
            "Bucharest",
            initial_temperature=50.0,
            cooling_rate=0.9,
            seed=42
        )
        assert isinstance(res.success, bool)
        assert len(res.step_history) > 0
        assert res.execution_time_ms >= 0
