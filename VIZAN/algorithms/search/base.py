"""
Base classes, data models, and graph structures for search algorithms in VIZAN.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math
import time


@dataclass
class SearchResult:
    """Standardized result returned by all search algorithms."""
    algorithm: str
    path: List[Any]
    explored_nodes: List[Any]
    path_cost: float
    nodes_expanded: int
    execution_time_ms: float
    step_history: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    message: str = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "Algorithm": self.algorithm,
            "Success": self.success,
            "Path Length": len(self.path),
            "Path": " -> ".join(map(str, self.path)) if self.path else "None",
            "Path Cost": round(self.path_cost, 2),
            "Nodes Expanded": self.nodes_expanded,
            "Explored Count": len(self.explored_nodes),
            "Execution Time (ms)": round(self.execution_time_ms, 3),
        }


class SearchGraph:
    """
    Weighted graph representation supporting both directed/undirected edges
    and geometric 2D coordinates for heuristic estimation.
    """

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: Dict[str, Dict[str, float]] = {}
        self.positions: Dict[str, Tuple[float, float]] = {}

    def add_node(self, node: str, pos: Optional[Tuple[float, float]] = None):
        """Add a node with optional (x, y) coordinates for heuristic calculations."""
        node = str(node).strip()
        if node not in self.adj:
            self.adj[node] = {}
        if pos is not None:
            self.positions[node] = (float(pos[0]), float(pos[1]))

    def add_edge(self, u: str, v: str, weight: float = 1.0, bidirectional: Optional[bool] = None):
        """Add a weighted edge between u and v."""
        u, v = str(u).strip(), str(v).strip()
        weight = float(weight)
        self.add_node(u)
        self.add_node(v)

        self.adj[u][v] = weight
        is_bi = (not self.directed) if bidirectional is None else bidirectional
        if is_bi:
            self.adj[v][u] = weight

    def get_neighbors(self, node: str) -> Dict[str, float]:
        """Return dict of {neighbor: weight}."""
        return self.adj.get(str(node), {})

    def nodes(self) -> List[str]:
        """Return list of all registered nodes."""
        return sorted(list(self.adj.keys()))

    def euclidean_heuristic(self, u: str, v: str) -> float:
        """Straight-line Euclidean distance heuristic between two positioned nodes."""
        if u in self.positions and v in self.positions:
            (x1, y1), (x2, y2) = self.positions[u], self.positions[v]
            return math.hypot(x2 - x1, y2 - y1)
        return 0.0

    def manhattan_heuristic(self, u: str, v: str) -> float:
        """Manhattan distance heuristic between two positioned nodes."""
        if u in self.positions and v in self.positions:
            (x1, y1), (x2, y2) = self.positions[u], self.positions[v]
            return abs(x2 - x1) + abs(y2 - y1)
        return 0.0


def create_romania_graph() -> SearchGraph:
    """
    Classic AI benchmark: Subgraph of Romania map with approximate (x, y) coordinates
    and highway distances.
    """
    g = SearchGraph(directed=False)
    # Positions (approximate geo coordinates scaled for 2D visualization)
    coords = {
        "Arad": (91, 492),
        "Zerind": (108, 531),
        "Oradea": (131, 571),
        "Sibiu": (207, 457),
        "Timisoara": (94, 411),
        "Lugoj": (165, 379),
        "Mehadia": (168, 339),
        "Drobeta": (165, 299),
        "Craiova": (253, 288),
        "Rimnicu Vilcea": (233, 410),
        "Fagaras": (305, 449),
        "Pitesti": (320, 368),
        "Bucharest": (400, 327),
        "Giurgiu": (375, 270),
        "Urziceni": (456, 350),
        "Neamt": (406, 537),
        "Iasi": (473, 506),
        "Vaslui": (509, 444),
        "Hirsova": (534, 350),
        "Eforie": (562, 293),
    }
    for node, pos in coords.items():
        g.add_node(node, pos)

    edges = [
        ("Arad", "Zerind", 75),
        ("Arad", "Sibiu", 140),
        ("Arad", "Timisoara", 118),
        ("Zerind", "Oradea", 71),
        ("Oradea", "Sibiu", 151),
        ("Timisoara", "Lugoj", 111),
        ("Lugoj", "Mehadia", 70),
        ("Mehadia", "Drobeta", 75),
        ("Drobeta", "Craiova", 120),
        ("Sibiu", "Fagaras", 99),
        ("Sibiu", "Rimnicu Vilcea", 80),
        ("Rimnicu Vilcea", "Craiova", 146),
        ("Rimnicu Vilcea", "Pitesti", 97),
        ("Craiova", "Pitesti", 138),
        ("Fagaras", "Bucharest", 211),
        ("Pitesti", "Bucharest", 101),
        ("Bucharest", "Giurgiu", 90),
        ("Bucharest", "Urziceni", 85),
        ("Urziceni", "Hirsova", 98),
        ("Hirsova", "Eforie", 86),
        ("Urziceni", "Vaslui", 142),
        ("Vaslui", "Iasi", 92),
        ("Iasi", "Neamt", 87),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def create_tree_graph() -> SearchGraph:
    """
    Hierarchical binary/ternary tree benchmark showing structural differences
    between DFS, BFS, and IDS.
    """
    g = SearchGraph(directed=False)
    nodes = {
        "A": (0, 3),
        "B": (-2, 2), "C": (2, 2),
        "D": (-3, 1), "E": (-1, 1), "F": (1, 1), "G": (3, 1),
        "H": (-3.5, 0), "I": (-2.5, 0), "J": (-1.5, 0), "K": (-0.5, 0),
        "L": (0.5, 0), "M": (1.5, 0), "N": (2.5, 0), "O": (3.5, 0),
    }
    for n, p in nodes.items():
        g.add_node(n, p)

    edges = [
        ("A", "B", 1), ("A", "C", 1),
        ("B", "D", 2), ("B", "E", 2),
        ("C", "F", 2), ("C", "G", 2),
        ("D", "H", 3), ("D", "I", 3),
        ("E", "J", 3), ("E", "K", 3),
        ("F", "L", 3), ("F", "M", 3),
        ("G", "N", 3), ("G", "O", 3),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    return g


def create_grid_graph(rows: int = 5, cols: int = 5, obstacles: Optional[Set[Tuple[int, int]]] = None) -> SearchGraph:
    """
    Grid maze graph connecting orthogonal grid cells, useful for pathfinding and D* dynamic obstacles.
    """
    if obstacles is None:
        obstacles = {(1, 2), (2, 2), (3, 2)}  # Simple vertical wall
    g = SearchGraph(directed=False)

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in obstacles:
                name = f"({r},{c})"
                g.add_node(name, (float(c), float(-r)))

    for r in range(rows):
        for c in range(cols):
            if (r, c) in obstacles:
                continue
            u = f"({r},{c})"
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obstacles:
                    v = f"({nr},{nc})"
                    g.add_edge(u, v, 1.0)
    return g

