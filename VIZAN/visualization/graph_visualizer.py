"""
Graph visualization utility for search algorithms using NetworkX and Matplotlib.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from VIZAN.algorithms.search.base import SearchGraph, SearchResult


def plot_search_graph(
    graph: SearchGraph,
    search_result: Optional[SearchResult] = None,
    start_node: Optional[str] = None,
    goal_node: Optional[str] = None,
    title: str = "Search Graph Visualization",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Renders a SearchGraph with NetworkX and Matplotlib.

    Visual encoding:
    - Path edges: bold emerald green
    - Explored nodes: warm amber
    - Path nodes: vibrant lime green
    - Start node: sky blue
    - Goal node: golden yellow
    - Normal unvisited: soft gray/steel blue
    """
    G = nx.DiGraph() if graph.directed else nx.Graph()

    for node in graph.nodes():
        G.add_node(node)

    for u in graph.nodes():
        for v, weight in graph.get_neighbors(u).items():
            if graph.directed or u <= v:
                G.add_edge(u, v, weight=weight)

    # Use graph coordinates if present, else spring layout
    pos = {}
    for node in graph.nodes():
        if node in graph.positions:
            pos[node] = graph.positions[node]

    if len(pos) < len(graph.nodes()):
        # Compute layout fallback for missing positions
        fallback_pos = nx.spring_layout(G, seed=42)
        for node in graph.nodes():
            if node not in pos:
                pos[node] = fallback_pos[node]

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # Extract path and explored nodes from result if available
    path_nodes: List[str] = []
    path_edges: Set[Tuple[str, str]] = set()
    explored_nodes: Set[str] = set()

    if search_result:
        path_nodes = [str(n) for n in search_result.path]
        explored_nodes = set(str(n) for n in search_result.explored_nodes)
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            path_edges.add((u, v))
            path_edges.add((v, u))

    # Node color mapping
    node_colors = []
    node_border_colors = []
    node_sizes = []

    for node in G.nodes():
        if node == start_node:
            node_colors.append("#00d2ff")
            node_border_colors.append("#ffffff")
            node_sizes.append(1100)
        elif node == goal_node:
            node_colors.append("#ffd700")
            node_border_colors.append("#ffffff")
            node_sizes.append(1100)
        elif node in path_nodes:
            node_colors.append("#00e676")
            node_border_colors.append("#b9f6ca")
            node_sizes.append(900)
        elif node in explored_nodes:
            node_colors.append("#ff9100")
            node_border_colors.append("#ffe082")
            node_sizes.append(750)
        else:
            node_colors.append("#2a2f3f")
            node_border_colors.append("#616161")
            node_sizes.append(650)

    # Base edges
    base_edges = [e for e in G.edges() if (e[0], e[1]) not in path_edges and (e[1], e[0]) not in path_edges]
    highlighted_edges = [e for e in G.edges() if (e[0], e[1]) in path_edges or (e[1], e[0]) in path_edges]

    edge_kwargs = {"ax": ax, "alpha": 0.6, "width": 1.5, "edge_color": "#4b5563"}
    if graph.directed:
        edge_kwargs["arrows"] = True
        edge_kwargs["arrowsize"] = 14

    nx.draw_networkx_edges(
        G, pos,
        edgelist=base_edges,
        **edge_kwargs
    )

    if highlighted_edges:
        hl_kwargs = {"ax": ax, "alpha": 0.95, "width": 3.5, "edge_color": "#00e676"}
        if graph.directed:
            hl_kwargs["arrows"] = True
            hl_kwargs["arrowsize"] = 18
        nx.draw_networkx_edges(
            G, pos,
            edgelist=highlighted_edges,
            **hl_kwargs
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        edgecolors=node_border_colors,
        linewidths=2.0,
        node_size=node_sizes,
        ax=ax
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=8,
        font_color="#ffffff",
        font_family="sans-serif",
        font_weight="bold",
        ax=ax
    )

    # Edge weight labels
    edge_labels = {(u, v): f"{w:.0f}" if float(w).is_integer() else f"{w:.1f}" for u, v, w in G.edges(data="weight") if w is not None}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="#b0bec5",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#1e222d", edgecolor="none", alpha=0.7),
        ax=ax
    )

    ax.set_title(title, color="#eceff1", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig

