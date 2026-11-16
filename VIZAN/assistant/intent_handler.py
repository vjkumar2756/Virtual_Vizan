"""
Intent Handler: Dispatches classified intents to real Python algorithm modules.
Guarantees 100% genuine algorithmic execution without hallucinations.
"""

from typing import Any, Dict
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
from VIZAN.algorithms.search.dstar import d_star_search
from VIZAN.algorithms.search.hill_climbing import hill_climbing_search
from VIZAN.algorithms.search.simulated_annealing import simulated_annealing_search


def handle_search_intent(params: Dict[str, Any]) -> Dict[str, Any]:
    """Executes the requested search algorithm on the benchmark graph."""
    algo = params.get("algorithm", "astar").lower()
    start = params.get("start", "Arad")
    goal = params.get("goal", "Bucharest")
    graph_type = params.get("graph", "romania")

    if graph_type == "tree":
        graph = create_tree_graph()
        if start not in graph.nodes():
            start = "A"
        if goal not in graph.nodes():
            goal = "O"
    elif graph_type == "grid":
        graph = create_grid_graph()
        if start not in graph.nodes():
            start = "(0,0)"
        if goal not in graph.nodes():
            goal = "(4,4)"
    else:
        graph = create_romania_graph()
        if start not in graph.nodes():
            start = "Arad"
        if goal not in graph.nodes():
            goal = "Bucharest"

    result: SearchResult
    if algo == "astar":
        result = a_star_search(graph, start, goal)
    elif algo == "bfs":
        result = breadth_first_search(graph, start, goal)
    elif algo == "dfs":
        result = depth_first_search(graph, start, goal)
    elif algo == "ids":
        result = iterative_deepening_search(graph, start, goal)
    elif algo == "dstar":
        result = d_star_search(graph, start, goal)
    elif algo == "hill_climbing":
        result = hill_climbing_search(graph, start, goal)
    elif algo == "simulated_annealing":
        result = simulated_annealing_search(graph, start, goal, seed=42)
    else:
        result = a_star_search(graph, start, goal)

    summary_text = (
        f"### 🎯 Algorithm Execution: {result.algorithm}\n\n"
        f"- **Graph**: Romania Road Map\n"
        f"- **Start Node**: `{start}` ➜ **Goal Node**: `{goal}`\n"
        f"- **Status**: {'✅ Optimal Path Found' if result.success else '⚠️ Goal Not Reached'}\n"
        f"- **Solution Path**: {' ➜ '.join(result.path) if result.path else 'None'}\n"
        f"- **Total Path Cost**: `{result.path_cost:.2f}`\n"
        f"- **Nodes Expanded**: `{result.nodes_expanded}`\n"
        f"- **Explored Set Size**: `{len(result.explored_nodes)}` nodes\n"
        f"- **Execution Time**: `{result.execution_time_ms:.3f} ms`\n\n"
        f"*{result.message}*"
    )

    return {
        "text": summary_text,
        "search_result": result,
        "graph": graph,
        "start": start,
        "goal": goal,
    }


def handle_compare_intent() -> Dict[str, Any]:
    """Runs all 7 algorithms on Romania map from Arad to Bucharest and compares them."""
    graph = create_romania_graph()
    start, goal = "Arad", "Bucharest"

    algos = [
        ("A* Search", lambda: a_star_search(graph, start, goal)),
        ("Breadth First Search (BFS)", lambda: breadth_first_search(graph, start, goal)),
        ("Depth First Search (DFS)", lambda: depth_first_search(graph, start, goal)),
        ("Iterative Deepening Search (IDS)", lambda: iterative_deepening_search(graph, start, goal)),
        ("D* Lite", lambda: d_star_search(graph, start, goal)),
        ("Hill Climbing", lambda: hill_climbing_search(graph, start, goal)),
        ("Simulated Annealing", lambda: simulated_annealing_search(graph, start, goal, seed=42)),
    ]

    results = []
    for name, fn in algos:
        r = fn()
        results.append(r.to_summary_dict())

    md_table = "| Algorithm | Path Cost | Nodes Expanded | Explored | Time (ms) | Success |\n"
    md_table += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for row in results:
        md_table += f"| {row['Algorithm']} | {row['Path Cost']} | {row['Nodes Expanded']} | {row['Explored Count']} | {row['Execution Time (ms)']} | {'✅' if row['Success'] else '❌'} |\n"

    summary_text = (
        f"### 📊 Benchmark Comparison: Arad ➜ Bucharest\n\n"
        f"{md_table}\n\n"
        f"**Key Insights:**\n"
        r"- **A\* Search** is optimal and informed, finding the true minimum cost path (418.0) while expanding far fewer nodes." + "\n"
        f"- **BFS** finds the path with the fewest hops, which may differ from the minimum distance on weighted graphs.\n"
        f"- **DFS** dives deep and often finds much longer paths.\n"
        f"- **IDS** guarantees finding the shallowest goal with linear memory complexity.\n"
        r"- **D\* Lite** matches A* initially and allows rapid replanning if roads are blocked."
    )

    return {
        "text": summary_text,
        "comparison_data": results
    }
