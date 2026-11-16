"""
Streamlit UI for Search Algorithms Module in VIZAN.
Supports full visual problem preview, custom graph editing, and step-by-step execution.
"""

import time
import pandas as pd
import streamlit as st
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
from VIZAN.visualization.graph_visualizer import plot_search_graph


def get_default_graph(choice: str) -> SearchGraph:
    """Returns a fresh instance of the selected benchmark graph."""
    if choice.startswith("Romania"):
        return create_romania_graph()
    elif choice.startswith("Hierarchical"):
        return create_tree_graph()
    elif choice.startswith("5x5"):
        return create_grid_graph(rows=5, cols=5, obstacles={(1, 2), (2, 2), (3, 2)})
    else:
        g = SearchGraph(directed=False)
        g.add_node("A", (0, 0))
        g.add_node("B", (2, 2))
        g.add_node("C", (2, -2))
        g.add_node("D", (5, 1))
        g.add_node("E", (6, -1))
        g.add_edge("A", "B", 4.0)
        g.add_edge("A", "C", 2.0)
        g.add_edge("B", "D", 5.0)
        g.add_edge("C", "D", 1.0)
        g.add_edge("C", "E", 8.0)
        g.add_edge("D", "E", 3.0)
        return g


def render_search_algorithms_page():
    st.title("🔍 Search Algorithms Studio")
    st.markdown(
        """
        Explore search algorithms on real graph problems.
        You can visually see the problem graph below, edit edges or weights, and run algorithms to find optimal paths.
        """
    )

    # 1. Benchmark Problem Selection
    col_bench, col_reset = st.columns([3, 1])

    with col_bench:
        benchmark_choice = st.selectbox(
            "Select Problem Graph",
            options=[
                "Romania Road Map (Classic A* Benchmark)",
                "Hierarchical Search Tree",
                "5x5 Grid Maze with Obstacles",
                "Custom Graph (User Defined)",
            ],
            index=0,
            key="search_bench_choice"
        )

    # Manage active graph in session state
    state_key = f"active_graph_{benchmark_choice}"
    if state_key not in st.session_state:
        st.session_state[state_key] = get_default_graph(benchmark_choice)

    with col_reset:
        st.write("") # spacing
        st.write("")
        if st.button("🔄 Reset Graph", use_container_width=True, help="Reset to original default benchmark"):
            st.session_state[state_key] = get_default_graph(benchmark_choice)
            st.session_state.pop("last_search_result", None)
            st.rerun()

    graph: SearchGraph = st.session_state[state_key]
    available_nodes = graph.nodes()

    if not available_nodes:
        st.error("The graph contains no nodes. Please reset or add nodes below.")
        return

    # Select sensible default start and goal
    if benchmark_choice.startswith("Romania"):
        def_start = "Arad" if "Arad" in available_nodes else available_nodes[0]
        def_goal = "Bucharest" if "Bucharest" in available_nodes else available_nodes[-1]
    elif benchmark_choice.startswith("Hierarchical"):
        def_start = "A" if "A" in available_nodes else available_nodes[0]
        def_goal = "O" if "O" in available_nodes else available_nodes[-1]
    elif benchmark_choice.startswith("5x5"):
        def_start = "(0,0)" if "(0,0)" in available_nodes else available_nodes[0]
        def_goal = "(4,4)" if "(4,4)" in available_nodes else available_nodes[-1]
    else:
        def_start = available_nodes[0]
        def_goal = available_nodes[-1]

    # Start and Goal Selectors
    col_s, col_g = st.columns(2)
    with col_s:
        start_node = st.selectbox(
            "📍 Start Node (Origin)",
            options=available_nodes,
            index=available_nodes.index(def_start) if def_start in available_nodes else 0,
            key=f"start_node_sel_{benchmark_choice}"
        )
    with col_g:
        goal_node = st.selectbox(
            "🎯 Goal Node (Destination)",
            options=available_nodes,
            index=available_nodes.index(def_goal) if def_goal in available_nodes else len(available_nodes) - 1,
            key=f"goal_node_sel_{benchmark_choice}"
        )

    # 2. Problem Statement Banner (Always Visible)
    st.markdown(
        f"""
        <div style="background: #1e222d; padding: 14px 18px; border-radius: 8px; border-left: 5px solid #00d2ff; margin: 12px 0 16px 0;">
            <h4 style="margin: 0; color: #60a5fa;">🎯 Current Problem Statement:</h4>
            <p style="margin: 5px 0 0 0; color: #e2e8f0; font-size: 1.05rem;">
                Find the path with minimal cost from <b>{start_node}</b> (Origin) to <b>{goal_node}</b> (Destination)
                across <b>{len(available_nodes)} nodes</b> and <b>{sum(len(v) for v in graph.adj.values()) // 2} connecting edges</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Graph Editor / Customizer Expander
    with st.expander("✏️ Customize Problem Graph (Add Roads, Change Weights, or Add Cities)"):
        st.markdown("**Modify or add edges in the current problem:**")
        c_e1, c_e2, c_e3, c_e4 = st.columns([2, 2, 2, 2])
        with c_e1:
            edit_u = st.selectbox("From Node", options=available_nodes, key="edit_u")
        with c_e2:
            edit_v = st.selectbox("To Node", options=available_nodes, index=min(1, len(available_nodes)-1), key="edit_v")
        with c_e3:
            edit_w = st.number_input("Edge Weight (Distance)", min_value=1.0, max_value=9999.0, value=50.0, step=5.0)
        with c_e4:
            st.write("")
            st.write("")
            if st.button("➕ Add / Update Edge", use_container_width=True):
                graph.add_edge(edit_u, edit_v, edit_w)
                st.session_state.pop("last_search_result", None)
                st.success(f"Edge ({edit_u} ⟷ {edit_v}) updated with weight {edit_w}!")
                st.rerun()

        col_del, col_add_node = st.columns(2)
        with col_del:
            st.markdown("**Remove an existing edge (simulate blockage):**")
            all_edges = []
            for u in graph.nodes():
                for v, w in graph.get_neighbors(u).items():
                    if u <= v:
                        all_edges.append(f"{u} <-> {v} (weight: {w})")
            if all_edges:
                edge_to_del = st.selectbox("Select Road to Remove", options=all_edges, key="del_edge_sel")
                if st.button("❌ Remove Road", use_container_width=True):
                    parts = edge_to_del.split("<->")
                    u_del = parts[0].strip()
                    v_del = parts[1].split("(")[0].strip()
                    if v_del in graph.adj.get(u_del, {}):
                        del graph.adj[u_del][v_del]
                    if u_del in graph.adj.get(v_del, {}):
                        del graph.adj[v_del][u_del]
                    st.session_state.pop("last_search_result", None)
                    st.success(f"Removed road between {u_del} and {v_del}")
                    st.rerun()

        with col_add_node:
            st.markdown("**Add a brand new node (City):**")
            new_node_name = st.text_input("New Node Name", placeholder="e.g. City_X")
            c_x, c_y = st.columns(2)
            with c_x:
                new_x = st.number_input("Coord X", value=250.0)
            with c_y:
                new_y = st.number_input("Coord Y", value=300.0)
            if st.button("📍 Add New Node", use_container_width=True):
                if new_node_name.strip():
                    graph.add_node(new_node_name.strip(), (new_x, new_y))
                    st.success(f"Node '{new_node_name.strip()}' added!")
                    st.rerun()

    # 4. Problem Graph Visualization (BEFORE or AFTER running)
    last_res = st.session_state.get("last_search_result", None)
    # Check if last_res matches current problem
    if last_res and (st.session_state.get("res_start") != start_node or st.session_state.get("res_goal") != goal_node):
        last_res = None

    fig_title = f"{last_res.algorithm} Solution Path ({start_node} ➜ {goal_node})" if last_res else f"Current Problem Graph (Start: {start_node} | Goal: {goal_node})"
    fig = plot_search_graph(
        graph=graph,
        search_result=last_res,
        start_node=start_node,
        goal_node=goal_node,
        title=fig_title,
        figsize=(10, 5.5)
    )
    st.pyplot(fig)

    st.divider()

    # 5. Algorithm Execution Controls
    tab_single, tab_compare = st.tabs(["🚀 Single Algorithm Runner", "📊 All-Algorithm Comparison"])

    with tab_single:
        col_algo, col_params = st.columns([1, 1])

        with col_algo:
            algo_choice = st.selectbox(
                "Select Search Algorithm to Apply",
                options=[
                    "A* Search",
                    "Breadth First Search (BFS)",
                    "Depth First Search (DFS)",
                    "Iterative Deepening Search (IDS)",
                    "D* Lite (Dynamic A*)",
                    "Hill Climbing",
                    "Simulated Annealing"
                ],
                index=0
            )

        with col_params:
            if algo_choice == "Iterative Deepening Search (IDS)":
                ids_max_depth = st.slider("Max Search Depth Limit", min_value=1, max_value=50, value=20)
            elif algo_choice == "Simulated Annealing":
                col_sa1, col_sa2 = st.columns(2)
                with col_sa1:
                    sa_temp = st.number_input("Initial Temp (T0)", min_value=1.0, max_value=1000.0, value=100.0)
                with col_sa2:
                    sa_alpha = st.slider("Cooling Rate (alpha)", min_value=0.5, max_value=0.99, value=0.95, step=0.01)
            elif algo_choice == "D* Lite (Dynamic A*)":
                st.caption("D* Lite computes optimal path and supports dynamic replanning.")

        run_btn = st.button("🚀 Run Algorithm on This Graph", type="primary", use_container_width=True)

        if run_btn:
            result: SearchResult = None
            if algo_choice == "A* Search":
                result = a_star_search(graph, start_node, goal_node)
            elif algo_choice == "Breadth First Search (BFS)":
                result = breadth_first_search(graph, start_node, goal_node)
            elif algo_choice == "Depth First Search (DFS)":
                result = depth_first_search(graph, start_node, goal_node)
            elif algo_choice == "Iterative Deepening Search (IDS)":
                result = iterative_deepening_search(graph, start_node, goal_node, max_search_depth=ids_max_depth)
            elif algo_choice == "D* Lite (Dynamic A*)":
                result = d_star_search(graph, start_node, goal_node)
            elif algo_choice == "Hill Climbing":
                result = hill_climbing_search(graph, start_node, goal_node)
            elif algo_choice == "Simulated Annealing":
                result = simulated_annealing_search(
                    graph, start_node, goal_node,
                    initial_temperature=sa_temp,
                    cooling_rate=sa_alpha,
                    seed=42
                )

            st.session_state["last_search_result"] = result
            st.session_state["res_start"] = start_node
            st.session_state["res_goal"] = goal_node
            st.rerun()

        if last_res:
            res: SearchResult = last_res
            if res.success:
                st.success(f"✅ **{res.algorithm}**: {res.message}")
            else:
                st.warning(f"⚠️ **{res.algorithm}**: {res.message}")

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Path Length", f"{len(res.path)} nodes")
            m2.metric("Path Cost", f"{res.path_cost:.2f}")
            m3.metric("Nodes Expanded", res.nodes_expanded)
            m4.metric("Explored Set", len(res.explored_nodes))
            m5.metric("Runtime (ms)", f"{res.execution_time_ms:.3f}")

            if res.path:
                st.info(f"📍 **Optimal Path**: {' ➜ '.join(res.path)}")
            else:
                st.error("No valid path found to the goal node.")

            if res.step_history:
                with st.expander(f"📋 Step-by-Step Search History ({len(res.step_history)} steps)"):
                    st.dataframe(pd.DataFrame(res.step_history), use_container_width=True)

    with tab_compare:
        st.subheader("Benchmark All 7 Algorithms on This Problem")
        compare_btn = st.button("Run All 7 Algorithms & Compare", type="primary", use_container_width=True)

        if compare_btn:
            algorithms_to_run = [
                ("Breadth First Search (BFS)", lambda: breadth_first_search(graph, start_node, goal_node)),
                ("Depth First Search (DFS)", lambda: depth_first_search(graph, start_node, goal_node)),
                ("Iterative Deepening Search (IDS)", lambda: iterative_deepening_search(graph, start_node, goal_node)),
                ("A* Search", lambda: a_star_search(graph, start_node, goal_node)),
                ("D* Lite", lambda: d_star_search(graph, start_node, goal_node)),
                ("Hill Climbing", lambda: hill_climbing_search(graph, start_node, goal_node)),
                ("Simulated Annealing", lambda: simulated_annealing_search(graph, start_node, goal_node, seed=42)),
            ]

            results_list = []
            with st.spinner("Benchmarking all algorithms..."):
                for name, fn in algorithms_to_run:
                    r = fn()
                    results_list.append(r.to_summary_dict())

            df_compare = pd.DataFrame(results_list)
            st.dataframe(df_compare, use_container_width=True)

            c_chart1, c_chart2 = st.columns(2)
            with c_chart1:
                st.markdown("##### Nodes Expanded (Efficiency)")
                st.bar_chart(df_compare.set_index("Algorithm")["Nodes Expanded"])
            with c_chart2:
                st.markdown("##### Path Cost (Optimality)")
                st.bar_chart(df_compare.set_index("Algorithm")["Path Cost"])
