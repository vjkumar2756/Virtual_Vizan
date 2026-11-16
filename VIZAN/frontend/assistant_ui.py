"""
Dedicated Virtual Assistant UI for VIZAN.
Full chat experience with live algorithm graphs and session controls.
"""

import streamlit as st
from VIZAN.assistant.agent import VizanAgent
from VIZAN.storage.database import get_recent_messages, clear_session_history
from VIZAN.visualization.graph_visualizer import plot_search_graph


def render_assistant_page():
    st.title("🤖 Chat with VIZAN")
    st.caption("AI-Powered Virtual Assistant for Classical Search, Problem Solving & Probabilistic AI")

    if "vizan_agent" not in st.session_state:
        st.session_state["vizan_agent"] = VizanAgent()
    agent = st.session_state["vizan_agent"]
    session_id = st.session_state.get("session_id", "default_user")

    if "chat_messages" not in st.session_state:
        db_msgs = get_recent_messages(session_id, limit=30)
        if db_msgs:
            st.session_state["chat_messages"] = db_msgs
        else:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "Hello! I am **VIZAN**, your AI Learning and Problem-Solving Assistant. What would you like to explore or compute?",
                    "extra_data": {}
                }
            ]

    # Quick prompt toolbar
    c1, c2, c3, c4 = st.columns(4)
    quick_text = None
    with c1:
        if st.button("🗺️ A* Arad ➜ Bucharest", key="as_a_star", use_container_width=True):
            quick_text = "Run A* from Arad to Bucharest"
    with c2:
        if st.button("📊 Benchmark Search", key="as_bench", use_container_width=True):
            quick_text = "Compare search algorithms"
    with c3:
        if st.button("💡 Explain Heuristics", key="as_heur", use_container_width=True):
            quick_text = "Explain admissible and consistent heuristics in A*"
    with c4:
        if st.button("🗑️ Reset Chat", key="as_clear", use_container_width=True):
            clear_session_history(session_id)
            st.session_state["chat_messages"] = [
                {"role": "assistant", "content": "Chat history cleared. How can I assist you now?", "extra_data": {}}
            ]
            st.rerun()

    st.markdown("---")

    # Render Messages
    for msg in st.session_state["chat_messages"]:
        role = msg["role"]
        avatar = "🤖" if role == "assistant" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            extra = msg.get("extra_data", {})
            if extra and "search_result" in extra and "graph" in extra:
                res = extra["search_result"]
                graph = extra["graph"]
                start = extra.get("start", "Arad")
                goal = extra.get("goal", "Bucharest")
                fig = plot_search_graph(
                    graph=graph,
                    search_result=res,
                    start_node=start,
                    goal_node=goal,
                    title=f"{res.algorithm} Path ({start} ➜ {goal})",
                    figsize=(9, 4.5)
                )
                st.pyplot(fig)

    user_input = st.chat_input("Type your message or command to VIZAN...")
    query = quick_text or user_input

    if query:
        st.session_state["chat_messages"].append({"role": "user", "content": query, "extra_data": {}})
        with st.spinner("Processing..."):
            res = agent.process_message(query, session_id=session_id)
        st.session_state["chat_messages"].append(res)
        st.rerun()
