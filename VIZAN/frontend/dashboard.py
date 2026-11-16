"""
Home Dashboard and Main Interface for VIZAN.
Presents the project identity, capabilities, and an intelligent problem-solving chat assistant.
"""

import streamlit as st
from VIZAN.assistant.agent import VizanAgent
from VIZAN.storage.database import get_recent_messages, clear_session_history
from VIZAN.visualization.graph_visualizer import plot_search_graph


def get_agent() -> VizanAgent:
    """Returns a fresh VizanAgent instance."""
    return VizanAgent()


def render_home_dashboard():
    agent = get_agent()
    session_id = st.session_state.get("session_id", "default_user")

    # 1. Project Hero Banner & Exact Project Definition
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #131b2e 0%, #17223b 100%); padding: 24px 28px; border-radius: 14px; border: 1px solid #202c45; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 style="margin: 0; color: #38bdf8; font-size: 2.2rem; font-weight: 800;">🤖 VIZAN</h1>
                    <h3 style="margin: 6px 0 0 0; color: #f1f5f9; font-size: 1.15rem; font-weight: 600;">
                        AI-Based Intelligent Virtual Assistant & Problem Solver
                    </h3>
                </div>
            </div>
            <p style="margin: 12px 0 0 0; color: #cbd5e1; font-size: 1.0rem; line-height: 1.6;">
                VIZAN is a comprehensive <b>Artificial Intelligence Virtual Assistant</b> engineered to solve complex computational problems. 
                Unlike standard chatbots that only generate text, VIZAN is powered by a <b>deterministic Python algorithmic brain</b> that actively plans routes, navigates state spaces, deduces logical knowledge, and reasons under uncertainty.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. What Exactly Does This Project Do? (3 Capability Pillars)
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown(
            """
            <div style="background: #131b2e; padding: 18px; border-radius: 12px; border: 1px solid #202c45; border-left: 5px solid #38bdf8; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 1.1rem;">🔍 1. Search & Navigation</h4>
                <p style="margin: 0; color: #94a3b8; font-size: 0.93rem; line-height: 1.55;">
                    Autonomous pathfinding on road networks, trees, and mazes using <b>A* Search, BFS, DFS, IDS, D* Lite, Hill Climbing,</b> and <b>Simulated Annealing</b> with live visual graph maps.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_p2:
        st.markdown(
            """
            <div style="background: #131b2e; padding: 18px; border-radius: 12px; border: 1px solid #202c45; border-left: 5px solid #10b981; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                <h4 style="margin: 0 0 8px 0; color: #10b981; font-size: 1.1rem;">🧩 2. Classical AI Solvers</h4>
                <p style="margin: 0; color: #94a3b8; font-size: 0.93rem; line-height: 1.55;">
                    State-space puzzle solving & knowledge deduction: <b>8-Puzzle</b> (A* with Manhattan heuristic), <b>Wumpus World</b> (knowledge-based agent), and <b>Towers of Hanoi</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_p3:
        st.markdown(
            """
            <div style="background: #131b2e; padding: 18px; border-radius: 12px; border: 1px solid #202c45; border-left: 5px solid #a855f7; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
                <h4 style="margin: 0 0 8px 0; color: #a855f7; font-size: 1.1rem;">🎲 3. Probabilistic Reasoning</h4>
                <p style="margin: 0; color: #94a3b8; font-size: 0.93rem; line-height: 1.55;">
                    Inference under uncertainty using <b>Bayesian Networks</b> (exact variable elimination), <b>Expectation Maximization</b> (GMM clustering), and <b>Hidden Markov Models</b> (Viterbi decoding).
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Header for the interactive chat
    c_title, c_clear = st.columns([4, 1])
    with c_title:
        st.subheader("💬 Chat & Problem-Solving Studio")
        st.caption("Talk with VIZAN in English or Hinglish. Ask questions, request explanations, or give commands to solve problems.")
    with c_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear current conversation history"):
            clear_session_history(session_id)
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": "🧹 **Chat Cleared**: Hello! I am **VIZAN**. What problem would you like to solve or discuss today?",
                    "extra_data": {}
                }
            ]
            st.rerun()

    # Initialize in-memory messages if empty
    if "chat_messages" not in st.session_state:
        db_msgs = get_recent_messages(session_id, limit=30)
        if db_msgs:
            st.session_state["chat_messages"] = db_msgs
        else:
            st.session_state["chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "👋 **Namaste! Main VIZAN hoon** — aapka Intelligent AI Assistant aur Problem Solver.\n\n"
                        "Aap mujhse:\n"
                        "- Kisi bhi AI topic ya problem ke baare me pooch sakte hain.\n"
                        "- Koi bhi algorithm chalaane ko keh sakte hain (e.g. *'Run A* from Arad to Bucharest'*).\n"
                        "- Coding, math, puzzles ya logic par discuss kar sakte hain.\n\n"
                        "Neeche box me apna sawaal ya message likhiye!"
                    ),
                    "extra_data": {}
                }
            ]

    # Display Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_messages"]:
            role = msg["role"]
            avatar = "🤖" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])

                # If this message produced a graph visualization, render it
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
                        title=f"{res.algorithm} Path Visualization ({start} ➜ {goal})",
                        figsize=(9, 4.5)
                    )
                    st.pyplot(fig)

    # Creative Quick Action Pills (Interactive Floating Chips)
    st.markdown("<p style='font-size:0.86rem; color:#94a3b8; margin: 18px 0 8px 4px;'>💡 <b>Quick Sparks:</b> Click an idea or type in the bar below:</p>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    quick_spark = None

    with p_col1:
        st.markdown('<div class="prompt-pill-btn">', unsafe_allow_html=True)
        if st.button("🚀 Run A* Search", use_container_width=True, key="spark_astar"):
            quick_spark = "Run A* from Arad to Bucharest"
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="prompt-pill-btn">', unsafe_allow_html=True)
        if st.button("📊 Benchmark 7 Algos", use_container_width=True, key="spark_bench"):
            quick_spark = "Compare search algorithms"
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col3:
        st.markdown('<div class="prompt-pill-btn">', unsafe_allow_html=True)
        if st.button("🧩 Solve 8-Puzzle", use_container_width=True, key="spark_puzzle"):
            quick_spark = "Solve 8 puzzle"
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col4:
        st.markdown('<div class="prompt-pill-btn">', unsafe_allow_html=True)
        if st.button("💡 Explain BFS vs DFS", use_container_width=True, key="spark_bfs"):
            quick_spark = "Explain BFS vs DFS in simple terms"
        st.markdown('</div>', unsafe_allow_html=True)

    # Futuristic Neon Chat Input Box
    user_input = st.chat_input("✨ Ask VIZAN anything or give a command (e.g., 'What is A* search?', 'Run BFS', 'Solve a puzzle')...")

    final_query = quick_spark or user_input

    if final_query:
        st.session_state["chat_messages"].append({"role": "user", "content": final_query, "extra_data": {}})
        with st.spinner("VIZAN is thinking and computing..."):
            response_payload = agent.process_message(final_query, session_id=session_id)
        st.session_state["chat_messages"].append(response_payload)
        st.rerun()
