"""
VIZAN Intelligent Agent Coordinator.
Orchestrates:
1. Natural language understanding & structured command conversion via Groq / RuleEngine.
2. 100% deterministic algorithm execution in pure Python (Zero hallucinations).
3. Post-execution conversational explanation of real results via Groq.
4. Answering conceptual AI questions with continuous multi-turn memory.
5. Seamless offline fallback when Groq API is unavailable.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from VIZAN.assistant.rule_engine import RuleEngine, IntentMatch
from VIZAN.assistant.intent_handler import handle_search_intent, handle_compare_intent
from VIZAN.storage.database import save_message, get_recent_messages, clear_session_history

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize Groq client helper
groq_client = None

def get_groq_client():
    global groq_client
    if groq_client is not None:
        return groq_client

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass

    if api_key:
        api_key = str(api_key).strip().strip('"').strip("'")

    if api_key and api_key.startswith("gsk_"):
        try:
            from groq import Groq
            groq_client = Groq(api_key=api_key)
            return groq_client
        except Exception as e:
            print(f"Notice: Groq client init skipped ({e})")
    return None

# Attempt early init
get_groq_client()

GROQ_MODELS = [
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

SYSTEM_PROMPT = """You are VIZAN, an advanced AI Virtual Assistant and Intelligent Problem Solver.
You specialize in Artificial Intelligence algorithms and problem solving:
1. Search Algorithms: BFS, DFS, Iterative Deepening (IDS), A* Search, D* Lite, Hill Climbing, Simulated Annealing.
2. Classical Problems: Wumpus World (Knowledge-Based logic agent), 8-Puzzle (Manhattan Heuristic A*), Towers of Hanoi.
3. Probabilistic Models: Bayesian Networks, Expectation Maximization (GMMs), Hidden Markov Models (Viterbi & Forward).

Guidelines:
- When asked who you are, introduce yourself as VIZAN (AI Virtual Assistant & Problem Solver).
- You are friendly, analytical, and articulate.
- If the user talks to you in Hindi or Hinglish, reply naturally in friendly Hinglish/English. If they speak English, reply in English.
- Always maintain context from ongoing conversation history.
- When explaining algorithms, be structured, intuitive, and mathematically rigorous.
- You do NOT hallucinate algorithm results: All mathematical and pathfinding outputs are computed by the live Python engine.
"""


class VizanAgent:
    """Core Virtual Assistant Agent using Groq conversational intelligence and verified Python execution."""

    def __init__(self):
        self.rule_engine = RuleEngine()

    def process_message(self, user_text: str, session_id: str = "default") -> Dict[str, Any]:
        """
        End-to-End Processing Loop:
        1. Semantic Command Parsing (Groq API / RuleEngine fallback).
        2. Verified Deterministic Algorithm Execution in Python.
        3. Conversational Result Explanation & Memory Persistence.
        """
        user_text = user_text.strip()
        if not user_text:
            return {"role": "assistant", "content": "Please enter a message or command."}

        # Fetch recent conversation history from SQLite for multi-turn context
        recent_history = get_recent_messages(session_id, limit=10)

        # Step 1: Semantic Intent Identification & Command Structuring
        parsed_command = self._parse_intent_with_groq(user_text)

        action = parsed_command.get("action", "GENERAL_CHAT")
        algo_name = parsed_command.get("algorithm")
        extra_data = {}

        # Step 2: Execution Routing
        if action == "RESET":
            clear_session_history(session_id)
            reply = "🧹 **Session Reset**: Conversation history cleared. How can I assist you with AI problem solving or exploration today?"

        elif action == "HELP":
            reply = (
                "### 🤖 VIZAN Capabilities & Commands\n\n"
                "I am your Problem Solving Virtual Assistant. Here is what we can do:\n\n"
                "**1. Search Algorithms (Executed Live in Python):**\n"
                "- `Run A* from Arad to Bucharest` (calculates the optimal 418km path and plots the graph)\n"
                "- `Run BFS`, `Run DFS`, `Run IDS`, `Run D* Lite`\n"
                "- `Run Hill Climbing` or `Run Simulated Annealing`\n"
                "- `Compare search algorithms` (benchmarks all 7 on the Romania map)\n\n"
                "**2. Classical AI Problems:**\n"
                "- `Solve 8 Puzzle` (A* with Manhattan distance heuristic)\n"
                "- `Wumpus World` (Knowledge-based propositional logic agent)\n"
                "- `Towers of Hanoi` (State-space transition and recursive moves)\n\n"
                "**3. Probabilistic AI Models:**\n"
                "- `Bayesian Network` (CPTs and exact inference)\n"
                "- `EM Algorithm` (Gaussian Mixture Models clustering)\n"
                "- `Hidden Markov Model` (Forward and Viterbi decoding)\n\n"
                "**4. Chat & Concept Explanation:**\n"
                "- Ask me any question about algorithms, concepts, math, logic, or programming in English or Hinglish!"
            )

        elif action == "RUN_ALGORITHM" and algo_name in [
            "astar", "bfs", "dfs", "ids", "dstar", "hill_climbing", "simulated_annealing"
        ]:
            # Step 2a: Execute purely in Python
            params = {
                "algorithm": algo_name,
                "start": parsed_command.get("start_node") or "Arad",
                "goal": parsed_command.get("goal_node") or "Bucharest",
                "graph": parsed_command.get("graph_type") or "romania"
            }
            search_res = handle_search_intent(params)
            result = search_res["search_result"]

            extra_data["search_result"] = result
            extra_data["graph"] = search_res["graph"]
            extra_data["start"] = search_res["start"]
            extra_data["goal"] = search_res["goal"]

            # Step 2b: Use Groq to explain the genuine Python output in simple, conversational language
            reply = self._explain_algorithm_result_with_groq(user_text, result, search_res["text"])

        elif action == "COMPARE_ALGORITHMS":
            # Execute all 7 algorithms in Python
            comp_res = handle_compare_intent()
            extra_data["comparison_data"] = comp_res["comparison_data"]
            reply = comp_res["text"]

        elif action == "RUN_ALGORITHM" and algo_name == "eight_puzzle":
            reply = (
                r"### 🧩 8-Puzzle Problem Solver" + "\n\n"
                r"The 8-Puzzle is solved using **A\* Search** with the **Manhattan Distance heuristic**:" + "\n"
                r"$$h(n) = \sum_{i=1}^{8} (|x_i - x_i^{goal}| + |y_i - y_i^{goal}|)$$" + "\n\n"
                "- Solvability is determined by checking the inversion count parity.\n"
                "- Head over to the **🧩 Problem Solving** tab in the sidebar navigation to test custom tile configurations!"
            )

        elif action == "RUN_ALGORITHM" and algo_name == "wumpus":
            reply = (
                "### 🏹 Wumpus World Knowledge-Based Agent\n\n"
                "Wumpus World demonstrates propositional logic and percept reasoning:\n"
                "- **Percepts**: `[Stench, Breeze, Glitter, Bump, Scream]`\n"
                "- **Rules**: $Breeze_{x,y} \\iff \\bigvee_{adj} Pit_{adj}$, $Stench_{x,y} \\iff \\bigvee_{adj} Wumpus_{adj}$\n"
                "- The agent infers provably safe squares before venturing forward.\n"
                "- Navigate to **🧩 Problem Solving** in the sidebar to interact with the grid dungeon!"
            )

        elif action == "RUN_ALGORITHM" and algo_name == "hanoi":
            reply = (
                "### 🗼 Towers of Hanoi\n\n"
                "- **Optimal Moves**: $2^n - 1$ moves for $n$ disks.\n"
                "- **State Space**: Forms a ternary Sierpinski fractal graph.\n"
                "- Navigate to **🧩 Problem Solving** in the sidebar to see disk movements step-by-step!"
            )

        else:
            # Step 2c: General Chat or AI Concept Explanation via Groq with conversation history
            reply = self._generate_conversational_reply(user_text, history=recent_history)

        # Step 3: Persist conversation turns in SQLite
        save_message(session_id, "user", user_text, intent=action)
        save_message(session_id, "assistant", reply, intent=action)

        return {
            "role": "assistant",
            "content": reply,
            "intent": action,
            "extra_data": extra_data
        }

    def _parse_intent_with_groq(self, user_text: str) -> Dict[str, Any]:
        """
        Uses Groq API to convert natural language requests into structured commands.
        Falls back to deterministic RuleEngine if offline or parsing fails.
        """
        # Quick check for rule engine override on clear / reset / help
        rule_match: IntentMatch = self.rule_engine.match(user_text)
        if rule_match.intent in ["RESET", "HELP", "COMPARE_SEARCH"]:
            action_map = {
                "RESET": "RESET",
                "HELP": "HELP",
                "COMPARE_SEARCH": "COMPARE_ALGORITHMS"
            }
            return {"action": action_map[rule_match.intent]}

        client = get_groq_client()
        if client:
            prompt = f"""You are the intent parser for VIZAN AI Assistant.
Analyze the user's message and output a JSON object with these exact keys:
- action: one of ["RUN_ALGORITHM", "COMPARE_ALGORITHMS", "CONCEPT_QUESTION", "GENERAL_CHAT", "RESET", "HELP"]
- algorithm: one of ["astar", "bfs", "dfs", "ids", "dstar", "hill_climbing", "simulated_annealing", "eight_puzzle", "wumpus", "hanoi", "bayesian", "em", "hmm", null]
- start_node: string or null (e.g. "Arad", "A", "(0,0)")
- goal_node: string or null (e.g. "Bucharest", "O", "(4,4)")
- graph_type: one of ["romania", "tree", "grid", null]

Rules:
- If user wants to execute/run/apply an algorithm or find a route, set action="RUN_ALGORITHM".
- If user wants to compare algorithms, set action="COMPARE_ALGORITHMS".
- If user asks to explain a concept or what something is (e.g. "what is BFS", "samjhao"), set action="CONCEPT_QUESTION".
- If it is general greeting or chit-chat, set action="GENERAL_CHAT".

Return ONLY the raw JSON object.
User Message: {json.dumps(user_text)}
"""
            for model_id in GROQ_MODELS:
                try:
                    res = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        response_format={"type": "json_object"}
                    )
                    parsed = json.loads(res.choices[0].message.content)
                    if isinstance(parsed, dict) and "action" in parsed:
                        return parsed
                except Exception:
                    continue

        # Deterministic Offline Fallback via RuleEngine
        if rule_match.intent == "RUN_SEARCH":
            return {
                "action": "RUN_ALGORITHM",
                "algorithm": rule_match.parameters.get("algorithm", "astar"),
                "start_node": rule_match.parameters.get("start", "Arad"),
                "goal_node": rule_match.parameters.get("goal", "Bucharest"),
                "graph_type": "romania"
            }
        elif rule_match.intent == "RUN_8_PUZZLE":
            return {"action": "RUN_ALGORITHM", "algorithm": "eight_puzzle"}
        elif rule_match.intent == "RUN_WUMPUS":
            return {"action": "RUN_ALGORITHM", "algorithm": "wumpus"}
        elif rule_match.intent == "RUN_HANOI":
            return {"action": "RUN_ALGORITHM", "algorithm": "hanoi"}

        return {"action": "GENERAL_CHAT"}

    def _explain_algorithm_result_with_groq(
        self,
        user_query: str,
        search_result: Any,
        fallback_markdown: str
    ) -> str:
        """
        Takes the genuine Python SearchResult and uses Groq to generate a simple,
        conversational explanation of the real numbers and path.
        """
        client = get_groq_client()
        if client:
            prompt = f"""The user asked: {json.dumps(user_query)}

Our Python search engine executed the algorithm and produced this EXACT verified result:
- Algorithm: {search_result.algorithm}
- Success: {search_result.success}
- Path Found: {' -> '.join(map(str, search_result.path)) if search_result.path else 'None'}
- Path Cost: {search_result.path_cost:.2f}
- Nodes Expanded: {search_result.nodes_expanded}
- Explored Set Count: {len(search_result.explored_nodes)} nodes
- Execution Time: {search_result.execution_time_ms:.3f} ms
- Diagnostic Message: {search_result.message}

Instructions:
1. Present the result clearly with bold formatting (Solution Path, Cost, Nodes Expanded, Time).
2. Briefly explain in simple language WHY this path was chosen, why it is optimal (or not, if DFS/Hill Climbing), and what the numbers mean.
3. If the user asked in Hindi/Hinglish, explain in friendly Hinglish. If in English, reply in English.
4. Do NOT change or invent any of the numbers or path steps. Use the exact data above.
"""
            for model_id in GROQ_MODELS:
                try:
                    res = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=600
                    )
                    reply = res.choices[0].message.content
                    if reply and reply.strip():
                        return reply.strip()
                except Exception:
                    continue

        return fallback_markdown

    def _generate_conversational_reply(
        self,
        user_text: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Answers general questions, explains AI concepts, or conducts natural
        conversation using Groq API with multi-turn memory.
        """
        client = get_groq_client()
        if client:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            if history:
                for turn in history:
                    r = turn.get("role")
                    c = turn.get("content")
                    if r in ["user", "assistant"] and c:
                        truncated_content = c if len(c) < 1500 else c[:1500] + "..."
                        messages.append({"role": r, "content": truncated_content})

            messages.append({"role": "user", "content": user_text})

            for model_id in GROQ_MODELS:
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800
                    )
                    reply_text = response.choices[0].message.content
                    if reply_text and reply_text.strip():
                        return reply_text.strip()
                except Exception:
                    continue

        # Offline Fallback
        cleaned = user_text.lower()
        if "bfs" in cleaned and "dfs" in cleaned:
            return (
                "**BFS vs DFS Comparison:**\n\n"
                "1. **Strategy**: BFS uses a FIFO queue (level-by-level), while DFS uses a LIFO stack.\n"
                "2. **Completeness**: BFS is complete on finite branching factors. DFS can get trapped in infinite paths.\n"
                "3. **Optimality**: BFS guarantees the shortest path on uniform costs. DFS is not optimal.\n"
                "4. **Memory**: BFS requires exponential $O(b^d)$ space. DFS requires only linear $O(bm)$ space.\n\n"
                "👉 Try typing: `Compare search algorithms` to see both run side-by-side on the Romania map!"
            )
        elif "astar" in cleaned or "a*" in cleaned:
            return (
                r"**A\* Search Algorithm:**" + "\n\n"
                "A* is an informed best-first search evaluating nodes using:\n"
                r"$$f(n) = g(n) + h(n)$$" + "\n"
                "- $g(n)$: Exact path cost from start to node $n$.\n"
                "- $h(n)$: Estimated heuristic cost from node $n$ to goal.\n\n"
                "**Optimality Guarantee**: A* is guaranteed to return the optimal path if $h(n)$ is **admissible** and **consistent**.\n\n"
                "👉 Try typing: `Run A* from Arad to Bucharest` to run it live!"
            )
        else:
            return (
                f"Hello! I am **VIZAN**, your AI Virtual Assistant and Problem Solver.\n\n"
                f"You asked: *\"{user_text}\"*\n\n"
                f"I can help you analyze and execute Classical Search algorithms (DFS, BFS, IDS, A*, D*, Hill Climbing, Simulated Annealing), "
                f"Classical AI Problems (Wumpus World, 8-Puzzle, Towers of Hanoi), and Probabilistic Models.\n\n"
                f"Try asking:\n"
                f"- `Run A* from Arad to Bucharest`\n"
                f"- `Compare search algorithms`\n"
                f"- `Explain BFS vs DFS`\n"
                f"- `Help` for all commands!"
            )
