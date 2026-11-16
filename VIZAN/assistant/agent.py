"""
VIZAN Intelligent Agent Coordinator.
Orchestrates rule detection, deterministic algorithm execution, SQLite memory,
and intelligent conversational problem solving via Groq API (openai/gpt-oss-120b).
"""

import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from VIZAN.assistant.rule_engine import RuleEngine, IntentMatch
from VIZAN.assistant.intent_handler import handle_search_intent, handle_compare_intent
from VIZAN.storage.database import save_message, get_recent_messages, clear_session_history

# Load environment
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Initialize Groq client
groq_client = None
api_key = os.getenv("GROQ_API_KEY")
if api_key and api_key.startswith("gsk_"):
    try:
        from groq import Groq
        groq_client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Notice: Groq client init skipped ({e})")

# Preferred models available on this API key in order of quality
GROQ_MODELS = [
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

SYSTEM_PROMPT = """You are VIZAN, an advanced AI Virtual Assistant and Intelligent Problem Solver.
You specialize in:
1. Classical Artificial Intelligence & Problem Solving:
   - Search Strategies: BFS, DFS, Iterative Deepening (IDS), A* Search, D* Lite, Hill Climbing, Simulated Annealing.
   - Puzzles & Logic: 8-Puzzle, Wumpus World (Knowledge-Based logic agent), Towers of Hanoi.
   - Probabilistic Models: Bayesian Networks, Expectation Maximization (GMMs), Hidden Markov Models (Viterbi & Forward).
2. General Programming, Mathematics, Reasoning, and Concept Explanations.

Guidelines for your personality and interaction:
- When asked "what is your name" or who you are, introduce yourself as VIZAN, the Intelligent Virtual Assistant and Problem Solver.
- You are genuinely conversational, friendly, analytical, and highly articulate.
- If the user talks to you in Hindi or Hinglish, reply naturally in friendly Hinglish/English. If they speak English, reply in English.
- Always remember the ongoing conversation history and refer back to previous context when helpful.
- Explain concepts step-by-step with clear examples, intuition, and formulas where appropriate.
- When the user asks you to execute an algorithm on a graph (e.g. "Run A* from Arad to Bucharest"), you have live access to the Python search engine that executes it with exact coordinates and costs.
"""


class VizanAgent:
    """Core Virtual Assistant Agent with continuous memory and problem-solving intelligence."""

    def __init__(self):
        self.rule_engine = RuleEngine()

    def process_message(self, user_text: str, session_id: str = "default") -> Dict[str, Any]:
        """Processes a user input with conversation memory and returns a structured response."""
        user_text = user_text.strip()
        if not user_text:
            return {"role": "assistant", "content": "Please enter a message or question."}

        # 1. Fetch recent conversation history from database for memory context
        recent_history = get_recent_messages(session_id, limit=10)

        # 2. Match Intent
        match: IntentMatch = self.rule_engine.match(user_text)
        intent = match.intent
        extra_data = {}

        # 3. Handle specific operational intents
        if intent == "RESET":
            clear_session_history(session_id)
            reply = "🧹 **Session Reset**: Conversation history cleared. How can I assist you with problem solving or AI concepts today?"

        elif intent == "HELP":
            reply = (
                "### 🤖 VIZAN Capabilities & Commands\n\n"
                "I am your Problem Solving Virtual Assistant. Here is what we can do:\n\n"
                "**1. Search Algorithms Execution (Live Python Engine):**\n"
                "- `Run A* from Arad to Bucharest` (finds the optimal 418km path and plots the graph)\n"
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
                "**4. General Conversation & Problem Solving:**\n"
                "- Ask me anything! Math, coding, algorithm questions, advice, or general conversation."
            )

        elif intent == "RUN_SEARCH":
            search_res = handle_search_intent(match.parameters)
            reply = search_res["text"]
            extra_data["search_result"] = search_res["search_result"]
            extra_data["graph"] = search_res["graph"]
            extra_data["start"] = search_res["start"]
            extra_data["goal"] = search_res["goal"]

        elif intent == "COMPARE_SEARCH":
            comp_res = handle_compare_intent()
            reply = comp_res["text"]
            extra_data["comparison_data"] = comp_res["comparison_data"]

        else:
            # General Chat, Conceptual Question, or Problem Solving Query
            # Dispatched to Groq with full conversational history memory
            reply = self._generate_conversational_reply(user_text, history=recent_history)

        # 4. Persist in database
        save_message(session_id, "user", user_text, intent=intent)
        save_message(session_id, "assistant", reply, intent=intent)

        return {
            "role": "assistant",
            "content": reply,
            "intent": intent,
            "extra_data": extra_data
        }

    def _generate_conversational_reply(self, user_text: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates conversational and problem-solving replies using Groq API with
        continuous conversational memory.
        """
        if groq_client:
            # Build conversation payload with memory history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            if history:
                for turn in history:
                    r = turn.get("role")
                    c = turn.get("content")
                    if r in ["user", "assistant"] and c:
                        # Truncate very long past turns to save context window
                        truncated_content = c if len(c) < 1500 else c[:1500] + "..."
                        messages.append({"role": r, "content": truncated_content})

            # Append current turn
            messages.append({"role": "user", "content": user_text})

            for model_id in GROQ_MODELS:
                try:
                    response = groq_client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=800
                    )
                    reply_text = response.choices[0].message.content
                    if reply_text and reply_text.strip():
                        return reply_text.strip()
                except Exception as err:
                    # Model failed, try fallback model in list
                    continue

        # Intelligent offline fallback if API is unreachable
        return (
            f"Hello! I am **VIZAN**, your AI Virtual Assistant and Problem Solver.\n\n"
            f"You asked: *\"{user_text}\"*\n\n"
            f"I can help you analyze and execute Classical Search algorithms (DFS, BFS, IDS, A*, D*, Hill Climbing, Simulated Annealing), "
            f"Classical AI Problems (Wumpus World, 8-Puzzle, Towers of Hanoi), and Probabilistic Models.\n\n"
            f"Try asking:\n"
            f"- `Run A* from Arad to Bucharest`\n"
            f"- `Compare search algorithms`\n"
            f"- `What is the difference between BFS and DFS?`\n"
            f"- `Help` for all commands!"
        )
