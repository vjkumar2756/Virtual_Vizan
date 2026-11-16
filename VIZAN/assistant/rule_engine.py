"""
Rule Engine for deterministic intent detection and parameter extraction in VIZAN.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class IntentMatch:
    intent: str
    confidence: float
    parameters: Dict[str, Any]
    raw_query: str


class RuleEngine:
    """Classifies user queries into structured actionable intents with extracted parameters."""

    def __init__(self):
        # Patterns for Search Algorithms with robust word boundaries
        self.search_patterns = [
            (r"(?:\b|^)(a\*|astar|a\s+star)(?:\s|$|\b)", "astar"),
            (r"\b(bfs|breadth\s+first)\b", "bfs"),
            (r"\b(dfs|depth\s+first)\b", "dfs"),
            (r"\b(ids|iterative\s+deepening)\b", "ids"),
            (r"(?:\b|^)(d\*|dstar|d\s+star|dynamic\s+a\*)(?:\s|$|\b)", "dstar"),
            (r"\b(hill\s+climbing|hill\s+climb)\b", "hill_climbing"),
            (r"\b(simulated\s+annealing|annealing)\b", "simulated_annealing"),
        ]

    def match(self, text: str) -> IntentMatch:
        cleaned = text.strip().lower()

        # 1. Reset / Clear
        if re.search(r"\b(clear|reset|restart|start over|naya chat|saaf karo|delete history)\b", cleaned):
            return IntentMatch(intent="RESET", confidence=1.0, parameters={}, raw_query=text)

        # 2. Help
        if re.search(r"\b(help|commands|what can you do|kya kar sakte ho|options|guide|madad)\b", cleaned):
            return IntentMatch(intent="HELP", confidence=1.0, parameters={}, raw_query=text)

        # 3. Compare algorithms
        if re.search(r"\b(compare|benchmark|comparison|vs|difference between)\b", cleaned) and any(
            kw in cleaned for kw in ["algorithm", "algo", "bfs", "dfs", "astar", "a*", "search"]
        ):
            return IntentMatch(intent="COMPARE_SEARCH", confidence=0.95, parameters={}, raw_query=text)

        # 4. Search algorithm execution (if asked to run, execute, or has start/goal or "run <algo>")
        is_run_request = bool(re.search(r"\b(run|execute|apply|find path|path from|chalao|solve on graph)\b", cleaned)) or ("from " in cleaned and "to " in cleaned)
        is_explain_request = bool(re.search(r"\b(what is|explain|define|kya hai|batao|samjhao|kya hota|kaise kaam|tell me about|how does)\b", cleaned))

        for pattern, algo_name in self.search_patterns:
            if re.search(pattern, cleaned):
                # If the user is asking to explain/teach/describe without asking to run, let conversational AI explain
                if is_explain_request and not is_run_request:
                    return IntentMatch(intent="GENERAL_CHAT", confidence=0.9, parameters={"algorithm": algo_name}, raw_query=text)

                # Only run if explicitly requested or clearly framed as a pathfinding query
                if is_run_request:
                    from_match = re.search(r"from\s+([A-Za-z0-9_\(\),]+)", text, re.IGNORECASE)
                    to_match = re.search(r"to\s+([A-Za-z0-9_\(\),]+)", text, re.IGNORECASE)

                    start_node = from_match.group(1).strip() if from_match else "Arad"
                    goal_node = to_match.group(1).strip() if to_match else "Bucharest"

                    return IntentMatch(
                        intent="RUN_SEARCH",
                        confidence=0.95,
                        parameters={
                            "algorithm": algo_name,
                            "start": start_node,
                            "goal": goal_node,
                            "graph": "romania"
                        },
                        raw_query=text
                    )

        # 5. 8-Puzzle
        if re.search(r"\b(8 puzzle|eight puzzle|tile puzzle)\b", cleaned):
            return IntentMatch(intent="RUN_8_PUZZLE", confidence=0.9, parameters={}, raw_query=text)

        # 6. Wumpus World
        if re.search(r"\b(wumpus|wumpus world)\b", cleaned):
            return IntentMatch(intent="RUN_WUMPUS", confidence=0.9, parameters={}, raw_query=text)

        # 7. Towers of Hanoi
        if re.search(r"\b(hanoi|towers of hanoi|tower of hanoi)\b", cleaned):
            disks_match = re.search(r"(\d+)\s+disks?", cleaned)
            disks = int(disks_match.group(1)) if disks_match else 3
            return IntentMatch(intent="RUN_HANOI", confidence=0.9, parameters={"disks": disks}, raw_query=text)

        # 8. Probabilistic Models
        if re.search(r"\b(bayesian|bayes|cpt|belief network)\b", cleaned):
            return IntentMatch(intent="RUN_BAYESIAN", confidence=0.85, parameters={}, raw_query=text)
        if re.search(r"\b(em algorithm|expectation maximization|gmm|gaussian mixture)\b", cleaned):
            return IntentMatch(intent="RUN_EM", confidence=0.85, parameters={}, raw_query=text)
        if re.search(r"\b(hmm|hidden markov|viterbi|forward algorithm)\b", cleaned):
            return IntentMatch(intent="RUN_HMM", confidence=0.85, parameters={}, raw_query=text)

        # 9. General Chat or Question
        return IntentMatch(intent="GENERAL_CHAT", confidence=0.5, parameters={}, raw_query=text)
