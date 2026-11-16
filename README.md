# VIZAN – AI-Based Intelligent Virtual Assistant & Problem Solver

VIZAN is a comprehensive Artificial Intelligence Virtual Assistant and Autonomous Problem Solver engineered from first principles in Python.

Unlike conventional chatbots that simply generate text, VIZAN's core computational engine is **100% deterministic, robust, and self-contained in pure Python**. It performs actual pathfinding on weighted graphs, executes knowledge-based propositional logic, solves sliding tile state-spaces with heuristics, and models uncertain probabilistic systems.

---

## 🌟 Key Capabilities & Architecture

### 1. 🔍 Algorithms Studio (Pathfinding & Optimization)
- **Uninformed Search**:
  - **Depth First Search (DFS)**: LIFO exploration with cycle detection.
  - **Breadth First Search (BFS)**: FIFO queue guaranteeing hop-optimal paths.
  - **Iterative Deepening Search (IDS)**: Combines DFS linear memory with BFS optimality.
- **Informed Search**:
  - **A\* Search**: Best-first heuristic search evaluating $f(n) = g(n) + h(n)$ with Euclidean & Manhattan heuristics.
- **Dynamic Replanning**:
  - **D\* Lite**: Incremental heuristic search capable of real-time replanning when dynamic edge obstacles are detected.
- **Local & Metaheuristic Search**:
  - **Hill Climbing**: Steepest-ascent local search detecting local maxima, plateaus, and ridges.
  - **Simulated Annealing**: Stochastic metaheuristic with Boltzmann acceptance probability $P = \exp(-\Delta E / T)$ and geometric cooling schedules.
- **Interactive Graph Customizer**:
  - Benchmark topologies (Romania Road Map, Hierarchical Tree, 5x5 Grid Maze).
  - Live graph editing: Add/update edges, modify weights, remove roads (obstacles), and add custom nodes with live visualization.
  - All-algorithm comparative benchmarking side-by-side.

### 2. 🧩 Classical AI Problem Solvers
- **Wumpus World**: Grid environment with stench, breeze, pits, wumpus, and logical proposition deduction agent.
- **8-Puzzle**: 3x3 sliding tile problem with inversion solvability checks and A* Manhattan heuristic solver.
- **Towers of Hanoi**: Configurable disk counts, recursive solver, and ternary Sierpinski state-space graph.

### 3. 🎲 Probabilistic Reasoning
- **Bayesian Networks**: Directed Acyclic Graphs with CPTs and exact inference via variable elimination.
- **Expectation Maximization (EM)**: Gaussian Mixture Models parameter estimation with log-likelihood convergence.
- **Hidden Markov Models (HMM)**: Forward evaluation algorithm and Viterbi decoding algorithm.

### 4. 🤖 Groq Conversational Intelligence & Execution Architecture
- **Multi-Stage Processing Pipeline**:
  1. **Natural Language Understanding & Semantic Parsing**: Groq API parses user queries in English, Hindi, or Hinglish into structured JSON command payloads (`action`, `algorithm`, `start_node`, `goal_node`).
  2. **Deterministic Algorithm Execution (Pure Python)**: Algorithms are executed 100% in pure Python (`astar.py`, `bfs.py`, etc.). The LLM **never** simulates or invents mathematical or pathfinding results.
  3. **Conversational Explanation**: Groq takes the verified Python output (path, cost, nodes expanded, runtime) and explains it in simple, intuitive language.
  4. **Multi-Turn Context & Memory**: Continuous SQLite persistence (`vizan.db`) enables multi-turn memory retention.
  5. **Guaranteed Offline Fallback**: If the Groq API key is missing or offline, the system automatically falls back to the deterministic RuleEngine, ensuring all algorithms remain 100% executable through both chat and the visual dashboard.

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/vjkumar2756/Virtual_Vizan.git
cd Virtual_Vizan
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r VIZAN/requirements.txt
```

### 4. Configure Environment Variables
Copy the example environment template:
```bash
cp VIZAN/.env.example VIZAN/.env
```
Add your Groq API key in `VIZAN/.env`:
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

### 5. Launch VIZAN Dashboard
```bash
streamlit run VIZAN/main.py
```
Open your browser and navigate to: **`http://localhost:8501`**

---

## 🧪 Running Unit Tests
VIZAN comes with an automated pytest suite covering all algorithms and edge cases:
```bash
pytest VIZAN/tests/ -v
```

---

## 📁 Project Structure
```text
VIZAN/
├── main.py                          # Application entry point
├── requirements.txt                 # Pinned dependencies
├── .env.example                     # Configuration template
├── README.md                        # Documentation
│
├── frontend/                        # Streamlit UI Layer
│   ├── dashboard.py                 # Home & Chat Assistant
│   ├── search_ui.py                 # Algorithms Studio
│   ├── problem_ui.py                # Classical Problems UI
│   └── assistant_ui.py              # Dedicated Chat UI
│
├── algorithms/                      # Core AI Computational Engine
│   ├── search/                      # Search Algorithms
│   │   ├── base.py                  # Graph structures & SearchResult contract
│   │   ├── dfs.py                   # Depth First Search
│   │   ├── bfs.py                   # Breadth First Search
│   │   ├── ids.py                   # Iterative Deepening Search
│   │   ├── astar.py                 # A* Search
│   │   ├── dstar.py                 # D* Lite
│   │   ├── hill_climbing.py         # Hill Climbing
│   │   └── simulated_annealing.py   # Simulated Annealing
│   ├── problems/                    # Classical AI Problems
│   └── probabilistic/               # Probabilistic AI Models
│
├── assistant/                       # Assistant Engine
│   ├── agent.py                     # VizanAgent coordinator
│   ├── rule_engine.py               # Intent classifier
│   └── intent_handler.py            # Execution dispatcher
│
├── visualization/                   # NetworkX & Matplotlib Visualizers
│   └── graph_visualizer.py          # Graph visualizer
│
├── storage/                         # SQLite Persistence Layer
│   └── database.py                  # Database operations
│
└── tests/                           # Unit Test Suite
    └── test_search.py               # Search algorithms tests
```

