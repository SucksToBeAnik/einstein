# einstein — Project Context for AI Agents

## What is this project?

A lightweight, local CLI research assistant built in Python. Users run it from the terminal to get structured, high-quality research help on any topic. No browser, no GUI, no account required. Just a clean terminal tool powered by LLMs.

The goal is to publish this as an installable Python package (`pipx install einstein`). It must be fast, useful, and work with any OpenAI-compatible API (including local Ollama models).

---

## Tech Stack

- **Python** — primary language
- **LangChain / LangGraph** — agent orchestration and graph-based workflow
- **Typer** — CLI interface
- **Rich** — terminal output formatting
- **LLM** — any OpenAI-compatible model (configurable via env var or config file)

---

## Architecture Overview

The app is a **LangGraph state graph**. Every user query enters the graph, gets routed, and flows through the appropriate specialized node. The graph grows with each new pattern implemented.

### LangGraph State

The shared state object carried through the graph contains at minimum:

```python
{
  "query": str,           # the original user query
  "intent": str,          # detected route (set by router node)
  "output": str,          # final response to show the user
  # more fields added as patterns are implemented
}
```

---

## Agentic Patterns Implemented (so far)

### ✅ Pattern 1 — Prompt Chaining (Ch. 1)
**What it is:** Break a task into a linear sequence of LLM calls where each step's output feeds the next.
**Where used:** Inside each specialized node (e.g. the summarize node runs: extract key concepts → write summary → format output).

---

### ✅ Pattern 2 — Routing (Ch. 2)
**What it is:** A classifier node sits at the entry point of the graph. It reads the user query, determines intent, and uses a LangGraph conditional edge to dispatch to the correct specialized node.

**Router node behaviour:**
- Takes `state["query"]` as input
- Calls the LLM with a classification prompt
- Returns one of the intent labels below
- LangGraph `add_conditional_edges` routes to the matching node

**Intent labels and their nodes:**

| Intent | Node | What it does |
|---|---|---|
| `summarize` | `summarize_node` | Concise overview of a topic |
| `explain` | `explain_node` | Step-by-step pedagogical breakdown |
| `sources` | `sources_node` | Suggests papers, books, links to explore |
| `compare` | `compare_node` | Side-by-side breakdown of two concepts/things |
| `deep_research` | `deep_research_node` | Multi-step research on a complex question |
| `critique` | `critique_node` | Challenges assumptions in a claim or idea |
| `unknown` | `clarify_node` | Query too vague — asks a clarifying question |

**Design note:** The router prints the detected intent to the terminal before processing (e.g. `Routing as: explain`). This is intentional — it gives the user transparency and a lightweight human-in-the-loop check. Later this can become an interactive confirmation step.

---

## Patterns Planned (to be implemented in order)

### ⏳ Pattern 3 — Parallelization (Ch. 3)
**What it is:** Run independent sub-tasks concurrently instead of sequentially to reduce latency.
**Where it will be used:** The `compare_node`. Comparing two things (e.g. "RAG vs fine-tuning") means generating an analysis of each independently — these two analyses don't depend on each other, so they can run in parallel. A final synthesis step merges them after both complete.
**LangChain mechanism:** `RunnableParallel` in LCEL, or parallel branches in LangGraph.

---

### ⏳ Pattern 4 — Reflection (Ch. 4)
**What it is:** The agent critiques its own output and iteratively refines it before returning the final answer.
**Where it will be used:** The `deep_research_node` and optionally `explain_node`. After generating an initial answer, a critic prompt evaluates it for gaps, inaccuracies, or lack of clarity. The output is then revised. Loop runs a fixed number of times (e.g. max 2 iterations).

---

### ⏳ Pattern 5 — Tool Use / Function Calling (Ch. 5)
**What it is:** The agent can call external tools (web search, file read, APIs) instead of relying purely on LLM knowledge.
**Where it will be used:** `sources_node` (web search for real papers/links), `deep_research_node` (search + retrieve). This is the step that makes answers grounded and current rather than purely parametric.

---

### ⏳ Pattern 6 — Planning (Ch. 6)
**What it is:** Before executing, the agent generates a multi-step plan, then executes each step.
**Where it will be used:** `deep_research_node`. Instead of one big prompt, the agent first produces a research plan (list of sub-questions or steps), then executes each step in sequence, then synthesizes.

---

### ⏳ Pattern 7 — Multi-Agent Collaboration (Ch. 7)
**What it is:** Orchestrate multiple specialized agents working together toward a shared goal.
**Where it will be used:** `deep_research_node` evolves into a mini multi-agent pipeline: a Planner agent, a Researcher agent, and a Writer agent each handle their own concern.

---

### ⏳ Pattern 8 — Memory Management (Ch. 8)
**What it is:** Persist context across turns using short-term, episodic, or semantic memory.
**Where it will be used:** Across the whole app. Users should be able to ask follow-up questions and reference earlier queries in the same session. Episodic memory stores past research sessions locally.

---

### ⏳ Pattern 9 — Learning & Adaptation (Ch. 9)
**What it is:** The agent improves over time based on feedback.
**Where it will be used:** After each response, optionally prompt the user for a thumbs up/down. Store feedback locally. Use it to improve prompt selection or node behaviour over time.

---

### ⏳ Pattern 10 — Model Context Protocol / MCP (Ch. 10)
**What it is:** A standardized protocol for agents to discover and use external tools/services.
**Where it will be used:** Tool integration layer — instead of hardcoding tools, the app can discover and use MCP-compatible tool servers.

---

### ⏳ Pattern 11 — Goal Setting & Monitoring (Ch. 11)
**What it is:** Define, track, and adjust goals dynamically during execution.
**Where it will be used:** `deep_research_node` — the agent sets a research goal at the start and checks after each step whether the goal has been sufficiently addressed.

---

### ⏳ Pattern 12 — Exception Handling & Recovery (Ch. 12)
**What it is:** Detect failures, retry intelligently, and fall back gracefully.
**Where it will be used:** Across all tool-using nodes. API failures, empty results, and malformed LLM outputs are caught and handled without crashing the session.

---

### ⏳ Pattern 13 — Human-in-the-Loop / HITL (Ch. 13)
**What it is:** Pause execution at critical points to get human approval or input.
**Where it will be used:** `deep_research_node` — show the generated research plan to the user and ask for confirmation before executing it. Also used in `clarify_node`.

---

### ⏳ Pattern 14 — Knowledge Retrieval / RAG (Ch. 14)
**What it is:** Fetch relevant content from a local knowledge base to ground LLM responses.
**Where it will be used:** Users can point the app at a folder of PDFs/notes. The app indexes them and retrieves relevant chunks to augment answers with the user's own materials.

---

### ⏳ Pattern 15 — Inter-Agent Communication / A2A (Ch. 15)
**What it is:** Agents discover and communicate with each other via a standard protocol.
**Where it will be used:** Advanced mode — the research assistant can delegate sub-tasks to external specialist agents (e.g. a dedicated math agent, a code agent).

---

### ⏳ Pattern 16 — Resource-Aware Optimization (Ch. 16)
**What it is:** Adapt model size, tool usage, and compute to available resources.
**Where it will be used:** Config layer — users can set a "fast mode" (smaller model, no reflection) vs "thorough mode" (larger model, reflection + planning enabled).

---

### ⏳ Pattern 17 — Reasoning Techniques (Ch. 17)
**What it is:** Chain-of-Thought, Tree-of-Thought, Graph of Debates and other structured reasoning methods.
**Where it will be used:** `explain_node` and `critique_node` — use CoT prompting explicitly. `deep_research_node` optionally uses ToT for complex multi-hypothesis questions.

---

### ⏳ Pattern 18 — Guardrails / Safety Patterns (Ch. 18)
**What it is:** Input/output validation, content filtering, policy enforcement.
**Where it will be used:** Input guardrail at the entry point (before routing) and output guardrail before displaying results. Catches junk input, prompt injection attempts, and malformed outputs.

---

### ⏳ Pattern 19 — Evaluation & Monitoring (Ch. 19)
**What it is:** Track agent performance, detect quality drift, benchmark outputs.
**Where it will be used:** Optional `--eval` flag. Runs a set of benchmark queries and scores outputs. Logs latency and response quality locally.

---

### ⏳ Pattern 20 — Prioritization (Ch. 20)
**What it is:** Rank and sequence tasks by urgency, value, or dependencies.
**Where it will be used:** When the user submits a batch of research questions (e.g. from a file), the app prioritizes which to answer first based on complexity and dependency between questions.

---

### ⏳ Pattern 21 — Exploration & Discovery (Ch. 21)
**What it is:** Agents that actively hypothesize, explore, and discover new knowledge rather than just answering.
**Where it will be used:** A special `explore` mode — given a broad topic, the agent generates its own sub-questions, researches them, and surfaces surprising or non-obvious connections.

---

## CLI Design (Typer)

```bash
# Single query
research "how does RLHF work"

# Force a specific route (bypass router)
research "transformers vs RNNs" --mode compare

# Use a local folder as knowledge base (RAG — Pattern 14)
research "summarize my notes on attention" --docs ./my-notes/

# Thorough mode (enables reflection + planning)
research "explain the replication crisis" --thorough

# Batch mode (Planning + Prioritization patterns)
research --batch questions.txt

# Explore mode (Pattern 21)
research --explore "consciousness"
```

---

## Output

- Terminal output formatted with **Rich** (panels, markdown rendering, coloured labels)
- Every response optionally saved to `~/.research/sessions/YYYY-MM-DD_HH-MM.md`
- Sessions are plain markdown — readable without the tool

---

## Key Design Principles

1. **Lightweight by default** — a simple query should feel instant. Heavy patterns (reflection, planning, multi-agent) only activate when needed or explicitly requested.
2. **OpenAI-compatible** — works with OpenAI, Anthropic (via compatible wrapper), or local Ollama. Model configured via `RESEARCH_MODEL` env var.
3. **No cloud lock-in** — all storage is local. No accounts, no telemetry.
4. **Progressive enhancement** — each pattern adds capability without breaking what came before. The router is always the entry point.
5. **The router is sacred** — all queries enter through the router node. Never bypass it in normal flow (only via explicit `--mode` flag).