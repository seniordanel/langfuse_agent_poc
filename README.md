# 🔬 Multi-Agent Research System — Langfuse Demo

A **9-agent research-to-publication pipeline** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Langfuse](https://langfuse.com), designed to stress-test and showcase Langfuse's tracing, evaluation, and experiment features.

## Architecture

```
Researcher → Analyst → Writer → Fact-Checker ─┬→ Writer (revision loop, max 2)
                                                └→ Editor → SEO Optimizer → Compliance ─┬→ Editor (revision loop, max 2)
                                                                                         └→ Exec Summarizer → Translator → END
```

### Agents

| # | Agent | Role | Tools Used |
|---|-------|------|------------|
| 1 | **Researcher** | Gathers data via search & scrape | `search_tool`, `scrape_tool` |
| 2 | **Analyst** | Identifies trends, contradictions, gaps | — |
| 3 | **Writer** | Drafts the full report with mandatory sections | — |
| 4 | **Fact-Checker** | Validates draft against research data | — |
| 5 | **Editor** | Polishes tone, formatting, structure | — |
| 6 | **SEO Optimizer** | Extracts keywords, suggests title/meta | `keyword_extraction_tool` |
| 7 | **Compliance Reviewer** | Checks word count, heading rules, forbidden phrases | `word_count_tool` |
| 8 | **Executive Summarizer** | Generates a 3-sentence executive summary | — |
| 9 | **Translator** | Translates exec summary to Spanish & French | — |

### Feedback Loops

- **Fact-Checker → Writer**: If the draft has factual errors, it loops back for revision (max 2 iterations).
- **Compliance → Editor**: If the report fails formatting/structure rules, it loops back to the editor (max 2 iterations).

---

## Langfuse Features Demonstrated

| Feature | How |
|---------|-----|
| **Nested Traces & Spans** | 9 agent nodes generate deep span hierarchy per run |
| **Sessions** | 3 research topics grouped under 1 `session_id` |
| **Tags & Metadata** | Per-topic tags (`edge-computing`, `quantum`, `ai-regulation`) |
| **Scores (8 per trace)** | 3 deterministic + 3 LLM-as-judge + 2 performance |
| **Tool Calls** | 4 tools visible in trace spans |
| **Conditional Loops** | Visible in the trace when agents retry |
| **Datasets & Experiments** | Upload topics to Langfuse Dataset, run pipeline, link results |

### Evaluation Scores

| Score | Type | Description |
|-------|------|-------------|
| `format_compliance` | Deterministic | Checks for mandatory sections (Executive Summary, Introduction, Conclusion) |
| `word_count_check` | Deterministic | Minimum 500 words |
| `has_references` | Deterministic | Checks for URLs or `[n]` citation markers |
| `analytical_rigor` | LLM Judge | Depth of analysis, data usage, logical flow (1-10) |
| `readability` | LLM Judge | Clarity, flow, accessibility (1-10) |
| `factual_consistency` | LLM Judge | Draft vs research data alignment (1-10) |
| `latency_check` | Performance | Flags if run exceeds 90s |
| `cost_check` | Performance | Flags if cost exceeds $0.10 |

---

## Project Structure

```
langfuse_agent_poc/
├── main.py                     # Entry point: runs 3 topics under 1 session
├── run_dataset_experiment.py   # Langfuse Dataset & Experiment runner
├── eval_dataset.json           # 3 research topics with expected properties
├── pyproject.toml              # Project metadata & dependencies
├── .env                        # API keys (Langfuse + OpenAI)
└── src/
    ├── states.py               # AgentState TypedDict definition
    ├── tools.py                # Mock tools (search, scrape, keywords, word count)
    ├── agents.py               # 9 agent node functions
    ├── graph.py                # LangGraph workflow (9 nodes, 2 conditional loops)
    └── evals.py                # 8-score evaluation suite
```

---

## Setup

### Prerequisites

- Python 3.12 (3.14 has compatibility issues with Langfuse's dependencies)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A running [Langfuse](https://langfuse.com/docs/deployment/self-host) instance (local or cloud)
- An OpenAI API key

### Installation

```bash
# Create a Python 3.12 virtual environment
uv venv --python 3.12 .venv

# Install dependencies
uv pip install --python .venv/bin/python \
  langfuse langchain langchain-core langchain-openai \
  langgraph python-dotenv openai
```

### Configuration

Create a `.env` file in the project root:

```env
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_BASE_URL=http://localhost:3000

OPENAI_API_KEY=your-openai-api-key
# OPENAI_BASE_URL=https://custom-endpoint (optional)
```

> Get your Langfuse API keys from **Settings → API Keys** in your Langfuse dashboard.

---

## Usage

### Multi-Topic Session Run

Runs **3 research topics** under a single Langfuse session. Each topic generates a trace with nested spans and 8 evaluation scores.

```bash
.venv/bin/python main.py
```

**Topics:**
1. Future of Edge Computing in 2026
2. Quantum Machine Learning: Promise vs Reality in 2026
3. Global AI Regulation Landscape in 2026

### Dataset Experiment

Creates a **Langfuse Dataset**, runs each topic through the pipeline, and links results as an **Experiment**.

```bash
.venv/bin/python run_dataset_experiment.py
```

---

## Example Output

```
======================================================================
  MULTI-AGENT RESEARCH SYSTEM — Enhanced Langfuse Demo
  9 Agents | 3 Topics | 8 Evaluations per Trace
======================================================================

# RUN 1/3: Future of Edge Computing in 2026
--- 1. Researcher ---
  [Tool] Searching for: future of edge computing in 2026
--- 2. Analyst ---
--- 3. Writer ---
--- 4. Fact-Checker ---
  [Router] Fact-check failed (revision 1), looping to writer
--- 3. Writer ---
--- 4. Fact-Checker ---
--- 5. Editor ---
--- 6. SEO Optimizer ---
--- 7. Compliance Reviewer ---
  [Router] Compliance failed (revision 1), looping to editor
--- 5. Editor ---
--- 6. SEO Optimizer ---
--- 7. Compliance Reviewer ---
--- 8. Executive Summarizer ---
--- 9. Translator ---

  ✅ format_compliance: 1.00
  ✅ word_count_check: 1.00
  ❌ has_references: 0.00
  ⚠️ analytical_rigor: 0.60
  ✅ readability: 0.70
  ✅ factual_consistency: 1.00
  ✅ latency_check: 1.00
  ✅ cost_check: 1.00

  ALL RUNS COMPLETE — 3 topics in ~120s
  Total scores submitted: 24
```

---

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini (via `langchain-openai` and `openai`)
- **Orchestration**: LangGraph (stateful, cyclic agent workflows)
- **Observability**: Langfuse v3 (OpenTelemetry-based tracing)
- **Language**: Python 3.12
