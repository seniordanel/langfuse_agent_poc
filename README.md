# 🔬 Multi-Agent Research System — Langfuse Demo

A **11-agent, 12-tool research-to-publication pipeline** built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Langfuse](https://langfuse.com), designed to stress-test and showcase Langfuse's tracing, evaluation, and experiment features.

## Architecture

```
Researcher → Analyst → Data Enricher → Writer ←──────── Fact-Checker (max 2 revisions)
                                                              │
                                                              ↓
            Quality Gate ← Translator ← Exec Summarizer ← Compliance ←── SEO ← Editor
                │                                          ↑ (max 2)            ↑
                └──────────────────────────────────────────────────────────────→─┘
                                      (max 1 revision)
```

### Agents

| # | Agent | Role | Tools Used |
|---|-------|------|------------|
| 1 | **Researcher** | Gathers data via search & scrape | `search_tool`, `scrape_tool` |
| 2 | **Analyst** | Identifies trends, contradictions, gaps | `sentiment_analysis_tool`, `statistics_extractor_tool` |
| 3 | **Data Enricher** | Second research pass to fill gaps from analysis | `search_tool`, `scrape_tool` |
| 4 | **Writer** | Drafts the full report with mandatory sections | `headline_generator_tool` |
| 5 | **Fact-Checker** | Validates draft against research data | `plagiarism_check_tool` |
| 6 | **Editor** | Polishes tone, formatting, structure | `citation_formatter_tool` |
| 7 | **SEO Optimizer** | Extracts keywords, suggests title/meta | `keyword_extraction_tool` |
| 8 | **Compliance Reviewer** | Checks word count, readability, heading rules | `word_count_tool`, `readability_score_tool` |
| 9 | **Executive Summarizer** | Generates a 3-sentence executive summary | `text_summarizer_tool` |
| 10 | **Translator** | Translates exec summary to Spanish & French | `translation_quality_tool` |
| 11 | **Quality Gate** | Final holistic quality check before publication | `word_count_tool`, `readability_score_tool` |

### Tools (12)

| Tool | Type | Description |
|------|------|-------------|
| `search_tool` | Research | Simulated web search with topic-aware results |
| `scrape_tool` | Research | Simulated page scraping with realistic article content |
| `keyword_extraction_tool` | SEO | Heuristic keyword extraction from text |
| `word_count_tool` | Metrics | Simple word count |
| `sentiment_analysis_tool` | Analysis | Lexicon-based sentiment scoring (-1 to +1) |
| `statistics_extractor_tool` | Analysis | Regex extraction of percentages, financials, dates, ranges |
| `headline_generator_tool` | Creative | 5 headline variations (news, question, how-to, listicle, provocative) |
| `plagiarism_check_tool` | Validation | 5-gram overlap detection between draft and source |
| `citation_formatter_tool` | Formatting | Converts raw URLs to numbered markdown citations |
| `readability_score_tool` | Metrics | Flesch-Kincaid grade level calculation |
| `text_summarizer_tool` | Summarization | TF-based extractive top-3 sentence selection |
| `translation_quality_tool` | Validation | Length ratio, sentence alignment, number preservation checks |

### Feedback Loops (3)

1. **Fact-Checker → Writer** — If the draft has factual errors or high plagiarism overlap, it loops back for revision (max 2 iterations)
2. **Compliance → Editor** — If the report fails formatting, word count, or readability rules, it loops back to the editor (max 2 iterations)
3. **Quality Gate → Editor** — Final quality check; if the report isn't publication-ready, it loops back for one more edit pass (max 1 iteration)

---

## Langfuse Features Demonstrated

| Feature | How |
|---------|-----|
| **Nested Traces & Spans** | 11 agent nodes with nested tool-call sub-spans per run |
| **Sessions** | 3 research topics grouped under 1 `session_id` |
| **Tags & Metadata** | Per-topic tags (`edge-computing`, `quantum`, `ai-regulation`) |
| **Scores (8 per trace)** | 3 deterministic + 3 LLM-as-judge + 2 performance |
| **Tool Calls** | 12 tools visible as sub-spans across 10 of 11 agents |
| **Conditional Loops** | 3 feedback loops visible when agents retry |
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
├── agent_poc.py                # Standalone ReAct agent with Langfuse @observe tracing
├── run_evals.py                # Direct evaluation runner (non-LangGraph)
├── run_dataset_experiment.py   # Langfuse Dataset & Experiment runner
├── eval_dataset.json           # 3 research topics with expected properties
├── pyproject.toml              # Project metadata & dependencies
├── .env                        # API keys (Langfuse + OpenAI)
└── src/
    ├── states.py               # AgentState TypedDict (21 fields)
    ├── tools.py                # 12 mock tools (search, scrape, sentiment, etc.)
    ├── agents.py               # 11 agent node functions
    ├── graph.py                # LangGraph workflow (11 nodes, 3 conditional loops)
    ├── evals.py                # 8-score evaluation suite
    ├── openai_client.py        # OpenAI client wrapper with Langfuse token tracking
    └── mock_langfuse.py        # Mock Langfuse client for offline testing
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

Runs **3 research topics** under a single Langfuse session. Each topic generates a trace with nested agent and tool spans, plus 8 evaluation scores.

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

### Standalone ReAct Agent

A simpler single-agent demo using Langfuse's `@observe` decorator for tracing a ReAct loop.

```bash
.venv/bin/python agent_poc.py
```

---

## Example Output

```
======================================================================
  MULTI-AGENT RESEARCH SYSTEM — Enhanced Langfuse Demo
  11 Agents | 12 Tools | 3 Topics | 3 Feedback Loops | 8 Evaluations per Trace
======================================================================

# RUN 1/3: Future of Edge Computing in 2026
--- 1. Researcher ---
  [Tool] Searching for: future of edge computing in 2026
  [Tool] Scraping: https://tech-trends-2026.com/edge-computing
--- 2. Analyst ---
  [Tool] Analyzing sentiment (1523 chars)
  [Tool] Extracting statistics (1523 chars)
--- 3. Data Enricher ---
  [Tool] Searching for: edge computing energy consumption 2026
--- 4. Writer ---
  [Tool] Generating headlines for: Future of Edge Computing in 2026
--- 5. Fact-Checker ---
  [Tool] Checking plagiarism (draft=2800 chars, source=1500 chars)
  [Router] Fact-check failed (revision 1), looping to writer
--- 4. Writer ---
--- 5. Fact-Checker ---
--- 6. Editor ---
  [Tool] Formatting citations (3200 chars)
--- 7. SEO Optimizer ---
  [Tool] Extracting keywords from text (1500 chars)
--- 8. Compliance Reviewer ---
  [Tool] Word count: 650
  [Tool] Calculating readability (3000 chars)
--- 9. Executive Summarizer ---
  [Tool] Summarizing text (3500 chars)
--- 10. Translator ---
  [Tool] Checking translation quality (original=180 chars, translation=210 chars)
  [Tool] Checking translation quality (original=180 chars, translation=195 chars)
--- 11. Quality Gate ---
  [Tool] Word count: 650
  [Tool] Calculating readability (3000 chars)

  ✅ format_compliance: 1.00
  ✅ word_count_check: 1.00
  ✅ has_references: 1.00
  ⚠️ analytical_rigor: 0.60
  ✅ readability: 0.70
  ✅ factual_consistency: 1.00
  ✅ latency_check: 1.00
  ✅ cost_check: 1.00

  ALL RUNS COMPLETE — 3 topics in ~150s
  Total scores submitted: 24
```

---

## Tech Stack

- **LLM**: OpenAI GPT-4o-mini (via `langchain-openai` and `openai`)
- **Orchestration**: LangGraph (stateful, cyclic agent workflows)
- **Observability**: Langfuse v3 (OpenTelemetry-based tracing)
- **Language**: Python 3.12
