from langgraph.graph import StateGraph, END
from src.states import AgentState
from src.agents import (
    researcher_node,
    analyst_node,
    writer_node,
    fact_checker_node,
    editor_node,
    seo_optimizer_node,
    compliance_reviewer_node,
    exec_summarizer_node,
    translator_node,
)


# ─── Conditional Routing Functions ───────────────────────────

def route_fact_checker(state: AgentState) -> str:
    """Route based on fact-check result: loop back to writer or proceed to editor."""
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    if not critique or critique == "None":
        return "editor"

    # Max 2 revision loops to avoid infinite cycling
    if revision_count >= 2:
        print("  [Router] Max fact-check revisions reached, proceeding to editor")
        return "editor"

    print(f"  [Router] Fact-check failed (revision {revision_count}), looping to writer")
    return "writer"


def route_compliance(state: AgentState) -> str:
    """Route based on compliance result: loop back to editor or proceed to summarizer."""
    compliance_notes = state.get("compliance_notes", "")
    comp_rev = state.get("compliance_revision_count", 0)

    if not compliance_notes or compliance_notes == "None":
        return "exec_summarizer"

    # Max 2 compliance loops
    if comp_rev >= 2:
        print("  [Router] Max compliance revisions reached, proceeding to summarizer")
        return "exec_summarizer"

    print(f"  [Router] Compliance failed (revision {comp_rev}), looping to editor")
    return "editor"


# ─── Build the Graph ─────────────────────────────────────────

workflow = StateGraph(AgentState)

# Add all 9 nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("writer", writer_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.add_node("editor", editor_node)
workflow.add_node("seo_optimizer", seo_optimizer_node)
workflow.add_node("compliance_reviewer", compliance_reviewer_node)
workflow.add_node("exec_summarizer", exec_summarizer_node)
workflow.add_node("translator", translator_node)

# ─── Edges ───────────────────────────────────────────────────
# Linear flow: Researcher → Analyst → Writer
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "writer")

# Writer → Fact-Checker
workflow.add_edge("writer", "fact_checker")

# Conditional 1: Fact-Checker → Writer (loop) or Editor
workflow.add_conditional_edges(
    "fact_checker",
    route_fact_checker,
    {"writer": "writer", "editor": "editor"},
)

# Editor → SEO Optimizer
workflow.add_edge("editor", "seo_optimizer")

# SEO → Compliance Reviewer
workflow.add_edge("seo_optimizer", "compliance_reviewer")

# Conditional 2: Compliance → Editor (loop) or Exec Summarizer
workflow.add_conditional_edges(
    "compliance_reviewer",
    route_compliance,
    {"editor": "editor", "exec_summarizer": "exec_summarizer"},
)

# Exec Summarizer → Translator → END
workflow.add_edge("exec_summarizer", "translator")
workflow.add_edge("translator", END)

# ─── Compile ─────────────────────────────────────────────────
app = workflow.compile()
