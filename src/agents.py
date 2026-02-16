import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig

from src.states import AgentState
from src.tools import search_tool, scrape_tool, keyword_extraction_tool, word_count_tool

# Initialize models — two temperature profiles
llm_creative = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

llm_precise = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# ──────────────────────────────────────────────────────────────
# 1. RESEARCHER — uses search + scrape tools
# ──────────────────────────────────────────────────────────────
def researcher_node(state: AgentState, config: RunnableConfig):
    print("--- 1. Researcher ---")
    task = state["task"]

    researcher_llm = llm_creative.bind_tools([search_tool, scrape_tool])
    msg = researcher_llm.invoke(
        f"Research this topic deeply. Use both search and scrape tools: {task}"
    )

    findings = []

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            if name == "search_tool":
                res = search_tool.invoke(args["query"])
                findings.append(f"Search '{args['query']}':\n{res}")
            elif name == "scrape_tool":
                res = scrape_tool.invoke(args["url"])
                findings.append(f"Scraped {args['url']}:\n{res}")
    else:
        findings.append("LLM knowledge: " + msg.content)

    return {
        "research_data": findings,
        "iteration_log": state.get("iteration_log", []) + ["researcher"],
    }


# ──────────────────────────────────────────────────────────────
# 2. ANALYST — identifies trends, contradictions, gaps
# ──────────────────────────────────────────────────────────────
def analyst_node(state: AgentState, config: RunnableConfig):
    print("--- 2. Analyst ---")
    research_data = "\n\n".join(state["research_data"])
    task = state["task"]

    prompt = f"""You are a Senior Data Analyst. Analyze the following research data about "{task}".

Structure your analysis with these sections:
1. **Key Trends** — What patterns emerge?
2. **Market Data** — Any numbers, projections, market sizes?
3. **Contradictions** — Where do sources disagree?
4. **Gaps** — What information is missing?
5. **Risk Factors** — What could go wrong?

Research Data:
{research_data}

Produce a structured analysis. Be specific and cite data points."""

    response = llm_precise.invoke(prompt, config=config)
    return {
        "analysis": response.content,
        "iteration_log": state.get("iteration_log", []) + ["analyst"],
    }


# ──────────────────────────────────────────────────────────────
# 3. WRITER — drafts the report
# ──────────────────────────────────────────────────────────────
def writer_node(state: AgentState, config: RunnableConfig):
    print("--- 3. Writer ---")
    analysis = state["analysis"]
    critique = state.get("critique", "")
    task = state["task"]

    prompt = f"""You are an expert Tech Writer. Write a comprehensive, well-structured report on "{task}".

MANDATORY STRUCTURE (use these exact markdown headings):
## Executive Summary
## Introduction
## Key Findings
## Analysis
## Implications
## Conclusion
## References

Based on this analysis:
{analysis}
"""

    if critique and critique != "None":
        prompt += f"\n\nIMPORTANT — Address this critique from the Fact-Checker:\n{critique}\nFix all issues raised."

    response = llm_creative.invoke(prompt, config=config)
    return {
        "draft": response.content,
        "iteration_log": state.get("iteration_log", []) + ["writer"],
    }


# ──────────────────────────────────────────────────────────────
# 4. FACT-CHECKER — validates draft vs research (loop → Writer)
# ──────────────────────────────────────────────────────────────
def fact_checker_node(state: AgentState, config: RunnableConfig):
    print("--- 4. Fact-Checker ---")
    draft = state["draft"]
    research_data = "\n\n".join(state["research_data"])

    prompt = f"""You are a strict Fact-Checker. Compare the draft against the original research data.

Original Research:
{research_data}

Draft:
{draft[:3000]}

Tasks:
1. Identify factual errors or hallucinations NOT supported by research.
2. Check if tone is objective (no promotional language).
3. Verify all cited numbers match the research.

If there are significant errors, list them clearly as a critique.
If the draft is factually sound, reply with EXACTLY: "APPROVED"
"""

    response = llm_precise.invoke(prompt, config=config)
    content = response.content

    rev_count = state.get("revision_count", 0)
    log_entry = f"fact_checker_pass_{rev_count + 1}"

    if "APPROVED" in content.upper():
        return {
            "critique": "None",
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_approved"],
        }
    else:
        return {
            "critique": content,
            "revision_count": rev_count + 1,
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_rejected"],
        }


# ──────────────────────────────────────────────────────────────
# 5. EDITOR — polishes tone, formatting, structure
# ──────────────────────────────────────────────────────────────
def editor_node(state: AgentState, config: RunnableConfig):
    print("--- 5. Editor ---")
    draft = state["draft"]
    compliance_notes = state.get("compliance_notes", "")

    prompt = f"""You are a Senior Editor. Polish the following report for publication.

Requirements:
- Clean, professional Markdown formatting
- Clear heading hierarchy (## for sections)
- Consistent tone — authoritative but accessible
- Remove any redundant sentences
- Ensure smooth transitions between sections
- The report MUST contain these sections: Executive Summary, Introduction, Key Findings, Analysis, Implications, Conclusion, References

Draft:
{draft}
"""

    if compliance_notes and compliance_notes != "None":
        prompt += f"\n\nCOMPLIANCE ISSUES TO FIX:\n{compliance_notes}\nAddress ALL compliance issues."

    response = llm_creative.invoke(prompt, config=config)
    return {
        "draft": response.content,
        "iteration_log": state.get("iteration_log", []) + ["editor"],
    }


# ──────────────────────────────────────────────────────────────
# 6. SEO OPTIMIZER — extracts keywords, suggests improvements
# ──────────────────────────────────────────────────────────────
def seo_optimizer_node(state: AgentState, config: RunnableConfig):
    print("--- 6. SEO Optimizer ---")
    draft = state["draft"]

    # Use keyword extraction tool
    keywords_raw = keyword_extraction_tool.invoke(draft[:2000])
    keywords = [k.strip() for k in keywords_raw.split(",")]

    seo_llm = llm_precise.bind_tools([keyword_extraction_tool])
    prompt = f"""You are an SEO Specialist. Given these extracted keywords: {keywords_raw}

And this draft (first 1500 chars):
{draft[:1500]}

Suggest:
1. An optimized title (max 70 chars)
2. A meta description (max 160 chars)
3. Top 5 keywords ranked by importance

Output as structured text."""

    response = seo_llm.invoke(prompt, config=config)

    return {
        "seo_keywords": keywords,
        "iteration_log": state.get("iteration_log", []) + ["seo_optimizer"],
    }


# ──────────────────────────────────────────────────────────────
# 7. COMPLIANCE REVIEWER — checks structure rules (loop → Editor)
# ──────────────────────────────────────────────────────────────
def compliance_reviewer_node(state: AgentState, config: RunnableConfig):
    print("--- 7. Compliance Reviewer ---")
    draft = state["draft"]

    # Use word count tool
    wc_result = word_count_tool.invoke(draft)

    prompt = f"""You are a Compliance Reviewer. Check this report against publishing standards.

Word count info: {wc_result}

Rules to check:
1. Report must have AT LEAST 500 words.
2. Must contain sections: Executive Summary, Introduction, Conclusion.
3. Must NOT contain: "As an AI", "language model", "I cannot", "I'm sorry".
4. Must have at least 3 markdown headings (##).
5. Must not have any heading deeper than ### (no ####).

Report:
{draft[:3000]}

If ALL rules pass, reply with EXACTLY: "COMPLIANT"
If ANY rule fails, list the violations clearly."""

    response = llm_precise.invoke(prompt, config=config)
    content = response.content

    comp_rev = state.get("compliance_revision_count", 0)
    log_entry = f"compliance_pass_{comp_rev + 1}"

    if "COMPLIANT" in content.upper():
        return {
            "compliance_notes": "None",
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_passed"],
        }
    else:
        return {
            "compliance_notes": content,
            "compliance_revision_count": comp_rev + 1,
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_failed"],
        }


# ──────────────────────────────────────────────────────────────
# 8. EXECUTIVE SUMMARIZER — 3-sentence summary
# ──────────────────────────────────────────────────────────────
def exec_summarizer_node(state: AgentState, config: RunnableConfig):
    print("--- 8. Executive Summarizer ---")
    draft = state["draft"]

    prompt = f"""You are an Executive Summarizer. Read this report and produce an executive summary.

Requirements:
- Exactly 3 sentences.
- First sentence: the main finding.
- Second sentence: the key implication.
- Third sentence: the recommended action.

Report:
{draft[:3000]}

Output ONLY the 3-sentence summary, nothing else."""

    response = llm_precise.invoke(prompt, config=config)
    return {
        "executive_summary": response.content,
        "iteration_log": state.get("iteration_log", []) + ["exec_summarizer"],
    }


# ──────────────────────────────────────────────────────────────
# 9. TRANSLATOR — translates exec summary into 2 languages
# ──────────────────────────────────────────────────────────────
def translator_node(state: AgentState, config: RunnableConfig):
    print("--- 9. Translator ---")
    summary = state.get("executive_summary", "")

    prompt = f"""Translate the following executive summary into Spanish and French.

Summary:
{summary}

Output format:
SPANISH:
[translation]

FRENCH:
[translation]"""

    response = llm_precise.invoke(prompt, config=config)
    content = response.content

    translations = {"raw": content}
    if "SPANISH:" in content and "FRENCH:" in content:
        parts = content.split("FRENCH:")
        spanish_part = parts[0].replace("SPANISH:", "").strip()
        french_part = parts[1].strip() if len(parts) > 1 else ""
        translations = {"es": spanish_part, "fr": french_part}

    return {
        "translated_summaries": translations,
        "final_output": state["draft"],
        "iteration_log": state.get("iteration_log", []) + ["translator"],
    }
