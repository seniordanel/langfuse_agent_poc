from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from src.states import AgentState
from src.tools import (
    search_tool,
    scrape_tool,
    keyword_extraction_tool,
    word_count_tool,
    sentiment_analysis_tool,
    readability_score_tool,
    plagiarism_check_tool,
    citation_formatter_tool,
    translation_quality_tool,
    headline_generator_tool,
    statistics_extractor_tool,
    text_summarizer_tool,
)

# Initialize models — two temperature profiles
llm_creative = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
)

llm_precise = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
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
# 2. ANALYST — uses sentiment_analysis_tool + statistics_extractor_tool
# ──────────────────────────────────────────────────────────────
def analyst_node(state: AgentState, config: RunnableConfig):
    print("--- 2. Analyst ---")
    research_data = "\n\n".join(state["research_data"])
    task = state["task"]

    # Run tools first to gather structured data
    sentiment_result = sentiment_analysis_tool.invoke(research_data[:3000])
    stats_result = statistics_extractor_tool.invoke(research_data[:3000])

    prompt = f"""You are a Senior Data Analyst. Analyze the following research data about "{task}".

You have access to pre-computed tool outputs below — incorporate them into your analysis.

## Sentiment Analysis Tool Output:
{sentiment_result}

## Statistics Extractor Tool Output:
{stats_result}

Structure your analysis with these sections:
1. **Key Trends** — What patterns emerge?
2. **Market Data** — Any numbers, projections, market sizes? (use the extracted stats)
3. **Sentiment Overview** — What is the overall tone? (use the sentiment analysis)
4. **Contradictions** — Where do sources disagree?
5. **Gaps** — What information is missing?
6. **Risk Factors** — What could go wrong?

Research Data:
{research_data}

Produce a structured analysis. Be specific and cite data points."""

    response = llm_precise.invoke(prompt, config=config)

    # Parse sentiment scores for state
    sentiment_scores = {"raw": sentiment_result}

    return {
        "analysis": response.content,
        "sentiment_scores": sentiment_scores,
        "iteration_log": state.get("iteration_log", []) + ["analyst"],
    }


# ──────────────────────────────────────────────────────────────
# 3. DATA ENRICHER — second research pass using search + scrape
# ──────────────────────────────────────────────────────────────
def data_enricher_node(state: AgentState, config: RunnableConfig):
    print("--- 3. Data Enricher ---")
    analysis = state["analysis"]
    task = state["task"]

    enricher_llm = llm_precise.bind_tools([search_tool, scrape_tool])
    msg = enricher_llm.invoke(
        f"""Based on this analysis of "{task}", identify the TOP knowledge gap and search for additional data to fill it.

Analysis:
{analysis[:2000]}

Use search_tool to find additional sources, then scrape_tool to get full content.
Focus on gaps, missing data points, or areas that need deeper research.""",
        config=config,
    )

    enrichments = []

    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            if name == "search_tool":
                res = search_tool.invoke(args["query"])
                enrichments.append(f"Enrichment search '{args['query']}':\n{res}")
            elif name == "scrape_tool":
                res = scrape_tool.invoke(args["url"])
                enrichments.append(f"Enrichment scraped {args['url']}:\n{res}")
    else:
        enrichments.append("Enrichment LLM knowledge: " + msg.content)

    # Merge enrichment data into research_data
    combined_research = state["research_data"] + enrichments

    return {
        "research_data": combined_research,
        "enrichment_data": enrichments,
        "iteration_log": state.get("iteration_log", []) + ["data_enricher"],
    }


# ──────────────────────────────────────────────────────────────
# 4. WRITER — uses headline_generator_tool
# ──────────────────────────────────────────────────────────────
def writer_node(state: AgentState, config: RunnableConfig):
    print("--- 4. Writer ---")
    analysis = state["analysis"]
    critique = state.get("critique", "")
    task = state["task"]

    # Generate headline options
    headlines_result = headline_generator_tool.invoke(task)

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

Here are suggested headline options from the headline generator tool — pick the best one or create your own:
{headlines_result}
"""

    if critique and critique != "None":
        prompt += f"\n\nIMPORTANT — Address this critique from the Fact-Checker:\n{critique}\nFix all issues raised."

    response = llm_creative.invoke(prompt, config=config)
    return {
        "draft": response.content,
        "iteration_log": state.get("iteration_log", []) + ["writer"],
    }


# ──────────────────────────────────────────────────────────────
# 5. FACT-CHECKER — uses plagiarism_check_tool
# ──────────────────────────────────────────────────────────────
def fact_checker_node(state: AgentState, config: RunnableConfig):
    print("--- 5. Fact-Checker ---")
    draft = state["draft"]
    research_data = "\n\n".join(state["research_data"])

    # Run plagiarism check tool
    plagiarism_result = plagiarism_check_tool.invoke({
        "draft": draft[:3000],
        "source_text": research_data[:3000],
    })

    prompt = f"""You are a strict Fact-Checker. Compare the draft against the original research data.

Original Research:
{research_data[:3000]}

Draft:
{draft[:3000]}

Plagiarism Check Tool Output:
{plagiarism_result}

Tasks:
1. Identify factual errors or hallucinations NOT supported by research.
2. Check if tone is objective (no promotional language).
3. Verify all cited numbers match the research.
4. Consider the plagiarism check results — if overlap is HIGH, the draft needs more original language.

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
# 6. EDITOR — uses citation_formatter_tool
# ──────────────────────────────────────────────────────────────
def editor_node(state: AgentState, config: RunnableConfig):
    print("--- 6. Editor ---")
    draft = state["draft"]
    compliance_notes = state.get("compliance_notes", "")

    # Format citations in the draft
    formatted_draft = citation_formatter_tool.invoke(draft[:4000])

    prompt = f"""You are a Senior Editor. Polish the following report for publication.

Requirements:
- Clean, professional Markdown formatting
- Clear heading hierarchy (## for sections)
- Consistent tone — authoritative but accessible
- Remove any redundant sentences
- Ensure smooth transitions between sections
- The report MUST contain these sections: Executive Summary, Introduction, Key Findings, Analysis, Implications, Conclusion, References
- Incorporate the formatted citations below where appropriate

Citation Formatter Output:
{formatted_draft[:2000]}

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
# 7. SEO OPTIMIZER — uses keyword_extraction_tool (unchanged)
# ──────────────────────────────────────────────────────────────
def seo_optimizer_node(state: AgentState, config: RunnableConfig):
    print("--- 7. SEO Optimizer ---")
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
# 8. COMPLIANCE REVIEWER — uses word_count_tool + readability_score_tool
# ──────────────────────────────────────────────────────────────
def compliance_reviewer_node(state: AgentState, config: RunnableConfig):
    print("--- 8. Compliance Reviewer ---")
    draft = state["draft"]

    # Use both tools
    wc_result = word_count_tool.invoke(draft)
    readability_result = readability_score_tool.invoke(draft[:3000])

    prompt = f"""You are a Compliance Reviewer. Check this report against publishing standards.

Word count info: {wc_result}

Readability Analysis:
{readability_result}

Rules to check:
1. Report must have AT LEAST 500 words.
2. Must contain sections: Executive Summary, Introduction, Conclusion.
3. Must NOT contain: "As an AI", "language model", "I cannot", "I'm sorry".
4. Must have at least 3 markdown headings (##).
5. Must not have any heading deeper than ### (no ####).
6. Readability grade level should be between 8 and 14 (accessible yet professional).

Report:
{draft[:3000]}

If ALL rules pass, reply with EXACTLY: "COMPLIANT"
If ANY rule fails, list the violations clearly."""

    response = llm_precise.invoke(prompt, config=config)
    content = response.content

    comp_rev = state.get("compliance_revision_count", 0)
    log_entry = f"compliance_pass_{comp_rev + 1}"

    # Parse readability grade from tool output
    import re
    grade_match = re.search(r"Grade Level:\s*([\d.]+)", readability_result)
    readability_grade = float(grade_match.group(1)) if grade_match else 0.0

    if "COMPLIANT" in content.upper():
        return {
            "compliance_notes": "None",
            "readability_grade": readability_grade,
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_passed"],
        }
    else:
        return {
            "compliance_notes": content,
            "compliance_revision_count": comp_rev + 1,
            "readability_grade": readability_grade,
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_failed"],
        }


# ──────────────────────────────────────────────────────────────
# 9. EXECUTIVE SUMMARIZER — uses text_summarizer_tool
# ──────────────────────────────────────────────────────────────
def exec_summarizer_node(state: AgentState, config: RunnableConfig):
    print("--- 9. Executive Summarizer ---")
    draft = state["draft"]

    # Run extractive summarizer tool first
    extractive_summary = text_summarizer_tool.invoke(draft[:4000])

    prompt = f"""You are an Executive Summarizer. Read this report and produce an executive summary.

Here is an extractive summary from the text summarizer tool to guide you:
{extractive_summary}

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
# 10. TRANSLATOR — uses translation_quality_tool
# ──────────────────────────────────────────────────────────────
def translator_node(state: AgentState, config: RunnableConfig):
    print("--- 10. Translator ---")
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

    # Run translation quality checks
    quality_results = {}
    if "es" in translations:
        es_quality = translation_quality_tool.invoke({
            "original": summary,
            "translation": translations["es"],
        })
        quality_results["es_quality"] = es_quality
    if "fr" in translations:
        fr_quality = translation_quality_tool.invoke({
            "original": summary,
            "translation": translations["fr"],
        })
        quality_results["fr_quality"] = fr_quality

    translations["quality_checks"] = quality_results

    return {
        "translated_summaries": translations,
        "final_output": state["draft"],
        "iteration_log": state.get("iteration_log", []) + ["translator"],
    }


# ──────────────────────────────────────────────────────────────
# 11. QUALITY GATE — uses word_count_tool + readability_score_tool
#     Final quality check with feedback loop to Editor
# ──────────────────────────────────────────────────────────────
def quality_gate_node(state: AgentState, config: RunnableConfig):
    print("--- 11. Quality Gate ---")
    draft = state["draft"]
    exec_summary = state.get("executive_summary", "")
    translations = state.get("translated_summaries", {})

    # Final quality checks using tools
    wc_result = word_count_tool.invoke(draft)
    readability_result = readability_score_tool.invoke(draft[:3000])

    prompt = f"""You are a Final Quality Gate reviewer. This is the LAST check before publication.

Perform a holistic quality assessment:

Word Count: {wc_result}
Readability: {readability_result}

Check items:
1. Does the report have a clear narrative arc (intro → findings → implications → conclusion)?
2. Is the executive summary present and concise (3 sentences)?
3. Are translations present?
4. Is the overall quality PUBLICATION-READY?

Executive Summary:
{exec_summary}

Report excerpt (first 2000 chars):
{draft[:2000]}

If quality is PUBLICATION-READY, respond with a score 8-10 and the word "PASSED".
If quality needs improvement, respond with a score 1-7, the word "FAILED", and list specific issues to fix.

Format: SCORE: [number] | VERDICT: [PASSED/FAILED] | NOTES: [details]"""

    response = llm_precise.invoke(prompt, config=config)
    content = response.content

    # Parse score
    import re
    score_match = re.search(r"SCORE:\s*(\d+)", content)
    quality_score = float(score_match.group(1)) / 10.0 if score_match else 0.5

    quality_rev = state.get("quality_revision_count", 0)
    log_entry = f"quality_gate_pass_{quality_rev + 1}"

    if "PASSED" in content.upper():
        return {
            "quality_score": quality_score,
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_passed"],
        }
    else:
        return {
            "quality_score": quality_score,
            "quality_revision_count": quality_rev + 1,
            "compliance_notes": f"Quality Gate feedback: {content}",
            "iteration_log": state.get("iteration_log", []) + [log_entry + "_failed"],
        }
