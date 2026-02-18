import re
import random
import math
from langchain_core.tools import tool


@tool
def search_tool(query: str) -> str:
    """
    Simulates a web search engine. Returns a list of search results with snippets.
    Useful for finding information about a topic.
    """
    print(f"  [Tool] Searching for: {query}")

    edge_computing_results = [
        {
            "title": "The State of Edge Computing in 2026",
            "snippet": "By 2026, edge computing is expected to process 75% of enterprise data. Key drivers include 6G rollouts and AI at the edge.",
            "url": "https://tech-trends-2026.com/edge-computing",
        },
        {
            "title": "Challenges in Edge AI Deployment",
            "snippet": "Security, latency consistency, and hardware fragmentation remain top challenges for edge AI deployments in 2025-2026.",
            "url": "https://ai-daily.org/edge-ai-challenges",
        },
        {
            "title": "Market Size of Edge Computing",
            "snippet": "The global edge computing market is projected to reach $150 billion by 2028, growing at a CAGR of 30%.",
            "url": "https://market-research.com/edge-growth",
        },
        {
            "title": "Edge vs Cloud: The Shift",
            "snippet": "The pendulum swings back to decentralized processing as bandwidth costs rise and privacy concerns mount.",
            "url": "https://cloud-insider.net/edge-vs-cloud",
        },
    ]

    quantum_ml_results = [
        {
            "title": "Quantum Machine Learning Breakthroughs in 2026",
            "snippet": "Hybrid quantum-classical algorithms show 100x speedup for specific optimization problems. Google and IBM lead race.",
            "url": "https://quantum-digest.com/qml-2026",
        },
        {
            "title": "NISQ Era Limitations for ML",
            "snippet": "Current noisy quantum hardware limits ML to toy problems. Error correction at scale expected by 2028.",
            "url": "https://quantum-research.org/nisq-limits",
        },
        {
            "title": "Quantum Feature Maps for Classification",
            "snippet": "Researchers demonstrate quantum kernel methods outperforming classical SVMs on 5 benchmark datasets.",
            "url": "https://arxiv-summary.com/quantum-kernels",
        },
    ]

    ai_regulation_results = [
        {
            "title": "EU AI Act 2026 Updates",
            "snippet": "The EU finalized tiered risk classification. High-risk AI systems now require conformity assessments and audit trails.",
            "url": "https://eu-policy-watch.com/ai-act-2026",
        },
        {
            "title": "US Executive Order on AI Safety",
            "snippet": "New NIST guidelines mandate red-teaming for frontier models. Reporting requirements for models above 10^26 FLOPs.",
            "url": "https://ai-policy-us.gov/executive-order",
        },
        {
            "title": "Global AI Governance Gap",
            "snippet": "While EU and US advance regulation, Asia-Pacific nations diverge. China favors state-led AI governance; India proposes voluntary codes.",
            "url": "https://global-ai-tracker.org/governance-gap",
        },
    ]

    # pick results based on keywords
    q = query.lower()
    if "quantum" in q or "qml" in q:
        results = quantum_ml_results
    elif "regulation" in q or "governance" in q or "policy" in q or "ai act" in q:
        results = ai_regulation_results
    else:
        results = edge_computing_results

    random.shuffle(results)

    formatted = ""
    for i, res in enumerate(results[:3]):
        formatted += f"{i+1}. [{res['title']}]({res['url']}): {res['snippet']}\n"

    return formatted


@tool
def scrape_tool(url: str) -> str:
    """
    Simulates scraping a web page. Returns the text content of the page.
    """
    print(f"  [Tool] Scraping: {url}")

    content_map = {
        "tech-trends-2026": (
            "Full Article: The State of Edge Computing in 2026.\n\n"
            "Edge computing is rapidly evolving. In 2026, we see a massive shift towards processing data locally. "
            "6G networks are starting to appear, providing the low latency needed for real-time edge AI. "
            "Autonomous vehicles and smart cities are the primary beneficiaries. "
            "However, energy consumption of edge nodes is a growing concern. "
            "Market analysts project $150B market size by 2028."
        ),
        "ai-daily": (
            "Full Report: Challenges in Edge AI.\n\n"
            "While promising, running AI at the edge is hard. Models need to be compressed. "
            "Security is a nightmare because devices are physically accessible. "
            "Fragmentation of hardware accelerators (TPUs, NPUs, GPUs) makes standardizing software difficult."
        ),
        "quantum-digest": (
            "Full Article: Quantum ML Breakthroughs.\n\n"
            "2026 marks a turning point for quantum machine learning. Hybrid quantum-classical models "
            "are achieving practical speedups on optimization tasks. Google's 72-qubit processor demonstrated "
            "quantum advantage for combinatorial problems. IBM's Heron chip enables error-corrected circuits."
        ),
        "quantum-research": (
            "Full Report: NISQ Limitations.\n\n"
            "The NISQ (Noisy Intermediate-Scale Quantum) era continues. Current hardware has 50-1000 qubits "
            "but error rates remain 0.1-1%. For real-world ML, we need millions of logical qubits. "
            "Error correction overhead means practical quantum ML is at least 3-5 years away."
        ),
        "eu-policy-watch": (
            "Full Article: EU AI Act 2026.\n\n"
            "The AI Act now enforces a four-tier risk system: Unacceptable, High, Limited, and Minimal risk. "
            "High-risk systems (healthcare, law enforcement) require conformity assessments. "
            "Foundation models must disclose training data summaries. Fines up to €35M or 7% of global revenue."
        ),
        "ai-policy-us": (
            "Full Report: US AI Executive Order.\n\n"
            "The Order mandates NIST to develop AI safety standards within 270 days. "
            "Companies training models using >10^26 FLOPs must report to the government. "
            "Red-teaming is now a requirement for frontier AI models before deployment."
        ),
    }

    for key, content in content_map.items():
        if key in url:
            return content

    return f"Content for {url}: (Simulated content) This page discusses emerging technology trends."


@tool
def keyword_extraction_tool(text: str) -> str:
    """
    Extracts SEO-relevant keywords from text.
    Returns a comma-separated list of keywords.
    """
    print(f"  [Tool] Extracting keywords from text ({len(text)} chars)")

    # Simple heuristic mock - extracts capitalized multi-word phrases
    common_keywords = {
        "edge computing": 5,
        "6G": 3,
        "AI": 8,
        "latency": 4,
        "cloud": 3,
        "quantum": 5,
        "machine learning": 4,
        "regulation": 3,
        "privacy": 3,
        "security": 4,
        "autonomous": 2,
        "IoT": 3,
        "NISQ": 2,
        "EU AI Act": 3,
        "governance": 2,
    }

    found = []
    text_lower = text.lower()
    for kw, _ in sorted(common_keywords.items(), key=lambda x: -x[1]):
        if kw.lower() in text_lower:
            found.append(kw)

    return ", ".join(found[:8]) if found else "technology, innovation, 2026"


@tool
def word_count_tool(text: str) -> str:
    """
    Returns the word count of the given text.
    """
    count = len(text.split())
    print(f"  [Tool] Word count: {count}")
    return f"Word count: {count}"



@tool
def sentiment_analysis_tool(text: str) -> str:
    """
    Analyzes the sentiment of text. Returns a sentiment score from -1.0 (very negative)
    to 1.0 (very positive) along with a breakdown of positive/negative/neutral signals.
    Useful for understanding the tone of research data or reports.
    """
    print(f"  [Tool] Analyzing sentiment ({len(text)} chars)")

    positive_words = [
        "breakthrough", "growth", "advantage", "success", "promising", "innovation",
        "opportunity", "leading", "improve", "benefit", "efficient", "powerful",
        "robust", "optimistic", "accelerate", "transform", "enable",
    ]
    negative_words = [
        "challenge", "risk", "limitation", "concern", "difficult", "failure",
        "threat", "problem", "costly", "noisy", "fragmentation", "gap",
        "nightmare", "error", "decline", "vulnerable", "barrier",
    ]
    neutral_words = [
        "report", "analysis", "data", "system", "model", "current",
        "expected", "projected", "continue", "remain",
    ]

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    neu_count = sum(1 for w in neutral_words if w in text_lower)
    total = max(pos_count + neg_count + neu_count, 1)

    score = round((pos_count - neg_count) / total, 3)
    score = max(-1.0, min(1.0, score))

    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"

    return (
        f"Sentiment Score: {score} ({label})\n"
        f"Positive signals: {pos_count} | Negative signals: {neg_count} | Neutral signals: {neu_count}\n"
        f"Positive words found: {', '.join(w for w in positive_words if w in text_lower)}\n"
        f"Negative words found: {', '.join(w for w in negative_words if w in text_lower)}"
    )


@tool
def readability_score_tool(text: str) -> str:
    """
    Calculates a Flesch-Kincaid readability grade level for the given text.
    Returns the grade level (e.g., 8.2 = 8th-grade reading level) and a qualitative label.
    Useful for ensuring reports are accessible to the target audience.
    """
    print(f"  [Tool] Calculating readability ({len(text)} chars)")

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)

    words = text.split()
    num_words = max(len(words), 1)

    # Approximate syllable count: count vowel groups per word
    def count_syllables(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"")
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)

    num_syllables = sum(count_syllables(w) for w in words)

    # Flesch-Kincaid Grade Level formula
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
    grade = round(max(0, grade), 1)

    if grade <= 6:
        label = "Easy (elementary school)"
    elif grade <= 8:
        label = "Fairly Easy (middle school)"
    elif grade <= 10:
        label = "Standard (high school)"
    elif grade <= 12:
        label = "Fairly Difficult (college prep)"
    elif grade <= 14:
        label = "Difficult (college level)"
    else:
        label = "Very Difficult (graduate level)"

    return (
        f"Flesch-Kincaid Grade Level: {grade}\n"
        f"Readability: {label}\n"
        f"Stats: {num_words} words, {num_sentences} sentences, {num_syllables} syllables\n"
        f"Avg words/sentence: {round(num_words / num_sentences, 1)}\n"
        f"Avg syllables/word: {round(num_syllables / num_words, 2)}"
    )


@tool
def plagiarism_check_tool(draft: str, source_text: str) -> str:
    """
    Checks a draft for potential plagiarism by comparing it against source research text.
    Finds overlapping phrases (5+ words) between draft and source.
    Returns a plagiarism score and list of matching phrases.
    """
    print(f"  [Tool] Checking plagiarism (draft={len(draft)} chars, source={len(source_text)} chars)")

    def get_ngrams(text: str, n: int = 5):
        words = text.lower().split()
        return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

    draft_ngrams = get_ngrams(draft, 5)
    source_ngrams = get_ngrams(source_text, 5)

    overlaps = draft_ngrams & source_ngrams
    total_draft = max(len(draft_ngrams), 1)
    overlap_ratio = round(len(overlaps) / total_draft, 3)

    if overlap_ratio > 0.3:
        verdict = "HIGH — significant overlap detected"
    elif overlap_ratio > 0.1:
        verdict = "MODERATE — some overlap detected"
    else:
        verdict = "LOW — minimal overlap"

    sample_matches = list(overlaps)[:5]

    return (
        f"Plagiarism Score: {overlap_ratio} ({len(overlaps)}/{total_draft} matching 5-grams)\n"
        f"Verdict: {verdict}\n"
        f"Sample matches: {'; '.join(sample_matches) if sample_matches else 'None'}"
    )


@tool
def citation_formatter_tool(text: str) -> str:
    """
    Finds raw URLs in the text and formats them as proper numbered markdown citations.
    Returns the text with URLs replaced by citation markers and a References section appended.
    """
    print(f"  [Tool] Formatting citations ({len(text)} chars)")

    urls = re.findall(r'https?://\S+', text)
    urls = list(dict.fromkeys(urls))  # deduplicate preserving order

    if not urls:
        return text + "\n\n_No URLs found to format as citations._"

    result = text
    references = []
    for i, url in enumerate(urls, 1):
        # Clean trailing punctuation from URL
        clean_url = url.rstrip(".,;:!?)")
        marker = f"[{i}]"
        result = result.replace(clean_url, marker, 1)
        # Generate a title from URL domain
        domain = re.search(r'https?://([^/]+)', clean_url)
        domain_name = domain.group(1) if domain else clean_url
        references.append(f"{marker} {domain_name} — {clean_url}")

    result += "\n\n## References\n" + "\n".join(references)

    return (
        f"Formatted {len(urls)} citation(s).\n\n"
        f"--- Formatted Text ---\n{result}"
    )


@tool
def translation_quality_tool(original: str, translation: str) -> str:
    """
    Evaluates translation quality by comparing the structure and length of the
    original text against its translation. Checks for completeness, sentence alignment,
    and length ratio. Returns a quality score and detailed breakdown.
    """
    print(f"  [Tool] Checking translation quality (original={len(original)} chars, translation={len(translation)} chars)")

    orig_sentences = [s.strip() for s in re.split(r'[.!?]+', original) if s.strip()]
    trans_sentences = [s.strip() for s in re.split(r'[.!?]+', translation) if s.strip()]

    # Length ratio check (translations are typically 1.0-1.3x length of original)
    len_ratio = len(translation) / max(len(original), 1)
    len_score = 1.0 if 0.7 <= len_ratio <= 1.5 else 0.5 if 0.5 <= len_ratio <= 2.0 else 0.2

    # Sentence count alignment
    sent_ratio = len(trans_sentences) / max(len(orig_sentences), 1)
    sent_score = 1.0 if 0.8 <= sent_ratio <= 1.2 else 0.5 if 0.5 <= sent_ratio <= 1.5 else 0.2

    # Check if key numbers from original are preserved in translation
    orig_numbers = set(re.findall(r'\d+\.?\d*', original))
    trans_numbers = set(re.findall(r'\d+\.?\d*', translation))
    preserved = orig_numbers & trans_numbers
    num_score = len(preserved) / max(len(orig_numbers), 1) if orig_numbers else 1.0

    overall = round((len_score + sent_score + num_score) / 3.0, 2)

    return (
        f"Translation Quality Score: {overall}\n"
        f"Length ratio: {round(len_ratio, 2)} (score: {len_score})\n"
        f"Sentence alignment: {len(orig_sentences)} → {len(trans_sentences)} (score: {sent_score})\n"
        f"Numbers preserved: {len(preserved)}/{len(orig_numbers)} (score: {round(num_score, 2)})\n"
        f"Overall verdict: {'GOOD' if overall >= 0.7 else 'NEEDS REVIEW' if overall >= 0.4 else 'POOR'}"
    )


@tool
def headline_generator_tool(topic: str) -> str:
    """
    Generates multiple alternative headline options for a given topic.
    Returns 5 headline variations using different styles (news, question, how-to, listicle, provocative).
    Useful for writers to pick the most engaging title.
    """
    print(f"  [Tool] Generating headlines for: {topic}")

    # Create variations using templates
    topic_short = topic.split(":")[0].strip() if ":" in topic else topic

    headlines = [
        f"Breaking: {topic_short} — What You Need to Know in 2026",
        f"Is {topic_short} the Next Big Thing? An In-Depth Analysis",
        f"How {topic_short} Is Reshaping the Technology Landscape",
        f"5 Key Insights About {topic_short} That Will Surprise You",
        f"The Truth About {topic_short}: Hype vs. Reality",
    ]

    formatted = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(headlines))
    return f"Generated 5 headline options:\n{formatted}"


@tool
def statistics_extractor_tool(text: str) -> str:
    """
    Extracts numeric statistics, percentages, financial figures, and date references from text.
    Returns a structured summary of all quantitative data found.
    Useful for analysts to verify data points.
    """
    print(f"  [Tool] Extracting statistics ({len(text)} chars)")

    # Extract percentages
    percentages = re.findall(r'\d+\.?\d*\s*%', text)
    # Extract dollar amounts
    dollars = re.findall(r'\$\d+[\d,.]*\s*(?:billion|million|B|M|K)?', text, re.IGNORECASE)
    # Extract years
    years = re.findall(r'\b20[2-3]\d\b', text)
    # Extract plain numbers with context (e.g., "72-qubit", "100x")
    numeric_refs = re.findall(r'\d+[\d,.]*\s*(?:x|qubit|FLOP|billion|million|CAGR|days)', text, re.IGNORECASE)
    # Extract ranges
    ranges = re.findall(r'\d+\.?\d*\s*[-–]\s*\d+\.?\d*\s*%?', text)

    result = "Extracted Statistics:\n"
    result += f"  Percentages: {', '.join(percentages) if percentages else 'None found'}\n"
    result += f"  Financial: {', '.join(dollars) if dollars else 'None found'}\n"
    result += f"  Years referenced: {', '.join(sorted(set(years))) if years else 'None found'}\n"
    result += f"  Numeric references: {', '.join(numeric_refs) if numeric_refs else 'None found'}\n"
    result += f"  Ranges: {', '.join(ranges) if ranges else 'None found'}\n"
    result += f"  Total data points found: {len(percentages) + len(dollars) + len(numeric_refs) + len(ranges)}"

    return result


@tool
def text_summarizer_tool(text: str) -> str:
    """
    Performs extractive summarization by scoring and selecting the most important sentences.
    Uses a simple TF-based scoring heuristic to pick the top 3 sentences.
    Useful for quickly distilling key points from long text.
    """
    print(f"  [Tool] Summarizing text ({len(text)} chars)")

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= 3:
        return "Text is already short enough. Summary: " + " ".join(sentences)

    # Simple TF-based scoring
    all_words = text.lower().split()
    word_freq = {}
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
                  "for", "of", "and", "or", "but", "with", "by", "from", "that", "this",
                  "it", "as", "be", "has", "have", "had", "not", "will", "can", "do"}
    for w in all_words:
        w_clean = w.strip(".,!?;:'\"()[]")
        if w_clean and w_clean not in stop_words and len(w_clean) > 2:
            word_freq[w_clean] = word_freq.get(w_clean, 0) + 1

    # Score each sentence
    scored = []
    for i, sent in enumerate(sentences):
        words = sent.lower().split()
        score = sum(word_freq.get(w.strip(".,!?;:'\"()[]"), 0) for w in words)
        # Boost first and early sentences
        position_boost = 1.5 if i == 0 else 1.2 if i <= 2 else 1.0
        score *= position_boost
        scored.append((score, i, sent))

    # Pick top 3 by score, then sort by original position
    top = sorted(scored, key=lambda x: -x[0])[:3]
    top = sorted(top, key=lambda x: x[1])  # restore original order

    summary = " ".join(t[2] for t in top)

    return (
        f"Extractive Summary ({len(sentences)} sentences → 3):\n\n"
        f"{summary}\n\n"
        f"Compression ratio: {round(len(summary) / max(len(text), 1) * 100, 1)}%"
    )
