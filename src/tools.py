import random
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
