from typing import List, TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    task: str
    topic_label: str
    research_data: List[str]
    analysis: str
    draft: str
    critique: str
    revision_count: int
    compliance_notes: str
    compliance_revision_count: int
    seo_keywords: List[str]
    executive_summary: str
    translated_summaries: dict  # {"es": "...", "fr": "..."}
    final_output: str
    iteration_log: List[str]  # tracks which loops were triggered
    messages: Annotated[List[BaseMessage], operator.add]
