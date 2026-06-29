import operator
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field

# 1. Pydantic models for structured LLM responses
class SearchPlan(BaseModel):
    queries: List[str] = Field(description="3 to 5 highly targeted search queries")

class Critique(BaseModel):
    is_sufficient: bool = Field(description="True if data covers all aspects of the topic")
    missing_angles: List[str] = Field(description="What specific sub-topics are we still missing?")
    feedback: str

# 2. The Graph State 
class ResearchState(TypedDict):
    topic: str
    search_queries: List[str]
    # operator.add is a Reducer: it tells LangGraph to APPEND new scrapes, not overwrite the old ones
    scraped_content: Annotated[List[str], operator.add] 
    critique_count: int
    max_revisions: int
    is_satisfactory: bool
    final_brief: str
