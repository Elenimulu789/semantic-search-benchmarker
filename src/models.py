"""Shared data contracts; evaluation labels never enter the search index."""
from typing import Any, Literal
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    parent_doc: str


class SourceSection(BaseModel):
    section_id: str
    parent_doc: str
    headers: list[str]
    start: int
    end: int


class EvidenceSpan(BaseModel):
    doc: str
    quote: str = Field(min_length=1)
    occurrence: int = Field(default=0, ge=0)


class EvalQuery(BaseModel):
    query_id: str
    query: str
    kind: Literal["exact", "conceptual"]
    relevant: list[EvidenceSpan] = Field(min_length=1)
    rationale: str


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int
    method: str
