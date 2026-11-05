"""Request and response formats for the API."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from src.config import load_anchors

AnchorId = StrEnum("AnchorId", {anchor: anchor for anchor in load_anchors()})
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=500)]


class QuestionRequest(BaseModel):
    anchor_id: AnchorId = Field(description="Which anchor paper's corpus to search.")
    question: Question = Field(description="The question to answer.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "anchor_id": "nested_learning",
                "question": "How have later papers used test-time memorization?",
            }
        }
    }


class Source(BaseModel):
    title: str
    authors: str | None = None
    year: int | None = None


class PredictResponse(BaseModel):
    anchor_id: str
    answer: str
    sources: list[Source]
    model: str
    setup: str
    latency_ms: int


class Passage(BaseModel):
    rank: int
    text: str
    paper_id: str
    title: str
    authors: str | None = None
    year: int | None = None
    score: float | None = None


class ExplainResponse(BaseModel):
    anchor_id: str
    question: str
    passages: list[Passage]
    setup: str
