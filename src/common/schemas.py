from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Criterion = Literal[
    "factual_grounding",
    "structural_coherence",
    "depth_of_analysis",
    "citation_integrity",
    "absence_of_filler",
]


class Finding(BaseModel):
    id: str = Field(description="Stable ID, F001 / F002 / ... Cited inline in the draft.")
    claim: str = Field(description="One factual assertion. One sentence. No hedging prose.")
    source_url: str = Field(description="URL the claim came from. Empty string if unattributable.")
    confidence: Confidence = Field(
        description=(
            "high: stated directly by a credible source. "
            "medium: implied or from a weaker source. "
            "low: inferred, contested, or single-source."
        )
    )


class Issue(BaseModel):
    span: str = Field(
        description="Verbatim substring of the draft. Must match exactly — verified in code."
    )
    criterion: Criterion
    problem: str
    fix: str


class Critique(BaseModel):
    scores: dict[str, int] = Field(description="Criterion name -> 1..5")
    verdict: Literal["pass", "revise"]
    target: Literal["writer", "analyst", "researcher"]
    issues: list[Issue] = Field(default_factory=list)

    def min_score(self) -> int:
        return min(self.scores.values()) if self.scores else 0

    def grounded_issues(self, draft: str) -> list[Issue]:
        return [i for i in self.issues if i.span and i.span in draft]


class FindingList(BaseModel):
    findings: list[Finding]
