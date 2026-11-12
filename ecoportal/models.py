from typing import Literal

from pydantic import BaseModel, ConfigDict


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchRequest(Record):
    intent: Literal["search", "assess", "follow-up", "scenario"]
    query: str
    exposure: str | None
    comparator: str | None
    outcome: str | None
    context: str | None
    assumptions: list[str]


class StudyComparison(Record):
    passage_id: str
    quote: str
    design: Literal[
        "experiment", "observation", "synthesis", "hypothesized mechanism", "unclear"
    ]
    exposure: str | None
    comparator: str | None
    outcome: str | None
    finding: str
    context: str | None


class Extraction(Record):
    comparisons: list[StudyComparison]


class OverviewFinding(Record):
    finding: str
    source_ids: list[str]


class LiteratureOverview(Record):
    findings: list[OverviewFinding]


class Claim(Record):
    source: str
    target: str
    direction: Literal["increase", "decrease", "mixed", "unknown"]
    scope: str
    relation: Literal["experimental contrast", "association", "proposed mechanism"]
    evidence_ids: list[str]


class Assessment(Record):
    verdict: Literal["direct", "indirect", "mixed", "insufficient"]
    summary: str
    gap: str
    claims: list[Claim]
