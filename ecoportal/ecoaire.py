"""Runtime checks and serialization for the bundled EcoAIRE field template."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from .models import Record, ResearchRequest

TEMPLATE = json.loads(
    (
        Path(__file__).resolve().parents[1] / "forked" / "ecoaire" / "schema.json"
    ).read_text()
)


class Hypothesis(Record):
    id: str
    name: str
    level: Literal["theory", "mechanistic", "operational"]
    description: str
    parent_id: str | None = None
    sub_hypotheses: list[dict] = Field(default_factory=list)
    causal_graph: dict = Field(default_factory=dict)


class Intervention(Record):
    treatment: str = Field(min_length=1)


class Replication(Record):
    n_replicates: int = Field(gt=0)
    unit: str = Field(min_length=1)


class Duration(Record):
    value: FiniteFloat = Field(gt=0)
    unit: Literal["days", "months", "years"]


class Experiment(Record):
    id: str | None = None
    design_type: Literal["field", "lab", "observational"] | None = None
    study_type: Literal["experimental", "observational", "synthesis"] | None = None
    interventions: list[Intervention] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=list)
    replication: Replication | None = None
    duration: Duration | None = None
    sampling_protocol: str | None = None
    outcomes_measured: list[str] = Field(default_factory=list)


class EffectSize(Record):
    metric: str
    value: FiniteFloat
    confidence_interval: tuple[FiniteFloat, FiniteFloat]


class Evidence(Record):
    id: str | None = None
    type: Literal["experimental", "observational", "meta-analysis"] | None = None
    strength: Literal["weak", "moderate", "strong"] | None = None
    direction: Literal["support", "contradict", "mixed"] | None = None
    effect_size: EffectSize | None = None
    statistical_support: dict = Field(default_factory=dict)
    evidence_summary: str | None = None


class Context(Record):
    ecosystem: Literal["grassland", "forest", "aquatic", "wetland", "other"] | None = (
        None
    )
    scale: Literal["plot", "landscape", "regional", "global"] | None = None
    biome: str | None = None
    geographic_region: str | None = None
    stress_level: Literal["low", "moderate", "high", "unknown"] | None = None
    disturbance: str | None = None
    climate_conditions: dict = Field(default_factory=dict)
    temporal_context: dict = Field(default_factory=dict)


class EcoAIRERecord(Record):
    hypothesis: Hypothesis
    causal_model: dict = Field(default_factory=dict)
    experiment: Experiment = Field(default_factory=Experiment)
    evidence: Evidence = Field(default_factory=Evidence)
    context: Context = Field(default_factory=Context)
    intervention_outcome: dict = Field(default_factory=dict)
    paper: dict = Field(default_factory=dict)
    links: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_template_fields(self):
        for section, fields in self.model_dump().items():
            unexpected = fields.keys() - TEMPLATE[section].keys()
            if unexpected:
                raise ValueError(
                    f"Unknown EcoAIRE fields in {section}: {sorted(unexpected)}"
                )
        return self

    def sections(self):
        values = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        return {section: values.get(section, {}) for section in TEMPLATE}

    def search_query(self, query, scope):
        terms = [item.treatment for item in self.experiment.interventions]
        terms.extend(self.experiment.controls)
        terms.extend(self.experiment.outcomes_measured)
        terms.extend(
            [
                scope,
                self.context.ecosystem,
                self.context.climate_conditions.get("temperature_regime"),
            ]
        )
        for term in terms:
            if (
                isinstance(term, str)
                and term
                and term.casefold() not in query.casefold()
            ):
                query += f" {term}"
        return query

    def agent_context(self, scope):
        return {
            "hypothesis": self.hypothesis.description,
            "experiment": self.experiment.model_dump(
                exclude_none=True, exclude_defaults=True
            ),
            "context": self.context.model_dump(
                exclude_none=True, exclude_defaults=True
            ),
            "requested_scope": scope,
        }


def draft_record(request: ResearchRequest, message: str, *, theory=False):
    exposure = (request.exposure or "").strip()
    comparator = (request.comparator or "").strip()
    outcome = (request.outcome or "").strip()
    name = request.query.strip() or message.strip()
    identity = json.dumps([name, exposure, comparator, outcome, request.context])
    return EcoAIRERecord(
        hypothesis=Hypothesis(
            id="h-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
            name=name,
            description=name if request.intent == "follow-up" else message.strip(),
            level="theory" if theory else "operational",
        ),
        experiment=Experiment(
            interventions=[Intervention(treatment=exposure)] if exposure else [],
            controls=[comparator] if comparator else [],
            outcomes_measured=[outcome] if outcome else [],
        ),
    )


def attach_finding(record: EcoAIRERecord, row):
    paper_id = row["doi"] or row["title"]
    paper = {"id": paper_id, "title": row["title"]}
    if row["doi"]:
        paper["doi"] = row["doi"]
    return EcoAIRERecord.model_validate(
        {
            **record.sections(),
            "evidence": {
                "id": row["id"],
                "type": {
                    "experiment": "experimental",
                    "observation": "observational",
                }.get(row["design"]),
                "evidence_summary": row["quote"],
            },
            "paper": paper,
            "links": {
                "produces_evidence": [row["id"]],
                "reported_in_paper": [paper_id],
            },
        }
    )
