"""Estimate one prespecified contrast from Carreira et al. (2017), S1 Table."""

import csv
from pathlib import Path

from .ecoaire import (
    Duration,
    EcoAIRERecord,
    Experiment,
    Hypothesis,
    Intervention,
    Replication,
)
from .models import ResearchRequest

CASE = "diet_growth"
QUESTION = (
    "How does a plant diet versus an animal diet affect juvenile red swamp crayfish "
    "growth over two months at 17°C?"
)
DOI = "10.1371/journal.pone.0183108"
TITLE = (
    "Can heat waves change the trophic role of the world’s most invasive crayfish? "
    "Diet shifts in Procambarus clarkii"
)
SCOPE = "Juvenile Procambarus clarkii, individually housed in the lab at constant 17°C for two months."
METHODS_QUOTE = (
    "We balanced the replicates’ sex ratio and randomly assigned each treatment "
    "combination to ten juveniles (5 Females:5 Males) and seven adult crayfish (4 Females:3 Males)."
)
ARMS = {"Plant-only diet": "Plant", "Animal-only diet": "Animal"}
STUDY = EcoAIRERecord(
    hypothesis=Hypothesis(
        id="carreira-cold-diet-growth",
        name="Diet and juvenile red swamp crayfish growth at 17°C",
        level="operational",
        description=QUESTION,
    ),
    experiment=Experiment(
        id="carreira-juvenile-cold-arms",
        design_type="lab",
        study_type="experimental",
        interventions=[Intervention(treatment="Plant-only diet")],
        controls=["Animal-only diet"],
        replication=Replication(n_replicates=20, unit="individual crayfish"),
        duration=Duration(value=2, unit="months"),
        sampling_protocol="Random assignment with a balanced sex ratio within each diet arm.",
        outcomes_measured=["Mean daily body mass change (mg/day)"],
    ),
    context={
        "ecosystem": "aquatic",
        "climate_conditions": {"temperature_regime": "constant 17°C"},
        "temporal_context": {"life_stage": "juvenile"},
    },
    paper={"id": "carreira-2017", "title": TITLE, "doi": DOI, "year": 2017},
)
SUMMARY = (
    "In this lab comparison, juveniles on the plant diet grew more slowly on average. "
    "The uncertainty interval includes zero."
)


def request():
    return ResearchRequest(
        intent="assess",
        query=STUDY.hypothesis.description,
        exposure=STUDY.experiment.interventions[0].treatment,
        comparator=STUDY.experiment.controls[0],
        outcome=STUDY.experiment.outcomes_measured[0],
        context=SCOPE,
        assumptions=["Compare diets within the cold treatment."],
    )


def study_evidence():
    return {
        "id": "example:diet_growth",
        "doi": DOI,
        "title": TITLE,
        "locator": "Materials and methods, Experimental procedures",
        "quote": METHODS_QUOTE,
        "design": "experiment",
        "context": SCOPE,
        "finding": "The study randomized diet and temperature combinations and balanced sexes.",
    }


def estimate(record: EcoAIRERecord):
    if (
        record.hypothesis.id != STUDY.hypothesis.id
        or record.experiment != STUDY.experiment
        or record.context != STUDY.context
        or record.paper.get("doi") != DOI
    ):
        return None

    import numpy as np
    from statsmodels.regression.linear_model import OLS

    source = (
        Path(__file__).resolve().parents[1] / "examples" / "carreira-cold-growth.csv"
    )
    with source.open(newline="") as file:
        rows = list(csv.DictReader(file))

    treatment = ARMS[record.experiment.interventions[0].treatment]
    control = ARMS[record.experiment.controls[0]]
    plant = np.array([row["diet"] == treatment for row in rows])
    if (
        len(rows) != STUDY.experiment.replication.n_replicates
        or set(row["diet"] for row in rows) != {treatment, control}
        or plant.sum() != 10
        or len({row["animal_id"] for row in rows}) != len(rows)
    ):
        raise ValueError(
            "The registered study requires ten unique observations per diet arm."
        )
    male = np.array([row["sex"] == "M" for row in rows])
    growth = np.array([float(row["growth_mg_day"]) for row in rows])
    fit = OLS(growth, np.column_stack((np.ones(len(rows)), plant, male))).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "kind": "estimated",
        "effect": float(fit.params[1]),
        "interval": fit.conf_int()[1].tolist(),
        "unit": "mg/day",
        "treatment_n": int(plant.sum()),
        "comparator_n": int((~plant).sum()),
        "method": "Sex-adjusted difference in mean growth, with an approximate 95% HC2 t interval.",
        "assumptions": [
            "The individual crayfish is the experimental unit. The interval assumes independent outcomes.",
            "The contrast uses the two complete cold-treatment diet arms from S1 Table.",
            "The estimate applies to the study diets and lab conditions at 17°C.",
        ],
        "data_doi": f"{DOI}.s001",
    }


def record_estimate(record: EcoAIRERecord, result):
    experiment_id = record.experiment.id
    evidence_id = f"{experiment_id}-growth"
    return EcoAIRERecord.model_validate(
        {
            **record.sections(),
            "evidence": {
                "id": evidence_id,
                "type": "experimental",
                "effect_size": {
                    "metric": "Plant-only minus animal-only diet, mean daily growth (mg/day)",
                    "value": result["effect"],
                    "confidence_interval": result["interval"],
                },
                "statistical_support": {
                    "model_type": "Sex-adjusted OLS with HC2 uncertainty"
                },
                "evidence_summary": SUMMARY,
            },
            "intervention_outcome": {
                "action": record.experiment.interventions[0].treatment,
                "observed_effect": [
                    {
                        "outcome": record.experiment.outcomes_measured[0],
                        "direction": "decrease",
                    }
                ],
                "interpretation": SUMMARY,
            },
            "links": {
                "tests_hypothesis": [record.hypothesis.id],
                "uses_design": [experiment_id],
                "produces_evidence": [evidence_id],
                "documents_intervention_outcome": [experiment_id, evidence_id],
                "reported_in_paper": [record.paper["id"]],
            },
        }
    )
