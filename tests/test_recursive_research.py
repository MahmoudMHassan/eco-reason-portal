import asyncio
import csv
from collections import Counter

import httpx
import pytest
from openrouter.errors import OpenRouterError

from ecoportal import app, workflow
from ecoportal.corpus import build_index

# Prompts from data/ecology/49-questions-ecology.csv in SciKnowOrg DeepResearch.
PHRAGMITES = "What is the best way to manage a 1 hectare stand of Phragmites Australis in a near-urban area?"
BAMBOO = "What determines the invasive potential of Phyllostachis bambusoides?"
MECHANISMS = "What are the key mechanisms in invasion ecology?"
POLLINATORS = "How do invasive plant and animal species disrupt pollination networks and affect native pollinator communities in tropical mountains?"
MOWING = "When could mowing be more efficient than grazing in restoring plant diversity, and why?"

DOI = "10.1234/"
INITIAL = ("alpha ecology", "beta ecology")
FOLLOW_UP = ("Compare field designs", "Check the measured outcome")
QUERIES = {
    INITIAL[0]: ("a1", "a2", "a3", "a4", "a5", "a6"),
    INITIAL[1]: ("b1", "b2", "b3", "b4", "b5", "b6"),
    "gamma comparison": ("later", "a2"),
    "delta evidence": ("result", "b2"),
}


def model_rate_limit():
    return OpenRouterError(
        "Rate limited", httpx.Response(429, json={"error": {"code": 429}})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "mode", "deep_search", "scenario"),
    [
        pytest.param(MOWING, "research", False, "recursive", id="48-mowing-recursion"),
        pytest.param(PHRAGMITES, "chat", True, "recursive", id="02-reed-deep-chat"),
        pytest.param(BAMBOO, "chat", False, "direct", id="05-bamboo-direct-chat"),
        pytest.param(MECHANISMS, "research", False, "overview", id="13-overview"),
        pytest.param(
            POLLINATORS, "chat", True, "branch-rate", id="16-branch-rate-limit"
        ),
        pytest.param(PHRAGMITES, "chat", True, "root-rate", id="02-root-rate-limit"),
        pytest.param(PHRAGMITES, "chat", True, "empty-plan", id="02-empty-plan"),
        pytest.param(
            PHRAGMITES, "chat", True, "malformed-plan", id="02-malformed-plan"
        ),
        pytest.param(PHRAGMITES, "chat", True, "timeout", id="02-discovery-timeout"),
        pytest.param(
            MOWING, "research", False, "invalid-quotes", id="48-invalid-quotes"
        ),
        pytest.param(
            MOWING, "research", False, "extract-rate", id="48-extraction-rate-limit"
        ),
        pytest.param(
            MOWING, "research", False, "assess-rate", id="48-assessment-rate-limit"
        ),
    ],
)
async def test_benchmark_question_routing(
    tmp_path, monkeypatch, question, mode, deep_search, scenario
):
    source = tmp_path / "papers.csv"
    papers = {"direct", *(name for hits in QUERIES.values() for name in hits)}
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ASK ID", "DOI", "Title", "Abstract", "Full-text"])
        for name in sorted(papers):
            writer.writerow(
                [
                    name,
                    DOI + name,
                    f"Paper {name}",
                    f"Source passage for {name}.",
                    "N/A",
                ]
            )
    build_index(tmp_path / "corpus.sqlite", source)
    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.delenv("ECO_QDRANT_URL", raising=False)
    calls = Counter()
    searches = []
    cancelled = []
    given_passages = []

    def search(_corpus, query, limit):
        searches.append(query)
        return [
            {
                "paper_id": DOI + name,
                "doi": DOI + name,
                "title": f"Paper {name}",
                "abstract": f"Source passage for {name}.",
            }
            for name in QUERIES.get(query, ("direct",))[:limit]
        ]

    monkeypatch.setattr(app.Corpus, "search", search)
    if scenario == "timeout":
        monkeypatch.setattr(workflow, "BOUNDED_DISCOVERY_TIMEOUT", 0.25)

    class Models:
        free_model = True

        async def complete(self, messages, response_format):
            schema = response_format["json_schema"]["name"]
            prompt = messages[-1]["content"]
            if schema == "serp_queries" and "Previous research goal:" not in prompt:
                calls["initial plan"] += 1
                assert "maximum of 2" in prompt
                if scenario == "root-rate":
                    raise model_rate_limit()
                if scenario == "empty-plan":
                    return workflow.completion_result({"queries": []})
                if scenario == "malformed-plan":
                    return workflow.completion_result({"queries": [{"query": ""}]})
                return workflow.completion_result(
                    {
                        "queries": [
                            {"query": query, "researchGoal": query} for query in INITIAL
                        ]
                    }
                )
            if schema == "serp_result_summary":
                calls["summary"] += 1
                is_first = INITIAL[0] in prompt
                if is_first and scenario == "branch-rate":
                    raise model_rate_limit()
                if is_first and scenario == "timeout":
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.append(True)
                follow_up = FOLLOW_UP[0] if is_first else FOLLOW_UP[1]
                return workflow.completion_result(
                    {
                        "learnings": ["A candidate study"],
                        "followUpQuestions": [follow_up],
                    }
                )
            calls["follow-up plan"] += 1
            assert "maximum of 1" in prompt
            query = "gamma comparison" if FOLLOW_UP[0] in prompt else "delta evidence"
            return workflow.completion_result(
                {"queries": [{"query": query, "researchGoal": query}]}
            )

        async def structured(self, role, schema, _instruction, data):
            calls[schema.__name__] += 1
            if role == "intake":
                assert data["message"] == question
                if scenario == "overview":
                    return schema(
                        intent="search",
                        query=question,
                        exposure=None,
                        comparator=None,
                        outcome=None,
                        context=None,
                        assumptions=[],
                    )
                return schema(
                    intent="assess",
                    query=question,
                    exposure="Mowing",
                    comparator="Grazing",
                    outcome="Plant diversity",
                    context=None,
                    assumptions=[],
                )
            if schema.__name__ == "LiteratureOverview":
                return schema(
                    findings=[{"finding": "A cited overview.", "source_ids": ["S1"]}]
                )
            if role == "extraction":
                if scenario == "extract-rate":
                    raise model_rate_limit()
                given_passages.extend(data["passages"])
                later = next(
                    row for row in data["passages"] if row["doi"] == DOI + "later"
                )
                comparison = {
                    "passage_id": later["id"],
                    "design": "experiment",
                    "exposure": "Mowing",
                    "comparator": "Grazing",
                    "outcome": "Plant diversity",
                    "context": None,
                    "finding": "Plant diversity was measured.",
                }
                comparisons = [{**comparison, "quote": "This quotation was invented."}]
                if scenario != "invalid-quotes":
                    comparisons.append({**comparison, "quote": later["text"]})
                    comparisons.append({**comparison, "quote": later["text"]})
                return schema(comparisons=comparisons)
            assert role == "assessment"
            if scenario == "assess-rate":
                raise model_rate_limit()
            assert [row["doi"] for row in data["evidence"]] == [DOI + "later"]
            return schema(
                verdict="indirect", summary="A comparison was found.", gap="", claims=[]
            )

        async def chat(self, text, history, sources):
            calls["chat"] += 1
            assert text == question
            assert not history
            return "A cited reply [S1] [S999]"

    monkeypatch.setattr(app, "LiveModels", Models)
    portal = app.create_app(tmp_path)
    async with portal.router.lifespan_context(portal):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=portal), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/turn",
                json={"message": question, "mode": mode, "deep_search": deep_search},
            )
            assert response.status_code == 200, response.text
            answer = response.json()
            saved = await client.get(f"/api/workspace/{answer['workspace']}")
            assert saved.json()[0]["answer"] == answer

    failed_plan = scenario in {"root-rate", "empty-plan", "malformed-plan"}
    used_recursion = not failed_plan and scenario not in {"direct", "overview"}
    if used_recursion:
        assert calls["initial plan"] == 1
        assert calls["summary"] == 2
        assert calls["follow-up plan"] == (
            1 if scenario in {"branch-rate", "timeout"} else 2
        )
        assert "delta evidence" in searches
        assert {row["doi"] for row in answer["hits"]} >= {DOI + "a2", DOI + "result"}
    else:
        assert calls["initial plan"] == int(failed_plan)
        assert searches == [question]

    assert answer["discovery_fallback"] is (
        failed_plan or scenario in {"branch-rate", "timeout"}
    )
    if scenario == "timeout":
        assert cancelled == [True]
    assert len(answer["hits"]) == len({hit["doi"] for hit in answer["hits"]})
    if mode == "chat":
        assert calls["chat"] == 1
        assert (
            calls["ResearchRequest"] == calls["Extraction"] == calls["Assessment"] == 0
        )
        assert answer["chat_answer"]["sources"][0]["doi"] == answer["hits"][0]["doi"]
        assert "[S999]" not in answer["chat_answer"]["text"]
    elif scenario == "overview":
        assert calls["LiteratureOverview"] == 1
        assert calls["Extraction"] == calls["Assessment"] == 0
        assert answer["overview"]["sources"][0]["doi"] == DOI + "direct"
    else:
        assert calls["Extraction"] == 1
        assert len(given_passages) == (0 if scenario == "extract-rate" else 5)
        assert answer["causal_result"]["kind"] == "insufficient"
        if scenario in {"invalid-quotes", "extract-rate"}:
            assert calls["Assessment"] == 0
            assert not answer["evidence"]
            assert any(row["doi"] == DOI + "later" for row in answer["passages"])
        else:
            assert calls["Assessment"] == 1
            assert [row["doi"] for row in answer["evidence"]] == [DOI + "later"]
            if scenario == "assess-rate":
                assert (
                    "assessment model could not finish"
                    in answer["assessment"]["summary"]
                )
            else:
                assert (
                    answer["ecoaire"]["hypothesis_benchmark_instance"]["paper"]["doi"]
                    == DOI + "later"
                )
