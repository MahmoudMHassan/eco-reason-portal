import csv

import httpx
import pytest

from ecoportal.app import create_app
from ecoportal.causal import QUESTION, STUDY, estimate
from ecoportal.corpus import build_index
from ecoportal.ecoaire import TEMPLATE, EcoAIRERecord
from ecoportal.models import Extraction
from ecoportal.workflow import validate_comparisons


@pytest.mark.asyncio
async def test_search_review_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("ECO_ENABLE_LIVE", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-be-used")
    source = tmp_path / "papers.csv"
    with source.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ASK ID", "DOI", "Title", "Abstract", "Full-text"])
        writer.writerow(
            [
                "1",
                "HTTPS://doi.org/10.1234/EXAMPLE",
                "Zebra mussels and phosphorus",
                "Zebra mussels redistribute phosphorus in this synthetic test record.",
                "N/A",
            ]
        )
    build_index(tmp_path / "corpus.sqlite", source)
    app = create_app(tmp_path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            health = (await client.get("/healthz")).json()
            assert health == {"mode": "offline", "snapshot": "Zenodo v3", "papers": 1}
            search = (
                await client.post(
                    "/api/turn", json={"message": "zebra mussel phosphorus"}
                )
            ).json()
            assert search["hits"][0]["doi"] == "10.1234/example"
            assert search["passages"][0]["text"].startswith("Zebra mussels")
            assert search["evidence"] == []
            assert search["claims"] == []
            assert search["ecoaire"] is None
            demo = (
                await client.post(
                    "/api/turn", json={"message": "example", "preset": "crayfish"}
                )
            ).json()
            claim = demo["claims"][0]
            update = {
                "workspace": demo["workspace"],
                "claim_id": claim["claim_id"],
                "version": claim["version"],
                "action": "revised",
                "scope": "Clear-water plots",
                "reason": "Limit transfer beyond the study.",
            }
            revised = await client.post("/api/review", json=update)
            assert revised.status_code == 200
            assert (await client.post("/api/review", json=update)).status_code == 409
    restarted = create_app(tmp_path)
    async with restarted.router.lifespan_context(restarted):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted),
            base_url="http://test",
        ) as client:
            reply = (
                await client.post(
                    "/api/turn",
                    json={
                        "workspace": demo["workspace"],
                        "message": "What about turbid water?",
                    },
                )
            ).json()
            assert reply["request"]["intent"] == "follow-up"
            assert "Turbid" in reply["request"]["context"]
            assert reply["claims"][0]["scope"] == "Clear-water plots"
            assert reply["claims"][0]["version"] == 2
            assert reply["claims"][0]["evidence"] == claim["evidence"]


def test_passage_validation_rejects_fabricated_quotes():
    passage = {
        "id": "p1",
        "doi": "10.1234/test",
        "title": "Test",
        "locator": "Abstract",
        "text": "Plant biomass was lower in access plots.",
    }
    record = {
        "passage_id": "p1",
        "quote": passage["text"],
        "design": "experiment",
        "exposure": None,
        "comparator": None,
        "outcome": "biomass",
        "finding": "Lower biomass",
        "context": None,
    }
    extract = Extraction.model_validate(
        {"comparisons": [record, {**record, "quote": "Imagined quotation"}]}
    )
    assert [item["quote"] for item in validate_comparisons(extract, [passage])] == [
        passage["text"]
    ]


@pytest.mark.asyncio
async def test_experimental_estimate_and_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("ECO_ENABLE_LIVE", "0")
    app = create_app(tmp_path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            answer = (await client.post("/api/turn", json={"message": QUESTION})).json()
            result = answer["causal_result"]
            assert result["kind"] == "estimated"
            assert result["effect"] == pytest.approx(-2.492)
            assert result["interval"] == pytest.approx([-5.29334, 0.30934], abs=1e-5)
            assert result["treatment_n"] == result["comparator_n"] == 10
            document = answer["ecoaire"]
            record = document["hypothesis_benchmark_instance"]
            assert record.keys() == TEMPLATE.keys()
            assert record["experiment"]["controls"] == ["Animal-only diet"]
            assert record["evidence"]["effect_size"]["value"] == result["effect"]
            assert (
                record["evidence"]["effect_size"]["confidence_interval"]
                == result["interval"]
            )
            assert "direction" not in record["evidence"]
            assert record["links"]["produces_evidence"] == [record["evidence"]["id"]]
            workspace = answer["workspace"]
            followup = await client.post(
                "/api/turn",
                json={"workspace": workspace, "message": "Does it hold at 26°C?"},
            )
            assert followup.json()["causal_result"]["kind"] == "insufficient"
            assert followup.json()["request"]["context"] == "26°C"
            assert (
                followup.json()["ecoaire"]["hypothesis_benchmark_instance"]["evidence"]
                == {}
            )
            history = (await client.get(f"/api/workspace/{workspace}")).json()
            assert [turn["message"] for turn in history] == [
                QUESTION,
                "Does it hold at 26°C?",
            ]
            assert history[0]["answer"]["ecoaire"] == document


def test_estimator_rejects_changed_ecoaire_context():
    sections = STUDY.sections()
    sections["context"]["climate_conditions"]["temperature_regime"] = "constant 26°C"
    assert estimate(EcoAIRERecord.model_validate(sections)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intake_intent", "empty_plan"), [("search", False), ("assess", True)]
)
async def test_ecoaire_draft_guides_live_agent_requests(
    tmp_path, monkeypatch, intake_intent, empty_plan
):
    import ecoportal.app as portal_app
    from ecoportal.workflow import completion_result

    class Models:
        free_model = True

        async def complete(self, messages, response_format):
            if response_format["json_schema"]["name"] == "serp_result_summary":
                return completion_result({"learnings": [], "followUpQuestions": []})
            assert response_format["json_schema"]["name"] == "serp_queries"
            assert "Crayfish access" in messages[-1]["content"]
            assert "Submerged plant growth" in messages[-1]["content"]
            reply = completion_result(
                {
                    "queries": [
                        {"query": "crayfish", "researchGoal": "Find the grazing study"}
                    ]
                }
            )
            if empty_plan:
                reply.choices[0].message.content = ""
            return reply

        async def structured(self, role, schema, instruction, data):
            if role == "intake":
                return schema(
                    intent=intake_intent,
                    query="crayfish",
                    exposure="Crayfish access",
                    comparator="Full exclosure",
                    outcome="Submerged plant growth",
                    context="Shallow lakes",
                    assumptions=[],
                )
            assert data["request"]["experiment"]["controls"] == ["Full exclosure"]
            if role == "extraction":
                passage = data["passages"][0]
                return schema(
                    comparisons=[
                        {
                            "passage_id": passage["id"],
                            "quote": passage["text"],
                            "design": "experiment",
                            "exposure": "Crayfish access",
                            "comparator": "Full exclosure",
                            "outcome": "Submerged plant growth",
                            "finding": "Plant growth differed by access treatment.",
                            "context": "Shallow lakes",
                        }
                    ]
                )
            return schema(
                verdict="indirect",
                summary="The enclosure study is relevant.",
                gap="The recorded comparison has a narrower scope.",
                claims=[],
            )

    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.setattr(portal_app, "LiveModels", Models)
    app = create_app(tmp_path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/turn",
                json={"message": "Does crayfish grazing affect plant growth?"},
            )
    assert response.status_code == 200
    answer = response.json()
    assert answer["request"]["intent"] == "assess"
    assert answer["discovery_fallback"] is empty_plan
    assert answer["overview"] is None
    record = answer["ecoaire"]["hypothesis_benchmark_instance"]
    assert record["paper"]["doi"] == "10.1371/journal.pone.0078579"
    assert record["evidence"]["evidence_summary"] == answer["evidence"][0]["quote"]
    assert "effect_size" not in record["evidence"]
    assert answer["causal_result"]["kind"] == "insufficient"


@pytest.mark.asyncio
async def test_forked_openrouter_discovery_sends_strict_schema(monkeypatch):
    import json

    from langchain_openrouter import ChatOpenRouter

    from forked.deep_research.deep_research import generate_serp_queries

    monkeypatch.setenv("ECO_ENABLE_LIVE", "0")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    with pytest.raises(RuntimeError, match="disabled"):
        await generate_serp_queries("Crayfish")

    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.setenv("ECO_DISCOVERY_MODEL", "provider/local-test")
    original = ChatOpenRouter._build_client
    seen = []

    def respond(request):
        seen.append(request)
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-only"
        assert request.extensions["timeout"]["read"] == 30
        body = json.loads(request.content)
        assert body["model"] == "provider/local-test"
        assert body["provider"]["require_parameters"]
        schema = body["response_format"]["json_schema"]
        assert schema["strict"] and schema["schema"]["additionalProperties"] is False
        assert (
            schema["schema"]["properties"]["queries"]["items"]["additionalProperties"]
            is False
        )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "system_fingerprint": None,
                "model": body["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "queries": [
                                        {
                                            "query": "crayfish diet",
                                            "researchGoal": "Identify a study",
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:

        def build_client(self):
            client = original(self)
            client.sdk_configuration.async_client = transport
            return client

        monkeypatch.setattr(ChatOpenRouter, "_build_client", build_client)
        assert (await generate_serp_queries("Crayfish"))[0]["query"] == "crayfish diet"
        assert len(seen) == 1

        def timeout(request):
            seen.append(request)
            raise httpx.ReadTimeout("test timeout", request=request)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(timeout)
        ) as transport:
            with pytest.raises(httpx.ReadTimeout):
                await generate_serp_queries("Crayfish")
        assert len(seen) == 2


@pytest.mark.asyncio
async def test_overview_shows_passages_when_model_output_is_invalid(
    tmp_path, monkeypatch
):
    from langchain_core.exceptions import OutputParserException

    import ecoportal.app as portal_app

    question = "Does crayfish grazing reduce submerged plant biomass?"
    calls = []

    class FreeModels:
        free_model = True

        async def structured(self, role, schema, instruction, data):
            calls.append(role)
            raise OutputParserException("Incomplete response")

    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.setattr(portal_app, "LiveModels", FreeModels)
    app = create_app(tmp_path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/turn", json={"message": question})
    assert response.status_code == 200
    assert response.json()["request"]["query"] == question
    assert response.json()["overview"] is None
    assert response.json()["hits"][0]["doi"] == "10.1371/journal.pone.0078579"
    assert response.json()["passages"]
    assert "incomplete overview" in response.json()["assessment"]["summary"]
    assert "literature overview" in response.json()["assessment"]["gap"]
    assert calls == ["intake", "extraction"]


@pytest.mark.asyncio
async def test_live_overview_uses_validated_sources_and_survives_reload(
    tmp_path, monkeypatch
):
    import ecoportal.app as portal_app
    from ecoportal.corpus import EXAMPLE_STUDIES
    from ecoportal.models import LiteratureOverview

    question = "What are the key mechanisms in invasion ecology?"
    paper = EXAMPLE_STUDIES["enemy_release"]
    finding = "The review distinguishes counts of natural enemies from effects on invader populations."
    calls = []

    class Models:
        async def structured(self, role, schema, instruction, data):
            calls.append(role)
            if role == "intake":
                return schema(
                    intent="search",
                    query="enemy release",
                    exposure=None,
                    comparator=None,
                    outcome=None,
                    context=None,
                    assumptions=[],
                )
            assert role == "extraction"
            assert schema is LiteratureOverview
            assert data == {
                "question": question,
                "sources": {"S1": {"title": paper["title"], "text": paper["passage"]}},
            }
            return schema(
                findings=[
                    {"finding": finding, "source_ids": ["S1", "S1"]},
                    {
                        "finding": "This finding has an invented reference.",
                        "source_ids": ["S1", "S9"],
                    },
                ]
            )

    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.setattr(portal_app, "LiveModels", Models)
    app = create_app(tmp_path)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            answer = (await client.post("/api/turn", json={"message": question})).json()
            overview = answer["overview"]
            assert overview["findings"] == [{"finding": finding, "source_ids": ["S1"]}]
            assert overview["sources"][0]["doi"] == paper["doi"]
            assert overview["sources"][0]["text"] == paper["passage"]
            assert answer["ecoaire"] is None and answer["evidence"] == []
            assert answer["claims"] == []
            workspace = answer["workspace"]
            history = (await client.get(f"/api/workspace/{workspace}")).json()
            assert history[0]["answer"]["overview"] == overview
            demo = await client.post(
                "/api/turn", json={"workspace": workspace, "message": QUESTION}
            )
            assert demo.json()["overview"] is None
            assert demo.json()["ecoaire"]
    assert calls == ["intake", "extraction"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limited_step", ["intake_provider", "intake_account", "intake_default", "overview"]
)
async def test_research_turn_preserves_results_under_rate_limiting(
    tmp_path, monkeypatch, limited_step
):
    import json

    from langchain_openrouter import ChatOpenRouter

    from forked.deep_research.ai.llms import NEMOTRON_FREE

    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.delenv("ECO_INTAKE_MODEL", raising=False)
    monkeypatch.delenv("ECO_EXTRACTION_MODEL", raising=False)
    intake_model = NEMOTRON_FREE
    if limited_step == "intake_provider":
        intake_model = "provider/alternate:free"
        monkeypatch.setenv("ECO_INTAKE_MODEL", intake_model)
    monkeypatch.delenv("ECO_QDRANT_URL", raising=False)
    original = ChatOpenRouter._build_client
    question = "What are the key mechanisms in invasion ecology?"
    calls = []

    def respond(request):
        payload = json.loads(request.content)
        model = payload["model"]
        schema = payload["response_format"]["json_schema"]["name"]
        calls.append((model, schema))
        if model == NEMOTRON_FREE:
            assert payload["reasoning"]["effort"] == "none"
            assert payload["max_tokens"] == 4096
        if (
            len(calls) == 1
            and limited_step.startswith("intake")
            or schema == "LiteratureOverview"
            and limited_step == "overview"
        ):
            error = {"code": 429, "message": "Rate limited"}
            if limited_step != "intake_account":
                error["metadata"] = {"provider_name": "Example provider"}
            return httpx.Response(429, json={"error": error})
        if schema == "ResearchRequest":
            content = {
                "intent": "search",
                "query": question,
                "exposure": None,
                "comparator": None,
                "outcome": None,
                "context": None,
                "assumptions": [],
            }
        else:
            content = {
                "findings": [{"finding": "Mechanisms vary.", "source_ids": ["S1"]}]
            }
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "system_fingerprint": None,
                "model": model,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content),
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:

        def build_client(self):
            client = original(self)
            client.sdk_configuration.async_client = transport
            return client

        monkeypatch.setattr(ChatOpenRouter, "_build_client", build_client)
        app = create_app(tmp_path)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            response = await client.post(
                "/api/turn", json={"message": question, "mode": "research"}
            )
            assert response.status_code == 200
            answer = response.json()
            history = await client.get(f"/api/workspace/{answer['workspace']}")
            assert history.json()[0]["answer"] == answer
    expected = [(intake_model, "ResearchRequest")]
    if limited_step == "intake_provider":
        expected.append((NEMOTRON_FREE, "ResearchRequest"))
        assert answer["overview"]["findings"][0]["source_ids"] == ["S1"]
    else:
        assert answer["passages"]
        assert "rate-limited" in answer["assessment"]["summary"]
    if limited_step in {"intake_provider", "overview"}:
        expected.append((NEMOTRON_FREE, "LiteratureOverview"))
    assert calls == expected


@pytest.mark.asyncio
async def test_plain_chat_answers_without_sources_and_retains_followup_context(
    tmp_path, monkeypatch
):
    import json

    from langchain_openrouter import ChatOpenRouter

    monkeypatch.setenv("ECO_ENABLE_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setenv("ECO_EXTRACTION_MODEL", "nvidia/nemotron-3.5-super-vl:free")
    monkeypatch.delenv("ECO_QDRANT_URL", raising=False)
    original = ChatOpenRouter._build_client
    question = "How to identify critical habitat for Peary caribou?"
    response_text = "Start with the recovery strategy and its habitat maps."
    requests = []

    def respond(request):
        assert request.headers["Authorization"] == "Bearer test-only"
        payload = json.loads(request.content)
        assert "response_format" not in payload
        assert payload["reasoning"]["effort"] == "none"
        assert payload["max_tokens"] == 4096
        messages = payload["messages"]
        requests.append(messages)
        if len(requests) in (3, 4):
            status = 429 if len(requests) == 3 else 200
            return httpx.Response(
                status, json={"error": {"code": 429, "message": "limited"}}
            )
        # A useful answer can arrive without the SDK's expected fingerprint metadata.
        return httpx.Response(
            200,
            json={
                "id": "test",
                "created": 1,
                "model": payload["model"],
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if len(requests) == 5 else "stop",
                        "message": {
                            "role": "assistant",
                            "content": None
                            if len(requests) == 5
                            else response_text + " [S999]",
                        },
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:

        def build_client(self):
            client = original(self)
            client.sdk_configuration.async_client = transport
            return client

        monkeypatch.setattr(ChatOpenRouter, "_build_client", build_client)
        app = create_app(tmp_path)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            answer = (
                await client.post(
                    "/api/turn", json={"message": question, "mode": "chat"}
                )
            ).json()
            assert answer["chat_answer"] == {"text": response_text, "sources": []}
            workspace = answer["workspace"]
            for message in ("What about sea ice?", "And on land?", "And in winter?"):
                reply = await client.post(
                    "/api/turn",
                    json={"workspace": workspace, "message": message, "mode": "chat"},
                )
                assert reply.status_code == 200
            assert "rate-limiting" in reply.json()["assessment"]["summary"]
            exhausted = await client.post(
                "/api/turn",
                json={
                    "workspace": workspace,
                    "message": "Try another example.",
                    "mode": "chat",
                },
            )
            assert "output limit" in exhausted.json()["assessment"]["summary"]
            assert len(requests) == 5
            assert requests[1][1]["content"] == question
            assert requests[1][2]["content"] == response_text


@pytest.mark.asyncio
async def test_upstream_discovery_reuses_local_provider_in_recursive_rounds(tmp_path):
    from ecoportal.corpus import Corpus
    from ecoportal.workflow import LocalSearch, completion_result
    from forked.deep_research.deep_research import deep_research

    build_index(tmp_path / "corpus.sqlite")
    search = LocalSearch(Corpus(tmp_path / "corpus.sqlite"))
    queries = iter(["crayfish", "enemy release"])
    calls = []

    async def complete(messages, response_format):
        schema = response_format["json_schema"]["name"]
        calls.append(schema)
        if schema == "serp_queries":
            return completion_result(
                {"queries": [{"query": next(queries), "researchGoal": "Test"}]}
            )
        return completion_result(
            {"learnings": [], "followUpQuestions": ["Check another mechanism"]}
        )

    await deep_research(
        "invasion ecology", breadth=1, depth=2, search=search, completion=complete
    )
    assert calls == ["serp_queries", "serp_result_summary"] * 2
    assert set(search.hits) == {
        "10.1371/journal.pone.0078579",
        "10.1007/s10526-011-9349-7",
    }
