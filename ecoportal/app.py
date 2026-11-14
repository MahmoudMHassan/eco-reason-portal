"""FastAPI backend for the local research workbench."""

# ruff: noqa: E402

import asyncio
import os
from contextlib import asynccontextmanager, closing, nullcontext
from json import JSONDecodeError
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

from fastapi import FastAPI, HTTPException
from httpx import TimeoutException
from langchain_core.exceptions import OutputParserException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from openrouter.errors import OpenRouterError
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from .corpus import Corpus, build_index
from .retrieval import HybridSearch
from .reviews import ReviewStore
from .workflow import (
    COMPARISON_GAP,
    EXTRACTION_PAPER_LIMIT,
    LITERATURE_GAP,
    LiveModels,
    build_workflow,
    overview_passages,
    provider_status,
    source_passages,
)


class Turn(BaseModel):
    workspace: UUID = Field(default_factory=uuid4)
    message: str = Field(min_length=1, max_length=2500)
    preset: Literal["crayfish", "enemy_release", "diet_growth"] | None = None
    mode: Literal["chat", "research"] = "research"
    deep_search: bool = False


class Review(BaseModel):
    workspace: UUID
    claim_id: str
    version: int = Field(ge=1)
    action: Literal["accepted", "revised", "rejected"]
    scope: str = Field(default="", max_length=1000)
    reason: str = Field(default="", max_length=1000)


class Portal:
    def __init__(self, corpus, reviews, graph, live):
        self.corpus = corpus
        self.reviews = reviews
        self.graph = graph
        self.live = live

    async def ask(
        self, workspace, message, preset=None, mode="research", deep_search=False
    ):
        if not message.strip() or len(message) > 2500:
            raise ValueError("Enter a question between 1 and 2,500 characters.")
        result = await asyncio.wait_for(
            self.graph.ainvoke(
                {
                    "workspace": workspace,
                    "message": message,
                    "preset": preset,
                    "mode": mode,
                    "deep_search": deep_search,
                },
                {"configurable": {"thread_id": workspace}},
            ),
            timeout=120,
        )
        return await self.turn_response(workspace, result)

    async def recent_turns(self, workspace):
        config = {"configurable": {"thread_id": workspace}}
        turns = []
        async for snapshot in self.graph.aget_state_history(config, limit=60):
            if not snapshot.next and "assessment" in snapshot.values:
                result = snapshot.values
                turns.append(
                    {
                        "message": result["message"],
                        "answer": await self.turn_response(workspace, result),
                    }
                )
                if len(turns) == 10:
                    break
        return list(reversed(turns))

    async def turn_response(self, workspace, result):
        request = result["request"]
        assessment = result["assessment"]
        record = result.get("ecoaire")
        overview = result.get("overview")
        chat_answer = result.get("chat_answer")
        literature_search = request["intent"] == "search"
        if literature_search and assessment["gap"] == COMPARISON_GAP:
            assessment = {**assessment, "gap": LITERATURE_GAP}
        return {
            "request": request,
            "assessment": assessment,
            "overview": overview,
            "chat_answer": chat_answer,
            "evidence": result["evidence"],
            "case": result["case"],
            "causal_result": result.get("causal_result", {"kind": "insufficient"}),
            "ecoaire": {"hypothesis_benchmark_instance": record} if record else None,
            "discovery_fallback": result.get("discovery_fallback", False),
            "retrieval": "qdrant"
            if any("text" in hit for hit in result["hits"])
            else "sqlite",
            "hits": [
                {key: hit[key] for key in ("doi", "title")} for hit in result["hits"]
            ],
            "passages": await asyncio.to_thread(
                overview_passages if literature_search else source_passages,
                self.corpus,
                result["hits"]
                if literature_search
                else result["hits"][:EXTRACTION_PAPER_LIMIT],
                result["request"]["query"],
            )
            if not result["evidence"]
            and not overview
            and not (chat_answer and chat_answer["text"])
            else [],
            "claims": await self.reviews.get(workspace),
            "workspace": workspace,
        }


def create_app(data_dir=None):
    directory = Path(data_dir or os.getenv("ECO_DATA_DIR", "runtime"))

    @asynccontextmanager
    async def lifespan(app):
        directory.mkdir(parents=True, exist_ok=True)
        corpus = Corpus(directory / "corpus.sqlite")
        if not corpus.path.is_file():
            await asyncio.to_thread(build_index, corpus.path)
        reviews = ReviewStore(directory / "workspace.sqlite")
        await reviews.setup()
        models = LiveModels() if os.getenv("ECO_ENABLE_LIVE") == "1" else None
        qdrant_url = os.getenv("ECO_QDRANT_URL")
        connection = (
            closing(QdrantClient(url=qdrant_url, timeout=4))
            if qdrant_url
            else nullcontext()
        )
        with connection as client:
            hybrid = HybridSearch(corpus, client) if client else None
            async with AsyncSqliteSaver.from_conn_string(
                str(directory / "checkpoints.sqlite")
            ) as saver:
                app.state.portal = Portal(
                    corpus,
                    reviews,
                    build_workflow(corpus, reviews, saver, models, hybrid),
                    models is not None,
                )
                yield

    app = FastAPI(title="EcoReason Portal", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        portal = app.state.portal
        return {"mode": "live" if portal.live else "offline", **portal.corpus.info()}

    @app.post("/api/turn")
    async def turn(payload: Turn):
        try:
            return await app.state.portal.ask(
                str(payload.workspace),
                payload.message,
                payload.preset,
                payload.mode,
                deep_search=payload.deep_search,
            )
        except (OutputParserException, JSONDecodeError) as error:
            raise HTTPException(
                502, "A model returned an incomplete structured response."
            ) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except (TimeoutError, TimeoutException) as error:
            raise HTTPException(504, "A research step timed out.") from error
        except OpenRouterError as error:
            status = provider_status(error)
            if status == 429:
                raise HTTPException(
                    429, "OpenRouter or the model provider is rate-limiting this route."
                ) from error
            detail = (
                "OpenRouter replied without a usable research response."
                if status == 200
                else f"OpenRouter returned HTTP {status} for the selected model."
            )
            raise HTTPException(502, detail) from error

    @app.get("/api/workspace/{workspace}")
    async def workspace_turns(workspace: UUID):
        return await app.state.portal.recent_turns(str(workspace))

    @app.post("/api/review")
    async def review(payload: Review):
        try:
            return await app.state.portal.reviews.review(
                str(payload.workspace),
                payload.claim_id,
                payload.version,
                payload.action,
                payload.scope,
                payload.reason,
            )
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    return app


app = create_app()
