"""Streamlit chat for the EcoReason API."""

import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import streamlit as st

from ecoportal.causal import CASE, QUESTION
from ecoportal.corpus import EXAMPLE_STUDIES, doi_url

API_URL = os.getenv("ECO_API_URL", "http://127.0.0.1:8000")
UI_ASSETS = Path(__file__).with_name("ui_assets")
INTRODUCTION = (
    "Ask an invasion ecology question or test a hypothesis against published evidence."
)


def api(method, path, payload=None, timeout=5):
    response = httpx.request(method, f"{API_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def start_new_chat():
    workspace = str(uuid4())
    st.session_state.workspace = workspace
    st.session_state.turns = []
    st.query_params["workspace"] = workspace


def render_answer(answer, turn_index):
    st.write(answer["assessment"]["summary"])
    if answer.get("discovery_fallback"):
        st.caption(
            "Search continued locally after the planner returned incomplete output."
        )
    result = answer["causal_result"]
    kind = result["kind"]
    request = answer["request"]
    literature_search = request["intent"] == "search"
    overview = answer.get("overview")
    chat_answer = answer.get("chat_answer")
    if kind == "estimated":
        low, high = result["interval"]
        st.markdown(f"**Estimated effect · {result['effect']:+.2f} {result['unit']}**")
        st.caption(
            f"Plant minus animal diet. 95% CI {low:+.2f} to {high:+.2f} {result['unit']}. "
            f"{result['treatment_n']} crayfish on the plant diet and {result['comparator_n']} on the animal diet."
        )
    elif kind == "reported":
        st.caption("Reported study finding")
    elif chat_answer is not None:
        if chat_answer["text"] and not chat_answer["sources"]:
            st.caption("General model answer · no supporting corpus citations")
    elif literature_search:
        st.caption("Literature search")
    else:
        st.caption("Insufficient evidence to estimate this comparison")

    if overview:
        for item in overview["findings"]:
            st.write(item["finding"])
            st.caption("Sources " + ", ".join(item["source_ids"]))

    chat_sources = chat_answer["sources"] if chat_answer else []
    rows = chat_sources or (
        overview["sources"] if overview else answer["evidence"] or answer["passages"]
    )
    if rows:
        candidates = not chat_sources and not overview and not answer["evidence"]
        with st.expander(
            "References"
            if chat_sources
            else "Overview sources"
            if overview
            else "Candidate passages"
            if candidates
            else "Study evidence",
            expanded=bool(literature_search and candidates and chat_answer is None),
        ):
            if candidates:
                method = (
                    "Qdrant hybrid search"
                    if answer.get("retrieval") == "qdrant"
                    else "SQLite keyword search"
                )
                st.caption(method)
            for row in rows:
                title = row["title"]
                if overview or chat_sources:
                    title = f"{row['label']} · {title}"
                st.write(f"**{title}**")
                st.caption(row["locator"])
                st.write(row.get("quote") or row["text"])
                if row["doi"]:
                    st.markdown(f"[Open publication]({doi_url(row['doi'])})")
            if kind == "estimated":
                st.markdown(f"[Open experimental data]({doi_url(result['data_doi'])})")

    with st.expander(
        "Search query and scope"
        if literature_search
        else "Interpreted question and scope"
    ):
        st.write(request["query"])
        for field, label in (
            ("exposure", "Intervention"),
            ("comparator", "Comparison"),
            ("outcome", "Outcome"),
            ("context", "Setting"),
        ):
            if request[field]:
                st.write(f"**{label}** · {request[field]}")
        if kind == "estimated":
            st.write(result["method"])
            for assumption in result["assumptions"]:
                st.write(assumption)
        if gap := answer["assessment"].get("gap"):
            st.write(gap)

    if document := answer.get("ecoaire"):
        with st.expander("EcoAIRE record"):
            st.json(document, expanded=False)
            st.download_button(
                "Download EcoAIRE JSON",
                data=json.dumps(document, ensure_ascii=False, indent=2),
                file_name="ecoaire-hypothesis.json",
                mime="application/json",
                key=f"ecoaire-download-{turn_index}",
                on_click="ignore",
            )


def render_review(claims, workspace):
    if not claims:
        return
    with st.expander("Review study claims"):
        by_id = {row["claim_id"]: row for row in claims}
        selected = st.selectbox(
            "Study claim",
            list(by_id),
            format_func=lambda key: f"{by_id[key]['source']} · {by_id[key]['target']}",
        )
        claim = by_id[selected]
        st.caption(f"{claim['status'].title()} · Version {claim['version']}")
        st.write(claim["scope"])
        if claim["reason"]:
            st.write(claim["reason"])
        with st.form(f"{selected}-{claim['version']}"):
            action = st.selectbox("Decision", ("accepted", "revised", "rejected"))
            scope = st.text_input(
                "Scope for a revision", value=claim["scope"], max_chars=1000
            )
            reason = st.text_input("Reason", max_chars=1000)
            if st.form_submit_button("Save review"):
                try:
                    updated = api(
                        "POST",
                        "/api/review",
                        {
                            "workspace": workspace,
                            "claim_id": selected,
                            "version": claim["version"],
                            "action": action,
                            "scope": scope,
                            "reason": reason,
                        },
                    )
                except httpx.HTTPStatusError as error:
                    st.error(
                        error.response.json().get(
                            "detail", "Could not save the review."
                        )
                    )
                except httpx.RequestError:
                    st.error("Could not reach the research service.")
                else:
                    st.session_state.turns[-1]["answer"]["claims"] = updated
                    st.rerun()


st.set_page_config(
    page_title="EcoReason Portal",
    layout="wide",
    # Keep the native theme menu available in the Docker UI's minimal toolbar.
    menu_items={"About": INTRODUCTION},
)
st.html(UI_ASSETS / "layout.css")
with st.container(
    key="portal_header",
    horizontal=True,
    horizontal_alignment="distribute",
    vertical_alignment="center",
):
    st.title("EcoReason Portal", anchor=False, width="content")
    with st.container(horizontal=True, width="content", vertical_alignment="center"):
        corpus_status = st.empty()
        st.button("New chat", icon=":material/add:", on_click=start_new_chat)
        st.html(
            UI_ASSETS / "theme_toggle.html",
            width="content",
            unsafe_allow_javascript=True,
        )

try:
    health = api("GET", "/healthz")
    if "workspace" not in st.session_state:
        try:
            workspace = str(UUID(st.query_params.get("workspace", "")))
        except ValueError:
            workspace = str(uuid4())
            turns = []
        else:
            turns = api("GET", f"/api/workspace/{workspace}")
        st.session_state.workspace = workspace
        st.session_state.turns = turns
        st.query_params["workspace"] = workspace
except httpx.HTTPError:
    st.error(
        "The research service is unavailable. Check that the API is running and reload."
    )
    st.stop()

workspace = st.session_state.workspace
turns = st.session_state.turns
corpus_status.caption(
    f"{health['mode'].title()} mode · {health['papers']:,} papers in the corpus",
    width="content",
)
if not turns:
    st.write(INTRODUCTION)
with st.container(
    key="research_controls", horizontal=True, vertical_alignment="center"
):
    research_mode = st.toggle("Analyze a hypothesis")
    preset = CASE if st.button("Try a causal analysis", type="primary") else None
    with st.popover("Example questions"):
        for key, label in (
            (CASE, "Crayfish diet and growth"),
            ("crayfish", "Crayfish and lake restoration"),
            ("enemy_release", "Enemy release"),
        ):
            if st.button(label):
                preset = key

for turn_index, turn in enumerate(turns):
    with st.chat_message("user"):
        st.write(turn["message"])
    with st.chat_message("assistant"):
        render_answer(turn["answer"], turn_index)

if turns:
    render_review(turns[-1]["answer"]["claims"], workspace)

message = st.chat_input("Ask an ecology question", key=workspace, max_chars=2500)
if preset:
    message = QUESTION if preset == CASE else EXAMPLE_STUDIES[preset]["question"]
if message:
    with st.chat_message("user"):
        st.write(message)
    try:
        with st.spinner("Preparing your answer…"):
            answer = api(
                "POST",
                "/api/turn",
                {
                    "workspace": workspace,
                    "message": message,
                    "preset": preset,
                    "mode": "research" if research_mode else "chat",
                },
                timeout=130,
            )
    except httpx.HTTPStatusError as error:
        if error.response.status_code in {429, 502, 504}:
            st.error(error.response.json()["detail"])
        else:
            st.error("The research turn could not finish. Try again.")
    except httpx.RequestError:
        st.error("The research turn could not finish. Try again.")
    else:
        turns.append({"message": message, "answer": answer})
        st.rerun()
