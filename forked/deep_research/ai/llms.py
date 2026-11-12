"""OpenRouter configuration for the forked discovery loop and the portal."""

import os
from copy import deepcopy
from types import SimpleNamespace

NEMOTRON_FREE = "nvidia/nemotron-3.5-super-vl:free"


def model_id(role):
    return os.getenv(f"ECO_{role.upper()}_MODEL") or NEMOTRON_FREE


def chat_model(role, max_tokens=None):
    if os.getenv("ECO_ENABLE_LIVE") != "1":
        raise RuntimeError("Live model calls are disabled.")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY to enable live research.")

    from langchain_openrouter import ChatOpenRouter

    model = model_id(role)
    is_nemotron_free = model == NEMOTRON_FREE
    if max_tokens is None:
        max_tokens = 4096 if is_nemotron_free else 1800
    return ChatOpenRouter(
        model=model,
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=max_tokens,
        # Nemotron can spend the entire completion budget before returning JSON or text.
        reasoning={"effort": "none"} if is_nemotron_free else None,
        timeout=30_000,
        max_retries=0,
        # The SDK otherwise retries transport errors even with max_retries=0.
        model_kwargs={"retries": None},
        openrouter_provider={"require_parameters": True},
    )


def strict_format(response_format):
    def close_objects(schema):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            for field in schema.get("properties", {}).values():
                close_objects(field)
        if "items" in schema:
            close_objects(schema["items"])

    result = deepcopy(response_format)
    specification = result["json_schema"]
    specification["strict"] = True
    close_objects(specification["schema"])
    return result


def discovery_completion(max_tokens=4096):
    model = chat_model("discovery", max_tokens=max_tokens)

    async def complete(messages, response_format):
        reply = await model.ainvoke(
            messages, response_format=strict_format(response_format)
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)])

    return complete
