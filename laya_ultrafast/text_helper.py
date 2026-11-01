"""The small text helper. It writes a field value only after TYPE_TEXT was chosen; it never decides anything."""

import json
import os
import ssl
import time

import httpx
import truststore

from .questions import TEXT_VALUE

_CLIENT = None


def client():
    """Created on first use: importing the package must not build a TLS context or touch the network.

    Certificates are verified against the operating system's trust store, so a corporate or antivirus CA the OS
    trusts works too. Verification is never disabled.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(http2=True, timeout=25, verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    return _CLIENT


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = client().post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Text helper connection failed; nothing typed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            raise RuntimeError(f"Text helper returned HTTP {response.status_code}; nothing typed.")
        return response.json()
    raise RuntimeError("Text helper unavailable")


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {
            **{k: action.get(k) for k in ("label", "role", "value")},
            # What the field accepts, from its accessible description ("Enter a date…"), when the page provides one.
            **({"description": action["description"]} if action.get("description") else {}),
        },
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
