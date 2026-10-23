"""
Multi-provider LLM configuration and access layer.

Every supported provider exposes an OpenAI-compatible /chat/completions
endpoint, so a single client implementation (the `openai` SDK pointed at a
different base_url) covers all of them.

References for the compatibility endpoints used below:
- Anthropic OpenAI SDK compatibility: https://docs.anthropic.com/en/api/openai-sdk
- Cohere compatibility API: https://docs.cohere.com/docs/compatibility-api
- Google Gemini OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- OpenRouter: https://openrouter.ai/docs/quickstart
"""

import json
import os
import re
import time

from openai import OpenAI

CUSTOM_MODEL_OPTION = "Custom model…"

# Keys are the display names shown in the UI.
PROVIDERS = {
    "OpenAI": {
        "icon": "🟩",
        "base_url": None,  # SDK default (https://api.openai.com/v1)
        "env_keys": ["OPENAI_API_KEY"],
        "env_model": "OPENAI_MODEL",
        "key_prefix_hint": "sk-…",
        "keys_url": "https://platform.openai.com/api-keys",
        "supports_json_mode": True,
        "fallback_models": [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o4-mini",
        ],
    },
    "Anthropic": {
        "icon": "🟪",
        "base_url": "https://api.anthropic.com/v1/",
        "env_keys": ["ANTHROPIC_API_KEY"],
        "env_model": "ANTHROPIC_MODEL",
        "key_prefix_hint": "sk-ant-…",
        "keys_url": "https://console.anthropic.com/settings/keys",
        # The compatibility layer does not implement response_format reliably.
        "supports_json_mode": False,
        "fallback_models": [
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-haiku-4-5",
            "claude-3-7-sonnet-latest",
        ],
    },
    "Google Gemini": {
        "icon": "🟦",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "env_model": "GEMINI_MODEL",
        "key_prefix_hint": "AIza…",
        "keys_url": "https://aistudio.google.com/app/apikey",
        "supports_json_mode": True,
        "fallback_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ],
    },
    "OpenRouter": {
        "icon": "🟧",
        "base_url": "https://openrouter.ai/api/v1",
        "env_keys": ["OPENROUTER_API_KEY"],
        "env_model": "OPENROUTER_MODEL",
        "key_prefix_hint": "sk-or-…",
        "keys_url": "https://openrouter.ai/keys",
        "supports_json_mode": True,
        "fallback_models": [
            "openai/gpt-4.1-mini",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-large",
        ],
    },
    "Cohere": {
        "icon": "🟨",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "env_keys": ["COHERE_API_KEY"],
        "env_model": "COHERE_MODEL",
        "key_prefix_hint": "your Cohere key",
        "keys_url": "https://dashboard.cohere.com/api-keys",
        "supports_json_mode": True,
        "fallback_models": [
            "command-a-03-2025",
            "command-r-plus",
            "command-r",
        ],
    },
    "Mistral AI": {
        "icon": "🟥",
        "base_url": "https://api.mistral.ai/v1",
        "env_keys": ["MISTRAL_API_KEY"],
        "env_model": "MISTRAL_MODEL",
        "key_prefix_hint": "your Mistral key",
        "keys_url": "https://console.mistral.ai/api-keys",
        "supports_json_mode": True,
        "fallback_models": [
            "mistral-large-latest",
            "mistral-small-latest",
            "open-mistral-nemo",
        ],
    },
    "Groq": {
        "icon": "🟫",
        "base_url": "https://api.groq.com/openai/v1",
        "env_keys": ["GROQ_API_KEY"],
        "env_model": "GROQ_MODEL",
        "key_prefix_hint": "gsk_…",
        "keys_url": "https://console.groq.com/keys",
        "supports_json_mode": True,
        "fallback_models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
    },
    "DeepSeek": {
        "icon": "🟦",
        "base_url": "https://api.deepseek.com/v1",
        "env_keys": ["DEEPSEEK_API_KEY"],
        "env_model": "DEEPSEEK_MODEL",
        "key_prefix_hint": "sk-…",
        "keys_url": "https://platform.deepseek.com/api_keys",
        "supports_json_mode": True,
        "fallback_models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "Custom (OpenAI-compatible)": {
        "icon": "⬛",
        "base_url": "",  # supplied by the user
        "env_keys": ["LLM_API_KEY", "OPENAI_API_KEY"],
        "env_model": "LLM_MODEL",
        "key_prefix_hint": "endpoint key",
        "keys_url": "",
        "supports_json_mode": True,
        "requires_base_url": True,
        "fallback_models": [],
    },
}

DEFAULT_PROVIDER = "OpenAI"

# Model ids that are not chat models and should never appear in the dropdown.
_NON_CHAT_PATTERNS = re.compile(
    r"(embed|embedding|whisper|tts|audio|dall-e|image|moderation|rerank|"
    r"guard|vision-encoder|codestral-embed|aqa)",
    re.IGNORECASE,
)


def provider_names():
    return list(PROVIDERS.keys())


def provider_config(provider):
    return PROVIDERS.get(provider, PROVIDERS[DEFAULT_PROVIDER])


def env_api_key(provider):
    """Read a pre-configured key for this provider from the environment/.env."""
    for name in provider_config(provider)["env_keys"]:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""


def env_model(provider):
    name = provider_config(provider).get("env_model")
    value = os.getenv(name) if name else None
    return value.strip() if value and value.strip() else ""


def fallback_models(provider):
    return list(provider_config(provider)["fallback_models"])


def _normalize_model_id(model_id):
    # Gemini's compatibility layer returns ids such as "models/gemini-2.5-flash".
    return model_id.split("/", 1)[1] if model_id.startswith("models/") else model_id


def _build_client(api_key, base_url):
    if not api_key:
        raise RuntimeError("No API key supplied. Add one in the sidebar under LLM Settings.")
    kwargs = {"api_key": api_key, "timeout": 120.0}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def resolve_base_url(provider, custom_base_url=""):
    config = provider_config(provider)
    if config.get("requires_base_url"):
        return (custom_base_url or "").strip()
    return config["base_url"]


def fetch_models(provider, api_key, custom_base_url=""):
    """
    Ask the provider for its model catalogue so the dropdown reflects what the
    key can actually use. Raises on failure so the UI can explain the problem.
    """
    base_url = resolve_base_url(provider, custom_base_url)
    client = _build_client(api_key, base_url)
    models = []
    for item in client.models.list():
        model_id = _normalize_model_id(getattr(item, "id", "") or "")
        if model_id and not _NON_CHAT_PATTERNS.search(model_id):
            models.append(model_id)
    return sorted(set(models))


def test_connection(llm_config):
    """Minimal round-trip used by the sidebar 'Test connection' button."""
    client = _build_client(llm_config["api_key"], llm_config.get("base_url"))
    response = client.chat.completions.create(
        model=llm_config["model"],
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=16,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _extract_json(content):
    """Parse JSON from a model reply, tolerating code fences and prose wrappers."""
    text = (content or "").strip()
    if not text:
        raise ValueError("The model returned an empty response.")
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLMError(RuntimeError):
    """Raised when a model cannot be reached or cannot be coaxed into valid JSON."""


#: Errors worth retrying verbatim: the request was fine, the service was not.
_TRANSIENT_ERROR = re.compile(
    r"(rate.?limit|429|too many requests|timeout|timed out|connection|"
    r"50[0234]|overloaded|temporarily unavailable|service unavailable)",
    re.IGNORECASE,
)

_JSON_REPAIR_PROMPT = (
    "Your previous reply could not be parsed as JSON. Return the same information again as a "
    "single valid JSON object. Output JSON only: no prose, no explanation, no markdown code "
    "fences, no trailing commas."
)


def _is_transient(error) -> bool:
    return bool(_TRANSIENT_ERROR.search(str(error)))


def _create_completion(client, request, use_json_mode):
    """One completion call, negotiating JSON mode down to plain text if rejected."""
    if use_json_mode:
        try:
            return client.chat.completions.create(**request, response_format={"type": "json_object"})
        except Exception as error:
            # Only fall back for endpoints that reject the JSON-mode parameter itself.
            if "response_format" not in str(error) and "json" not in str(error).lower():
                raise
    return client.chat.completions.create(**request)


def chat(llm_config, messages, network_attempts=3):
    """
    Send a message list and return the reply text.

    Transient provider failures (rate limits, timeouts, 5xx) are retried with
    exponential backoff. Everything else is raised immediately, because retrying
    a bad API key or an unknown model id only wastes the user's time.
    """
    provider = llm_config.get("provider", DEFAULT_PROVIDER)
    config = provider_config(provider)
    client = _build_client(llm_config["api_key"], llm_config.get("base_url"))
    model = (llm_config.get("model") or "").strip()
    if not model:
        raise LLMError("No model selected. Pick one in the sidebar under LLM Settings.")

    request = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_config.get("temperature", 0)),
    }

    last_error = None
    for attempt in range(1, max(1, network_attempts) + 1):
        try:
            response = _create_completion(client, request, config.get("supports_json_mode", True))
            return response.choices[0].message.content or ""
        except Exception as error:
            last_error = error
            if attempt >= network_attempts or not _is_transient(error):
                raise
            time.sleep(min(8.0, 1.5 * (2 ** (attempt - 1))))
    raise LLMError(str(last_error))


def chat_json(llm_config, system_prompt, user_prompt):
    """Send a request and return the parsed JSON object. Raises on unparseable replies."""
    content = chat(llm_config, [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return _extract_json(content)


def chat_json_validated(
    llm_config,
    system_prompt,
    user_prompt,
    validator=None,
    max_attempts=3,
):
    """
    Request JSON, validate it, and ask the model to repair it when it is wrong.

    This is the self-repair loop that makes structured output dependable without
    a heavyweight framework. Each failed attempt is pushed back into the
    conversation together with the specific validation errors, so the model is
    told exactly which fields to fix rather than being asked to guess a second
    time. The schema error messages double as the repair instructions.

    ``validator`` takes the parsed payload and returns an object exposing ``ok``,
    ``errors`` and ``warnings`` (see `schemas.ValidationReport`).

    Returns ``{"payload": ..., "diagnostics": {...}}``. If the final attempt
    still fails validation the best available payload is returned with
    ``schema_failed`` set, because the downstream normalizer can repair a
    partially wrong payload and record what it had to fix. An unparseable reply
    on every attempt is a hard failure and raises `LLMError`.
    """
    from schemas import repair_instruction  # local import keeps this module import-light

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    diagnostics = {
        "attempts": [],
        "attempt_count": 0,
        "repaired": False,
        "schema_failed": False,
        "schema_warnings": [],
    }
    best_payload = None
    parse_errors = []

    for attempt in range(1, max(1, max_attempts) + 1):
        content = chat(llm_config, messages)
        diagnostics["attempt_count"] = attempt
        record = {"attempt": attempt, "ok": False, "stage": "parse", "errors": []}

        try:
            payload = _extract_json(content)
        except Exception as error:
            record["errors"] = [f"Reply was not valid JSON: {error}"]
            diagnostics["attempts"].append(record)
            parse_errors.append(str(error))
            if attempt >= max_attempts:
                break
            messages.extend([
                {"role": "assistant", "content": (content or "")[:4000]},
                {"role": "user", "content": _JSON_REPAIR_PROMPT},
            ])
            continue

        best_payload = payload
        if validator is None:
            record.update({"ok": True, "stage": "parsed"})
            diagnostics["attempts"].append(record)
            return {"payload": payload, "diagnostics": diagnostics}

        report = validator(payload)
        record.update({
            "ok": bool(report.ok),
            "stage": "schema",
            "errors": list(report.errors[:10]),
            "warnings": list(report.warnings[:10]),
        })
        diagnostics["attempts"].append(record)

        if report.ok:
            diagnostics["schema_warnings"] = list(report.warnings)
            diagnostics["repaired"] = attempt > 1
            return {"payload": payload, "diagnostics": diagnostics}

        if attempt >= max_attempts:
            diagnostics["schema_warnings"] = list(report.warnings)
            diagnostics["schema_failed"] = True
            diagnostics["unresolved_errors"] = list(report.errors)
            return {"payload": payload, "diagnostics": diagnostics}

        messages.extend([
            {"role": "assistant", "content": (content or "")[:6000]},
            {"role": "user", "content": repair_instruction(report)},
        ])

    if best_payload is not None:
        diagnostics["schema_failed"] = True
        return {"payload": best_payload, "diagnostics": diagnostics}

    raise LLMError(
        f"The model did not return valid JSON after {max_attempts} attempt(s). "
        f"Last parser error: {parse_errors[-1] if parse_errors else 'unknown'}."
    )
