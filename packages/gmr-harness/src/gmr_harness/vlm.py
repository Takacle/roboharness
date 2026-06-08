"""VLM (Vision Language Model) client for alignment analysis.

Provides the API call layer, decoupled from GMR / MuJoCo logic.
Guards optional dependencies via ``_deps.require``.
"""

from __future__ import annotations

import json
from typing import Any


def _require_vlm() -> Any:
    from gmr_harness._deps import require

    return require("openai", "VLM-based alignment agent")


def create_client(api_key: str = "", api_base: str = "") -> tuple[Any, str]:
    """Create an OpenAI-compatible VLM client.

    Returns (client, client_type).
    """
    import os

    openai_mod = _require_vlm()
    import httpx

    base_url = (api_base or "https://open.bigmodel.cn/api/paas/v4").rstrip("/")
    key = api_key or os.environ.get("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))

    client = openai_mod.OpenAI(
        api_key=key or "sk-placeholder",
        base_url=base_url,
        http_client=httpx.Client(trust_env=False),
    )
    return client, "openai"


def ask_vision_model(
    client: Any,
    model: str,
    image_paths: list[Any],
    config: dict,
    iteration: int = 1,
    client_type: str = "openai",
    deviation_text: str | None = None,
    pos_deviation_text: str | None = None,
    tune_mode: str = "scale",
    system_prompt: str = "",
) -> dict:
    """Send images + config to VLM and return parsed response.

    Returns ``{"verdict": str, "analysis": str, "patch": dict}``.
    """
    if client_type == "openai":
        return _ask_openai(
            client,
            model,
            image_paths,
            config,
            iteration,
            deviation_text,
            pos_deviation_text,
            tune_mode,
            system_prompt,
        )
    raise ValueError(f"Unsupported client_type: {client_type!r}")


def _ask_openai(
    client: Any,
    model: str,
    image_paths: list[Any],
    config: dict,
    iteration: int,
    deviation_text: str | None,
    pos_deviation_text: str | None,
    tune_mode: str,
    system_prompt: str,
) -> dict:
    from pathlib import Path

    from gmr_harness._utils import encode_image_base64

    content_items: list[dict] = []
    for p in image_paths[:6]:
        path = Path(str(p))
        if path.exists():
            b64 = encode_image_base64(path)
            ext = path.suffix.lstrip(".") or "png"
            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                }
            )

    text_parts = [
        f"Iteration: {iteration}",
        f"Tune mode: {tune_mode}",
        f"Current config:\n```json\n{json.dumps(config, indent=2)[:4000]}\n```",
    ]
    if deviation_text:
        text_parts.append(f"Position deviation report:\n{deviation_text}")
    if pos_deviation_text:
        text_parts.append(pos_deviation_text)

    content_items.insert(0, {"type": "text", "text": "\n\n".join(text_parts)})

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or "You are a robotics alignment expert."},
            {"role": "user", "content": content_items},
        ],
        max_tokens=2048,
    )
    raw = resp.choices[0].message.content or ""
    return _parse_vlm_response(raw)


def _parse_vlm_response(raw: str) -> dict:
    """Extract JSON patch from VLM response text."""
    if "```json" in raw:
        start = raw.index("```json") + 7
        end = raw.index("```", start)
        raw = raw[start:end]
    elif "```" in raw:
        start = raw.index("```") + 3
        end = raw.index("```", start)
        raw = raw[start:end]

    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"verdict": "needs_fix", "analysis": raw, "patch": {}}

    verdict = "ok" if parsed.get("verdict") == "ok" else "needs_fix"
    return {
        "verdict": verdict,
        "analysis": parsed.get("analysis", ""),
        "patch": parsed.get("patch", {}),
    }
