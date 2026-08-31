"""
OpenRouter-based LLM reviewer for scripts/discover_new.py.

This script reads one candidate-review JSON object from stdin and returns one
JSON object to stdout. It is intended to be used through:

python scripts/discover_new.py \
  --llm-review-mode ambiguous \
  --llm-review-command "python scripts/llm_reviewer_openrouter.py"
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"


REQUIRED_FIELDS = {
    "has_verified_model_link": False,
    "has_verified_artifact_link": False,
    "entry_type": "unclear",
    "decision": "needs_review",
    "entry_summary": "",
    "maintainer_summary": "",
    "reason": "",
    "model_name": "",
    "organization": "",
    "categories": [],
    "hardware_targets": [],
    "learning_methods": [],
    "framework": [],
    "communication": [],
}


SYSTEM_PROMPT = """
You are reviewing candidates for an Awesome Physical AI repository.

Judge whether the candidate is relevant to Physical AI, robotics, embodied AI,
robot learning, manipulation, locomotion, simulation, tactile sensing, navigation,
or related physical-world AI systems.

Use the provided paper title, abstract, authors, links, duplicate matches,
keyword-based review, artifact_availability, and link check results.

Rules:
- Reject autonomous-driving-only, traffic-only, ADAS-only, unrelated, or duplicate entries.
- Reject clearly unofficial reimplementations, fine-tunes, converted models, or community-only artifacts.
- Mark paper-only entries as needs_review unless they are clearly irrelevant.
- Treat a verified official model link as the strongest positive signal for inclusion.
- Do not treat a generic project page as a verified model release unless artifact_availability explicitly shows a verified model/code/dataset/space artifact link.
- If has_verified_model_link is false, state that no verified model link was found in the reason and maintainer_summary.
- Prefer needs_review over reject when the candidate is plausibly Physical AI but model/artifact availability is unclear.
- The entry_summary should be a public-facing 2-3 sentence description that could be used as an Awesome-list item description.
- The entry_summary must summarize the candidate's task, method/artifact, and Physical AI relevance, but must not invent artifact availability.
- Always write maintainer_summary as a concise 2-3 sentence note for the generated model PR review.
- The maintainer_summary should explain inclusion relevance, artifact availability, and any caution such as paper-only, unofficial, gated, placeholder, or inconclusive links.
- Do not invent links, stars, datasets, models, code releases, benchmarks, or claims not present in the input.
- model_name and organization must be supported by the title or an official repository namespace; otherwise return an empty string.
- Metadata arrays are optional PR preparation annotations. Select only explicitly supported values from the allowed lists below.
- Return only valid JSON. Do not wrap it in markdown.

Required JSON shape:
{
  "has_verified_model_link": boolean,
  "has_verified_artifact_link": boolean,
  "entry_type": "model" | "dataset" | "tool" | "benchmark" | "simulator" | "paper_only" | "irrelevant" | "unclear",
  "decision": "accept" | "needs_review" | "reject",
  "entry_summary": string,
  "maintainer_summary": string,
  "reason": string,
  "model_name": string,
  "organization": string,
  "categories": ["manipulation" | "locomotion" | "navigation" | "dexterous" | "whole-body" | "aerial"],
  "hardware_targets": ["manipulator" | "humanoid" | "quadruped" | "biped" | "mobile" | "drone" | "hand"],
  "learning_methods": ["VLA" | "IL" | "RL" | "diffusion" | "world_model" | "sim2real"],
  "framework": ["pytorch" | "jax" | "tensorflow" | "other"],
  "communication": ["ros2" | "grpc" | "lcm" | "zenoh" | "other"]
}
"""


def read_stdin_json() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        raise ValueError("empty stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("stdin JSON must be an object")
    return data


def build_user_prompt(candidate: dict[str, Any]) -> str:
    return "Candidate JSON:\n" + json.dumps(candidate, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("OpenRouter returned non-object JSON")
    return data


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(REQUIRED_FIELDS)
    normalized.update(review)

    normalized["has_verified_model_link"] = bool(normalized.get("has_verified_model_link"))
    normalized["has_verified_artifact_link"] = bool(normalized.get("has_verified_artifact_link"))

    if normalized.get("entry_type") not in {
        "model",
        "dataset",
        "tool",
        "benchmark",
        "simulator",
        "paper_only",
        "irrelevant",
        "unclear",
    }:
        normalized["entry_type"] = "unclear"

    if normalized.get("decision") not in {"accept", "needs_review", "reject"}:
        normalized["decision"] = "needs_review"

    for field in ("entry_summary", "maintainer_summary", "reason", "model_name", "organization"):
        normalized[field] = str(normalized.get(field) or "")

    allowed_lists = {
        "categories": {"manipulation", "locomotion", "navigation", "dexterous", "whole-body", "aerial"},
        "hardware_targets": {"manipulator", "humanoid", "quadruped", "biped", "mobile", "drone", "hand"},
        "learning_methods": {"VLA", "IL", "RL", "diffusion", "world_model", "sim2real"},
        "framework": {"pytorch", "jax", "tensorflow", "other"},
        "communication": {"ros2", "grpc", "lcm", "zenoh", "other"},
    }
    for field, allowed in allowed_lists.items():
        values = normalized.get(field)
        normalized[field] = [value for value in values if value in allowed] if isinstance(values, list) else []

    normalized.setdefault("status", "ok")
    return normalized


def call_openrouter(*, api_key: str, model: str, candidate: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://github.com/PyTorchKR/Awesome-Physical-AI"),
        "X-OpenRouter-Title": os.environ.get("OPENROUTER_APP_TITLE", "Awesome Physical AI Discovery"),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(candidate)},
        ],
        "temperature": 0,
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("OpenRouter returned no choices")

    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return normalize_review(extract_json_object(content))


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": "OPENROUTER_API_KEY environment variable is not set",
                },
                ensure_ascii=False,
            )
        )
        return 0

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

    try:
        candidate = read_stdin_json()
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": f"invalid stdin JSON: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        review = call_openrouter(api_key=api_key, model=model, candidate=candidate)
        print(json.dumps(review, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": f"OpenRouter review failed: {exc}",
                },
                ensure_ascii=False,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
