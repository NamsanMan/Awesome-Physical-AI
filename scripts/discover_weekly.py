"""Run the discovery crawler with a persistent seen-paper cache.

This is the operational entry point used by the weekly GitHub Actions workflow.
The discovery and evaluation rules remain in ``discover_new.py``; this module
only filters papers already processed by earlier runs and persists that state.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

import discover_new as discover


SEEN_CACHE_VERSION = 1
ARXIV_ABS_RE = re.compile(r"^/abs/([^/?#]+)")


class SeenCandidate(Protocol):
    source: str
    title: str
    url: str
    published: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_seen_cache() -> dict[str, Any]:
    return {"version": SEEN_CACHE_VERSION, "seen": {}}


def canonicalize_seen_url(url: str) -> str:
    """Normalize paper URLs so arXiv versions are treated as one paper."""
    url = (url or "").strip().rstrip(".,;:")
    if not url:
        return ""

    try:
        parts = urlsplit(url)
    except ValueError:
        return url.rstrip("/")

    host = (parts.hostname or "").lower()
    match = ARXIV_ABS_RE.match(parts.path or "")
    if host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} and match:
        arxiv_id = re.sub(r"v\d+$", "", match.group(1))
        return f"https://arxiv.org/abs/{arxiv_id}"

    return url.rstrip("/")


def candidate_seen_key(candidate: SeenCandidate) -> str:
    return canonicalize_seen_url(candidate.url)


def load_seen_cache(path: str | None) -> dict[str, Any]:
    if not path:
        return empty_seen_cache()

    cache_path = Path(path)
    if not cache_path.exists():
        return empty_seen_cache()

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_seen_cache()

    if not isinstance(data, dict):
        return empty_seen_cache()

    seen = data.get("seen", {})
    if isinstance(seen, list):
        seen = {canonicalize_seen_url(str(url)): {} for url in seen if canonicalize_seen_url(str(url))}
    if not isinstance(seen, dict):
        seen = {}

    return {
        "version": int(data.get("version", SEEN_CACHE_VERSION) or SEEN_CACHE_VERSION),
        "seen": {
            canonicalize_seen_url(str(url)): meta
            for url, meta in seen.items()
            if canonicalize_seen_url(str(url))
        },
    }


def filter_seen_candidates(
    candidates: list[SeenCandidate],
    seen_cache: dict[str, Any],
) -> tuple[list[SeenCandidate], list[SeenCandidate]]:
    seen = seen_cache.get("seen", {})
    if not isinstance(seen, dict) or not seen:
        return candidates, []

    fresh: list[SeenCandidate] = []
    skipped: list[SeenCandidate] = []
    for candidate in candidates:
        if candidate_seen_key(candidate) in seen:
            skipped.append(candidate)
        else:
            fresh.append(candidate)
    return fresh, skipped


def update_seen_cache(
    seen_cache: dict[str, Any],
    candidates: list[SeenCandidate],
    *,
    seen_at: str | None = None,
) -> dict[str, Any]:
    seen_at = seen_at or utc_now_iso()
    seen = seen_cache.setdefault("seen", {})
    if not isinstance(seen, dict):
        seen = {}
        seen_cache["seen"] = seen

    for candidate in candidates:
        key = candidate_seen_key(candidate)
        if not key:
            continue

        existing = seen.get(key, {})
        if not isinstance(existing, dict):
            existing = {}

        seen[key] = {
            "title": candidate.title,
            "source": candidate.source,
            "published": candidate.published,
            "url": candidate.url,
            "first_seen_at": existing.get("first_seen_at") or seen_at,
            "last_seen_at": seen_at,
        }

    seen_cache["version"] = SEEN_CACHE_VERSION
    return seen_cache


def write_seen_cache(path: str, seen_cache: dict[str, Any]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(seen_cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run weekly Physical AI discovery with seen-paper filtering.")
    parser.add_argument("--days", type=int, default=7, help="Look back this many days.")
    parser.add_argument("--max-arxiv", type=int, default=20, help="Maximum arXiv papers to fetch.")
    parser.add_argument("--format", choices=("markdown", "json", "jsonl"), default="markdown")
    parser.add_argument("--output", help="Write report to this file instead of stdout.")
    parser.add_argument("--no-verify", action="store_true", help="Skip network checks for extracted official links.")
    parser.add_argument("--seen-cache", required=True, help="JSON cache path for papers already processed.")
    parser.add_argument(
        "--no-update-seen-cache",
        action="store_true",
        help="Filter with --seen-cache but do not persist newly processed papers.",
    )
    parser.add_argument(
        "--max-ambiguous",
        type=int,
        default=10,
        help="Maximum ambiguous candidates to send to optional LLM review.",
    )
    parser.add_argument(
        "--llm-review-mode",
        choices=("off", "ambiguous", "all"),
        default="off",
        help="Choose which candidates are sent to --llm-review-command.",
    )
    parser.add_argument(
        "--llm-review-command",
        help="Command that receives candidate JSON on stdin and returns one JSON review object.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        candidates = discover.fetch_arxiv_cs_ro(args.days, args.max_arxiv)
        seen_cache = load_seen_cache(args.seen_cache)
        fresh_candidates, skipped_candidates = filter_seen_candidates(candidates, seen_cache)
        if skipped_candidates:
            print(
                f"seen-cache: skipped {len(skipped_candidates)} previously seen candidate(s)",
                file=sys.stderr,
            )

        evaluated = discover.evaluate_candidates(
            fresh_candidates,
            verify_links=not args.no_verify,
            llm_review_command=args.llm_review_command,
            llm_review_mode=args.llm_review_mode,
            max_ambiguous=args.max_ambiguous,
        )

        if not args.no_update_seen_cache:
            update_seen_cache(seen_cache, evaluated + skipped_candidates)
            write_seen_cache(args.seen_cache, seen_cache)
    except requests.RequestException as exc:
        print(f"error: discovery request failed: {exc}", file=sys.stderr)
        return 1

    discover.write_output(evaluated, args.format, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
