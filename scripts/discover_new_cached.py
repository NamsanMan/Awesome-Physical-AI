"""
Run discover_new.py with an external seen-cache layer.

This wrapper keeps scripts/discover_new.py focused on discovery/report rendering
while adding a reusable cache that skips arXiv papers already seen in previous
runs.
"""

from __future__ import annotations

import argparse
import sys

import requests

import discover_new as discover
from discovery_seen_cache import (
    filter_seen_candidates,
    load_seen_cache,
    update_seen_cache,
    write_seen_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover Physical AI candidates with seen-cache filtering.")
    parser.add_argument("--days", type=int, default=7, help="Look back this many days.")
    parser.add_argument("--max-arxiv", type=int, default=20, help="Maximum arXiv papers to fetch.")
    parser.add_argument("--format", choices=("markdown", "json", "jsonl"), default="markdown")
    parser.add_argument("--output", help="Write report to this file instead of stdout.")
    parser.add_argument("--no-verify", action="store_true", help="Skip network checks for extracted official links.")
    parser.add_argument("--seen-cache", required=True, help="JSON cache path for arXiv papers already seen.")
    parser.add_argument(
        "--no-update-seen-cache",
        action="store_true",
        help="Filter with --seen-cache but do not write newly seen candidates back to the cache.",
    )
    parser.add_argument(
        "--max-ambiguous",
        type=int,
        default=10,
        help="Maximum ambiguous candidates to select for human/LLM review.",
    )
    parser.add_argument(
        "--llm-review-mode",
        choices=("off", "ambiguous", "all"),
        default="ambiguous",
        help=(
            "Choose which candidates are sent to --llm-review-command. "
            "'ambiguous' limits LLM review to --max-ambiguous borderline candidates."
        ),
    )
    parser.add_argument(
        "--llm-review-command",
        help=(
            "Optional command that receives candidate JSON on stdin and returns one JSON object. "
            "The returned JSON may include has_verified_model_link, has_verified_artifact_link, "
            "entry_type, decision, entry_summary, maintainer_summary, and reason."
        ),
    )
    args = parser.parse_args()

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
    sys.exit(main())
