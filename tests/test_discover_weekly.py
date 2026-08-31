"""Unit tests for the weekly discovery runner and its integrated seen cache."""

from dataclasses import dataclass
from pathlib import Path

import discover_weekly as weekly


@dataclass
class Candidate:
    source: str = "arxiv"
    title: str = "Open Robot Manipulation Policy"
    url: str = "https://arxiv.org/abs/2601.00001"
    published: str = "2026-01-01"


def test_canonicalize_seen_url_normalizes_arxiv_versions_and_scheme():
    assert weekly.canonicalize_seen_url("http://arxiv.org/abs/2601.00001v3") == "https://arxiv.org/abs/2601.00001"
    assert weekly.canonicalize_seen_url("https://arxiv.org/abs/2601.00001v2?x=1") == "https://arxiv.org/abs/2601.00001"


def test_seen_cache_filters_canonical_paper_url():
    seen_cache = {
        "version": 1,
        "seen": {"https://arxiv.org/abs/2601.00001": {"title": "Already Seen"}},
    }
    seen = Candidate(url="http://arxiv.org/abs/2601.00001v2")
    fresh = Candidate(title="New Robot Policy", url="https://arxiv.org/abs/2601.00002")

    remaining, skipped = weekly.filter_seen_candidates([seen, fresh], seen_cache)

    assert remaining == [fresh]
    assert skipped == [seen]


def test_seen_cache_update_preserves_first_seen_timestamp():
    candidate = Candidate()
    seen_cache = {
        "version": 1,
        "seen": {"https://arxiv.org/abs/2601.00001": {"first_seen_at": "2026-01-01T00:00:00Z"}},
    }

    updated = weekly.update_seen_cache(seen_cache, [candidate], seen_at="2026-01-02T00:00:00Z")
    entry = updated["seen"]["https://arxiv.org/abs/2601.00001"]

    assert entry["title"] == "Open Robot Manipulation Policy"
    assert entry["first_seen_at"] == "2026-01-01T00:00:00Z"
    assert entry["last_seen_at"] == "2026-01-02T00:00:00Z"


def test_seen_cache_write_and_load_roundtrip(tmp_path):
    cache_path = tmp_path / "seen.json"
    candidate = Candidate(url="http://arxiv.org/abs/2601.00001v3")
    seen_cache = weekly.update_seen_cache(
        weekly.empty_seen_cache(), [candidate], seen_at="2026-01-02T00:00:00Z"
    )

    weekly.write_seen_cache(str(cache_path), seen_cache)
    loaded = weekly.load_seen_cache(str(cache_path))

    assert "https://arxiv.org/abs/2601.00001" in loaded["seen"]
    assert loaded["seen"]["https://arxiv.org/abs/2601.00001"]["last_seen_at"] == "2026-01-02T00:00:00Z"


def test_weekly_cli_defaults_to_llm_off():
    args = weekly.build_parser().parse_args(["--seen-cache", "seen.json"])

    assert args.llm_review_mode == "off"


def test_workflow_uses_integrated_runner_and_dedicated_submission_token():
    workflow = (Path(__file__).parent.parent / ".github" / "workflows" / "discover-weekly.yml").read_text(
        encoding="utf-8"
    )

    assert "python scripts/discover_weekly.py" in workflow
    assert "python scripts/discovery_model_submitter.py" in workflow
    assert "secrets.DISCOVERY_BOT_TOKEN" in workflow
    assert "ARGS+=(--no-update-seen-cache)" in workflow
    assert "discover_new_cached.py" not in workflow
    assert "discovery_issue_creator.py" not in workflow
