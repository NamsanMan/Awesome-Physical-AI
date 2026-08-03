"""Unit tests for scripts/discovery_seen_cache.py."""

from dataclasses import dataclass

import discovery_seen_cache as cache


@dataclass
class Candidate:
    source: str = "arxiv"
    title: str = "Open Robot Manipulation Policy"
    url: str = "https://arxiv.org/abs/2601.00001"
    published: str = "2026-01-01"


def test_canonicalize_seen_url_normalizes_arxiv_versions_and_scheme():
    assert cache.canonicalize_seen_url("http://arxiv.org/abs/2601.00001v3") == "https://arxiv.org/abs/2601.00001"
    assert cache.canonicalize_seen_url("https://arxiv.org/abs/2601.00001v2?x=1") == "https://arxiv.org/abs/2601.00001"


def test_seen_cache_filters_canonical_paper_url():
    seen_cache = {
        "version": 1,
        "seen": {
            "https://arxiv.org/abs/2601.00001": {
                "title": "Already Seen",
            }
        },
    }
    seen = Candidate(url="http://arxiv.org/abs/2601.00001v2")
    fresh = Candidate(title="New Robot Policy", url="https://arxiv.org/abs/2601.00002")

    remaining, skipped = cache.filter_seen_candidates([seen, fresh], seen_cache)

    assert remaining == [fresh]
    assert skipped == [seen]


def test_seen_cache_update_preserves_first_seen_timestamp():
    candidate = Candidate()
    seen_cache = {
        "version": 1,
        "seen": {
            "https://arxiv.org/abs/2601.00001": {
                "first_seen_at": "2026-01-01T00:00:00Z",
            }
        },
    }

    updated = cache.update_seen_cache(seen_cache, [candidate], seen_at="2026-01-02T00:00:00Z")
    entry = updated["seen"]["https://arxiv.org/abs/2601.00001"]

    assert entry["title"] == "Open Robot Manipulation Policy"
    assert entry["first_seen_at"] == "2026-01-01T00:00:00Z"
    assert entry["last_seen_at"] == "2026-01-02T00:00:00Z"


def test_seen_cache_write_and_load_roundtrip(tmp_path):
    cache_path = tmp_path / "seen.json"
    candidate = Candidate(url="http://arxiv.org/abs/2601.00001v3")
    seen_cache = cache.update_seen_cache(cache.empty_seen_cache(), [candidate], seen_at="2026-01-02T00:00:00Z")

    cache.write_seen_cache(str(cache_path), seen_cache)
    loaded = cache.load_seen_cache(str(cache_path))

    assert "https://arxiv.org/abs/2601.00001" in loaded["seen"]
    assert loaded["seen"]["https://arxiv.org/abs/2601.00001"]["last_seen_at"] == "2026-01-02T00:00:00Z"
