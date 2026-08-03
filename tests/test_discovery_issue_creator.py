import json

import discovery_issue_creator as dic


def test_select_issue_candidates_defaults_to_normal_needs_review():
    candidates = [
        {"title": "Keep", "recommendation": "needs_review", "review_bucket": "normal"},
        {"title": "Skip reject", "recommendation": "reject", "review_bucket": "normal"},
        {"title": "Skip ambiguous", "recommendation": "needs_review", "review_bucket": "ambiguous"},
    ]

    selected = dic.select_issue_candidates(candidates, include_review_buckets={"normal"}, limit=5)

    assert [candidate["title"] for candidate in selected] == ["Keep"]


def test_select_issue_candidates_can_include_ambiguous_and_limit():
    candidates = [
        {"title": "One", "recommendation": "needs_review", "review_bucket": "normal"},
        {"title": "Two", "recommendation": "needs_review", "review_bucket": "ambiguous"},
        {"title": "Three", "recommendation": "needs_review", "review_bucket": "normal"},
    ]

    selected = dic.select_issue_candidates(
        candidates,
        include_review_buckets={"normal", "ambiguous"},
        limit=2,
    )

    assert [candidate["title"] for candidate in selected] == ["One", "Two"]


def test_render_issue_body_includes_rule_based_and_llm_context():
    candidate = {
        "title": "Robot Model",
        "url": "https://arxiv.org/abs/2601.00001",
        "published": "2026-01-01T00:00:00+00:00",
        "source": "arxiv:cs.RO",
        "relevance": "high",
        "recommendation": "needs_review",
        "review_bucket": "normal",
        "artifact_availability": {
            "has_verified_model_link": True,
            "has_verified_artifact_link": True,
            "has_verified_project_page": False,
        },
        "keyword_hits": ["robotics", "manipulation"],
        "reasons": ["high Physical AI relevance", "verified artifact link found"],
        "llm_review": {
            "status": "ok",
            "decision": "candidate",
            "entry_type": "model",
            "entry_summary": "Short model summary.",
            "maintainer_summary": "Maintainer note.",
        },
        "checks": [
            {
                "status": "available",
                "kind": "hf_model",
                "url": "https://huggingface.co/org/model",
                "reason": "model repository is available",
            }
        ],
        "summary": "A robotics paper.",
    }

    body = dic.render_issue_body(candidate, source_run="https://github.com/org/repo/actions/runs/1")

    assert "Rule-based verified model link: `true`" in body
    assert "Rule-based verified artifact link: `true`" in body
    assert "Short model summary." in body
    assert "Maintainer note." in body
    assert "`available` `hf_model`: https://huggingface.co/org/model" in body
    assert "https://github.com/org/repo/actions/runs/1" in body


def test_load_candidates_requires_json_list(tmp_path):
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps({"items": []}), encoding="utf-8")

    try:
        dic.load_candidates(str(path))
    except ValueError as exc:
        assert "must be a list" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-list discovery JSON")
