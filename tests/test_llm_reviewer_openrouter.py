import llm_reviewer_openrouter as reviewer


def test_extract_json_object_handles_markdown_fence():
    data = reviewer.extract_json_object(
        """```json
{"decision": "needs_review", "entry_type": "paper_only"}
```"""
    )

    assert data["decision"] == "needs_review"
    assert data["entry_type"] == "paper_only"


def test_normalize_review_fills_required_fields():
    data = reviewer.normalize_review({"decision": "accept", "has_verified_model_link": True})

    assert data["status"] == "ok"
    assert data["decision"] == "accept"
    assert data["entry_type"] == "unclear"
    assert data["has_verified_model_link"] is True
    assert data["has_verified_artifact_link"] is False
    assert "maintainer_summary" in data


def test_normalize_review_clamps_unknown_enums():
    data = reviewer.normalize_review({"decision": "maybe", "entry_type": "other"})

    assert data["decision"] == "needs_review"
    assert data["entry_type"] == "unclear"
