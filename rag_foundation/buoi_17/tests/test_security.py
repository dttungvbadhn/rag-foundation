from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compliance_gap import assess
from scripts.internal_lookup import NO_INFO, lookup
from scripts.rbac import can_access, validate_role
from scripts.secure_retrieval_adapter import SecureRetrievalAdapter


class FakeRetriever:
    corpus = []
    row = {"chunk_id": "c1", "document_id": "d1", "title": "T", "article": "Điều 1",
           "text": "Authorized evidence.", "citation": "[T | Điều 1 | c1]", "allowed_roles": '["Admin"]'}

    def retrieve(self, query, roles, method="hybrid", top_k=5):
        return [dict(self.row)] if "Admin" in roles else []

    def filtered_count(self, roles):
        return 0 if "Admin" in roles else 1


def test_allowed_and_citation_preserved():
    payload = SecureRetrievalAdapter(FakeRetriever()).retrieve("q", "Admin")
    assert payload["results"][0]["citation"] == "[T | Điều 1 | c1]"
    assert payload["results"][0]["document_id"] == "d1"


def test_denied_does_not_leak_context():
    result = lookup("q", "Guest", adapter=SecureRetrievalAdapter(FakeRetriever()))
    assert result["answer"] == NO_INFO and not result["citations"] and not result["results"]


def test_unknown_role_default_deny():
    try:
        validate_role("Unknown")
        assert False
    except PermissionError:
        pass


def test_rbac_parser():
    assert can_access('["Admin", "HR"]', "HR")
    assert not can_access('["Admin", "HR"]', "Guest")


def test_gap_guardrails():
    finding = assess("requirement")
    assert finding["classification"] == "CHUA_DU_BANG_CHUNG"
    assert finding["review_status"] == "NEEDS_HUMAN_REVIEW"
    assert finding["confidence"] == 0.0


def test_audit_has_status_and_no_secret(tmp_path, monkeypatch):
    import scripts.audit_logger as audit
    monkeypatch.setattr(audit, "AUDIT_PATH", tmp_path / "audit.jsonl")
    audit.log_event(user_id_demo="u", user_role="Admin", action="TEST",
                    query="password=hunter2", retrieval_method="fake", results=[],
                    filtered_count=0, status="SUCCESS")
    raw = audit.AUDIT_PATH.read_text(encoding="utf-8")
    event = json.loads(raw)
    assert event["status"] == "SUCCESS" and "hunter2" not in raw

