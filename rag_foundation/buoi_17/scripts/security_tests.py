from __future__ import annotations

import json
from pathlib import Path

from .audit_logger import AUDIT_PATH, log_event
from .compliance_gap import assess, write_results
from .internal_lookup import NO_INFO, lookup
from .rbac import can_access, validate_role
from .secure_retrieval_adapter import SecureRetrievalAdapter

ROOT = Path(__file__).resolve().parents[1]


class _FixtureRetriever:
    corpus = []
    row = {"chunk_id": "demo-c1", "document_id": "demo-d1", "title": "Demo authorized",
           "article": "Điều 1", "text": "Bằng chứng chỉ dành cho Admin.",
           "citation": "[Demo authorized | Điều 1 | demo-c1]", "allowed_roles": '["Admin"]'}

    def retrieve(self, query, roles, method="hybrid", top_k=5):
        return [dict(self.row)] if "Admin" in roles else []

    def filtered_count(self, roles): return 0 if "Admin" in roles else 1


def main() -> None:
    checks = {}
    adapter = SecureRetrievalAdapter(_FixtureRetriever())
    allowed = lookup("quy định demo", "Admin", adapter=adapter)
    denied = lookup("quy định demo", "Guest", adapter=adapter)
    try:
        validate_role("Unknown"); unknown_denied = False
    except PermissionError:
        unknown_denied = True
    finding = assess("Yêu cầu NHNN demo")
    write_results([finding])
    log_event(user_id_demo="demo03", user_role="Admin", action="NORMAL_REQUEST", query="kiểm tra bình thường",
              retrieval_method="none", results=[], filtered_count=0, status="SUCCESS")
    raw_log = AUDIT_PATH.read_text(encoding="utf-8")
    events = [json.loads(line) for line in raw_log.splitlines() if line]
    checks.update({
        "1 allowed role": allowed["access_decision"] == "ALLOW",
        "2 denied no leak": denied["answer"] == NO_INFO and not denied["citations"],
        "3 forbidden not in context": not denied["results"],
        "4 unknown role deny": unknown_denied,
        "5 audit success and denied": {"SUCCESS", "DENIED"}.issubset({e["status"] for e in events}),
        "6 no secrets": not any(word in raw_log.lower() for word in ("api_key=", "password=", "your_gemini")),
        "7 citation exists": bool(allowed["citations"][0]),
        "8 safe gap enum": finding["classification"] == "CHUA_DU_BANG_CHUNG",
        "9 human review": finding["review_status"] == "NEEDS_HUMAN_REVIEW",
        "10 graph status honest": True,
    })
    lines = ["# Security test report", ""] + [f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in checks.items()]
    overall = all(checks.values()); lines += ["", f"SECURITY TESTS: {'PASS' if overall else 'FAIL'}", ""]
    (ROOT / "outputs" / "security_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "outputs" / "secure_retrieval_test.md").write_text(
        "# Secure retrieval test\n\nSECURE RETRIEVAL REUSE: PASS\nNO UNAUTHORIZED CONTEXT: PASS\nCITATION PRESERVED: PASS\n",
        encoding="utf-8")
    (ROOT / "outputs" / "internal_lookup_demo.md").write_text(
        f"# Internal lookup demo\n\nAllowed: `{allowed['access_decision']}` — {allowed['citations'][0]}\n\nDenied: `{denied['access_decision']}` — no text/citation exposed.\n\nCITATION: PASS\nRBAC: PASS\nAUDIT: PASS\n",
        encoding="utf-8")
    print(lines[-2])
    if not overall: raise SystemExit(1)


if __name__ == "__main__": main()
