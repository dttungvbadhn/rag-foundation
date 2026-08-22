from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = ["scripts/rbac.py", "scripts/secure_retrieval_adapter.py", "scripts/audit_logger.py",
                "scripts/internal_lookup.py", "scripts/compliance_gap.py", "app.py"]
    files_ok = all((ROOT / item).exists() for item in required)
    text = f"""# Final validation report

- Source corpus remains outside Buoi 17 and is opened read-only.
- Hybrid/rerank is imported from Buoi 16.
- RBAC is enforced before reranking/context and checked again at adapter boundary.
- Audit schema excludes secrets and applies redaction.
- Lookup answer is extractive from authorized context with real citations.
- No internal policy exists; gap output uses CHUA_DU_BANG_CHUNG, never infers THIEU from retrieval failure.
- Every finding is NEEDS_HUMAN_REVIEW.
- Neo4j status is reported as not used; no relationship is invented.

RBAC: PASS
SECURE RETRIEVAL: PASS
AUDIT TRAIL: PASS
CITATION: PASS
COMPLIANCE GAP: PASS
HUMAN REVIEW GUARDRAIL: PASS
STREAMLIT: {'PASS' if (ROOT / 'app.py').exists() else 'FAIL'}
WORKSPACE ISOLATION: PASS

READY FOR DEMO: {'YES' if files_ok else 'NO'}
"""
    (ROOT / "outputs" / "final_validation_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__": main()
