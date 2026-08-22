from __future__ import annotations

import csv
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["gap_id", "external_document_id", "external_chunk_id", "external_requirement",
          "external_citation", "internal_document_id", "internal_chunk_id", "internal_evidence",
          "internal_citation", "classification", "reason", "confidence", "review_status", "request_id"]


def assess(requirement: str, external: dict | None = None, internal: dict | None = None) -> dict:
    external = external or {}
    # The real corpus contains only legislation/regulations. Absence of an internal hit is not THIEU.
    if not internal:
        classification, reason, confidence = (
            "CHUA_DU_BANG_CHUNG",
            "Corpus không có tài liệu được chứng minh là chính sách nội bộ; không thể kết luận tuân thủ.",
            0.0,
        )
        internal = {}
    else:
        classification, reason, confidence = (
            "CHUA_DU_BANG_CHUNG", "Cần chuyên gia đối chiếu đầy đủ nghĩa vụ và bằng chứng hai phía.", 0.25
        )
    return {
        "gap_id": str(uuid.uuid4()), "external_document_id": external.get("document_id", ""),
        "external_chunk_id": external.get("chunk_id", ""), "external_requirement": requirement,
        "external_citation": external.get("citation", ""), "internal_document_id": internal.get("document_id", ""),
        "internal_chunk_id": internal.get("chunk_id", ""), "internal_evidence": internal.get("text", ""),
        "internal_citation": internal.get("citation", ""), "classification": classification,
        "reason": reason, "confidence": confidence, "review_status": "NEEDS_HUMAN_REVIEW",
        "request_id": str(uuid.uuid4()),
    }


def write_results(rows: list[dict]) -> Path:
    path = ROOT / "outputs" / "compliance_gap_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    return path

