from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from .rbac import can_access, validate_role

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT.parent
DEFAULT_SOURCE = FOUNDATION / "buoi_16" / "data" / "processed" / "chunks_secure.csv"


def source_path() -> Path:
    configured = os.getenv("SOURCE_SECURE_CSV")
    return (ROOT / configured).resolve() if configured else DEFAULT_SOURCE


class SecureRetrievalAdapter:
    """Thin adapter around Buoi 15 SecureRetriever; source corpus stays read-only."""

    def __init__(self, retriever=None):
        if retriever is None:
            if str(FOUNDATION) not in sys.path:
                sys.path.insert(0, str(FOUNDATION))
            from buoi_16.src.secure_retriever import SecureRetriever

            retriever = SecureRetriever()
        self.retriever = retriever

    def retrieve(self, query: str, role: str, top_k: int = 5, method: str = "hybrid_rerank") -> dict:
        validate_role(role)
        rows = self.retriever.retrieve(query, [role], method=method, top_k=top_k)
        safe = [row for row in rows if can_access(row.get("allowed_roles"), role)]
        if len(safe) != len(rows):
            raise AssertionError("Unauthorized chunk crossed retrieval boundary")
        results = []
        for rank, row in enumerate(safe, 1):
            citation = row.get("citation") or "[{} | {} | {}]".format(
                row.get("title", ""), row.get("article", ""), row.get("chunk_id", "")
            )
            results.append({
                "rank": rank, "chunk_id": row.get("chunk_id", ""),
                "document_id": row.get("document_id", ""), "title": row.get("title", ""),
                "article": row.get("article", ""), "text": row.get("text", ""),
                "citation": citation, "allowed_roles": row.get("allowed_roles", []),
                "access_decision": "ALLOW", "retrieval_method": row.get("retrieval_method", method),
            })
        filtered = self.retriever.filtered_count([role])
        return {"results": results, "filtered_count": filtered, "access_decision": "ALLOW" if results else "DENY"}


def inspect_source() -> tuple[list[dict[str, str]], list[str]]:
    with source_path().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])
