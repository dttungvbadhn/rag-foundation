from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "outputs" / "audit_log.jsonl"
SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*\S+")


def _clean(value: object) -> object:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, str):
        return SECRET_PATTERN.sub(r"\1=[REDACTED]", value)
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def log_event(*, user_id_demo: str, user_role: str, action: str, query: str,
              retrieval_method: str, results: list[dict], filtered_count: int,
              status: str, request_id: str | None = None, error: str = "") -> str:
    request_id = request_id or str(uuid.uuid4())
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "request_id": request_id,
        "user_id_demo": user_id_demo, "user_role": user_role, "action": action,
        "query": query, "retrieval_method": retrieval_method,
        "retrieved_document_ids": [r.get("document_id", "") for r in results],
        "retrieved_chunk_ids": [r.get("chunk_id", "") for r in results],
        "citation_ids": [r.get("citation", "") for r in results],
        "rbac_filtered_candidates": filtered_count, "status": status,
    }
    if error:
        event["error"] = error
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean(event), ensure_ascii=False) + "\n")
    return request_id


def read_events() -> list[dict]:
    if not AUDIT_PATH.exists():
        return []
    return [json.loads(line) for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
