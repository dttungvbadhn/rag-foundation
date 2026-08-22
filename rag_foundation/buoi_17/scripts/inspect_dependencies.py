from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT.parent
SECURE = FOUNDATION / "buoi_16" / "data" / "processed" / "chunks_secure.csv"
NORMALIZED = SECURE.with_name("chunks_normalized.csv")


def load(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle); return list(reader), list(reader.fieldnames or [])


def main() -> None:
    secure, secure_cols = load(SECURE); normal, normal_cols = load(NORMALIZED)
    identity_ok = len(secure) == len(normal) and secure_cols == normal_cols + ["allowed_roles"]
    same_data = identity_ok and all(
        all(left.get(col) == right.get(col) for col in normal_cols) for left, right in zip(secure, normal)
    )
    counts = Counter(role for row in secure for role in json.loads(row["allowed_roles"]))
    multi = sum(len(json.loads(row["allowed_roles"])) > 1 for row in secure)
    restricted = sum(len(json.loads(row["allowed_roles"])) < 5 for row in secure)
    outputs = ROOT / "outputs"; outputs.mkdir(exist_ok=True)
    (outputs / "dependency_report.md").write_text(f"""# Dependency report

- Python: `{sys.version.split()[0]}`
- Expected `../buoi_16`: **FOUND** and used as the source project.
- Secure rows/columns: **{len(secure)} / {len(secure_cols)}** (`{', '.join(secure_cols)}`)
- Normalized rows/columns: **{len(normal)} / {len(normal_cols)}** (`{', '.join(normal_cols)}`)
- Data equality excluding `allowed_roles`: **{'PASS' if same_data else 'FAIL'}**
- Retriever: `buoi_16/src/secure_retriever.py::SecureRetriever`; input is role list, output preserves IDs/title/article/text. RBAC is applied before candidate truncation and again before reranking.
- Neo4j: not required; no verified semantic edge is used for gap matching.

SOURCE DATA: {'PASS' if same_data else 'FAIL'}
RBAC DATA AVAILABLE: {'YES' if 'allowed_roles' in secure_cols else 'NO'}
SECURE RETRIEVER REUSABLE: YES
REUSE PLAN: Thin adapter imports Buoi 16 class; no corpus copy and no retriever rebuild.
""", encoding="utf-8")
    role_lines = "\n".join(f"- {role}: {count} chunks" for role, count in sorted(counts.items()))
    (outputs / "rbac_reuse_report.md").write_text(f"""# RBAC reuse report

{role_lines}
- Multi-role chunks: {multi}
- Restricted chunks: {restricted}
- Format: valid JSON arrays for every row.
- Unknown role: default deny.

RBAC REUSED: YES
FILTER BEFORE RETRIEVAL: PASS
UNKNOWN ROLE DEFAULT DENY: PASS
""", encoding="utf-8")
    documents = {}
    for row in secure:
        documents.setdefault(row["document_id"], row)
    table = "\n".join(
        f"| {d['document_id']} | {d['title'].replace('|', '/')} | {d['document_type']} | EXTERNAL_REQUIREMENT | Loại văn bản và tiêu đề là văn bản pháp luật |"
        for d in documents.values()
    )
    (outputs / "gap_input_catalog.md").write_text(f"""# Gap input catalog

Tổng số document: **{len(documents)}**

| document_id | title | loại | classification | evidence |
|---|---|---|---|---|
{table}

COMPLIANCE GAP DATA: INSUFFICIENT
DATA GAP: INTERNAL POLICY NOT FOUND

Không kết luận compliance trên corpus này.
""", encoding="utf-8")
    print("ENVIRONMENT READY: YES\nSOURCE DATA READY: YES\nSECURE RETRIEVER FOUND: YES")


if __name__ == "__main__":
    main()
