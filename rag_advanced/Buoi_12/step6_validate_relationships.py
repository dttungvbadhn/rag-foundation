from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "relationships_raw.csv"
DOCUMENTS_PATH = BASE_DIR / "cleaned_documents.csv"
ENTITIES_PATH = BASE_DIR / "entities.csv"
OUTPUT_PATH = BASE_DIR / "relationships.csv"
REPORT_PATH = BASE_DIR / "validation_report.csv"

DOCUMENT_RELATIONS = {"THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI"}
ENTITY_RELATIONS = {
    "BAN_HANH_BOI": "CoQuan",
    "KY_BOI": "NguoiKy",
    "AP_DUNG_CHO": "DoiTuongApDung",
    "THUOC_LINH_VUC": "LinhVuc",
}
ALLOWED_RELATIONS = DOCUMENT_RELATIONS | set(ENTITY_RELATIONS)


def text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raw = pd.read_csv(RAW_PATH, dtype={"source": "string", "target": "string"})
    documents = pd.read_csv(DOCUMENTS_PATH, dtype={"id": "string"})
    entities = pd.read_csv(ENTITIES_PATH, dtype={"source_doc_id": "string"})
    required = {"source", "target", "relationship_type", "method", "confidence", "evidence"}
    if missing := required.difference(raw.columns):
        raise ValueError(f"relationships_raw.csv thiếu cột: {sorted(missing)}")

    document_ids = set(documents["id"].dropna().astype(str))
    entity_types = (
        entities[["entity_id", "entity_type"]]
        .drop_duplicates("entity_id")
        .set_index("entity_id")["entity_type"]
        .to_dict()
    )
    seen: set[tuple[str, str, str]] = set()
    report_rows = []
    pass_rows = []

    for index, row in raw.iterrows():
        source = text(row["source"])
        target = text(row["target"])
        relation_type = text(row["relationship_type"])
        evidence = text(row["evidence"])
        reasons = []
        if not source:
            reasons.append("missing_source")
        if not target:
            reasons.append("missing_target")
        if not relation_type:
            reasons.append("missing_relationship_type")
        elif relation_type not in ALLOWED_RELATIONS:
            reasons.append("invalid_relationship_type")
        if not evidence:
            reasons.append("missing_evidence")
        try:
            confidence = float(row["confidence"])
            if not 0 <= confidence <= 1:
                reasons.append("confidence_out_of_range")
        except (TypeError, ValueError):
            reasons.append("invalid_confidence")

        if relation_type in DOCUMENT_RELATIONS:
            if source not in document_ids:
                reasons.append("document_source_not_in_corpus")
            if target not in document_ids:
                reasons.append("document_target_not_in_corpus")
            if source and source == target:
                reasons.append("meaningless_self_loop")
        elif relation_type in ENTITY_RELATIONS:
            if source not in document_ids:
                reasons.append("document_source_not_in_corpus")
            actual_entity_type = entity_types.get(target)
            if actual_entity_type is None:
                reasons.append("entity_target_not_found")
            elif actual_entity_type != ENTITY_RELATIONS[relation_type]:
                reasons.append("entity_type_mismatch")

        edge_key = (source, target, relation_type)
        if edge_key in seen:
            reasons.append("duplicate_edge")
        else:
            seen.add(edge_key)

        status = "FAIL" if reasons else "PASS"
        report_rows.append(
            {
                "raw_row": index + 2,
                "source": source,
                "target": target,
                "relationship_type": relation_type,
                "status": status,
                "reason": " | ".join(reasons),
            }
        )
        if status == "PASS":
            pass_rows.append(row.to_dict())

    validated = pd.DataFrame(pass_rows, columns=list(raw.columns))
    report = pd.DataFrame(report_rows)
    validated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    report.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    fail_report = report[report["status"].eq("FAIL")]
    print(f"relationships_raw={len(raw)}")
    print(f"relationships_pass={len(validated)}")
    print(f"relationships_fail={len(fail_report)}")
    print("pass_by_type:")
    print(validated["relationship_type"].value_counts().to_string())
    print("common_fail_reasons:")
    reasons = fail_report["reason"].str.split(r" \| ").explode().value_counts()
    print(reasons.to_string() if not reasons.empty else "none")
    print("sample_pass:")
    print(validated.head(10).to_string(index=False))

    output_duplicates = int(
        validated.duplicated(["source", "target", "relationship_type"]).sum()
    )
    serious_output_failures = int(
        validated[["source", "target", "relationship_type", "evidence"]]
        .fillna("")
        .apply(lambda col: col.astype(str).str.strip().eq(""))
        .any(axis=1)
        .sum()
    )
    passed = all(
        (
            OUTPUT_PATH.exists(),
            REPORT_PATH.exists(),
            not validated.empty,
            output_duplicates == 0,
            serious_output_failures == 0,
            len(report) == len(raw),
        )
    )
    print(f"STEP_6={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
