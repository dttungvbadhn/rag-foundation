from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_PATH = BASE_DIR / "cleaned_documents.csv"
CANDIDATES_PATH = BASE_DIR / "relation_candidates.csv"
ENTITIES_PATH = BASE_DIR / "entities.csv"
ENRICHED_PATH = BASE_DIR / "enriched_metadata.csv"
OUTPUT_PATH = BASE_DIR / "relationships_raw.csv"

ENTITY_RELATIONS = {
    "CoQuan": "BAN_HANH_BOI",
    "NguoiKy": "KY_BOI",
    "DoiTuongApDung": "AP_DUNG_CHO",
    "LinhVuc": "THUOC_LINH_VUC",
}
REFERENCE_TRIGGERS = {
    "Căn cứ", "Thông tư số", "Nghị định số", "Luật số",
    "Quyết định số", "Văn bản số",
}


def compact(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def fold(value: object) -> str:
    return compact(value).casefold()


def document_ref(number: str, number_to_id: dict[str, str]) -> str:
    return number_to_id.get(fold(number), f"docnum:{compact(number).upper()}")


def classify_document_candidate(
    candidate: object,
    title: str,
    number_to_id: dict[str, str],
) -> dict[str, object] | None:
    trigger = compact(candidate.trigger)
    evidence = compact(candidate.evidence)
    target_number = compact(candidate.target_so_ky_hieu).upper()
    source_id = compact(candidate.source_id)
    target_ref = document_ref(target_number, number_to_id)

    if trigger in REFERENCE_TRIGGERS:
        return {
            "source": source_id,
            "target": target_ref,
            "relationship_type": "THAM_CHIEU",
            "method": "rule",
            "confidence": 0.95 if trigger == "Căn cứ" else 0.78,
            "evidence": evidence,
        }

    if trigger == "Sửa đổi, bổ sung":
        title_fold = fold(title)
        evidence_fold = fold(evidence)
        target_fold = fold(target_number)
        if evidence_fold.startswith("căn cứ"):
            return {
                "source": source_id,
                "target": target_ref,
                "relationship_type": "THAM_CHIEU",
                "method": "rule",
                "confidence": 0.95,
                "evidence": evidence,
            }
        target_named_in_title = target_fold in title_fold and "sửa đổi" in title_fold
        direct_heading = (
            (evidence_fold.startswith("chương") or evidence_fold.startswith("sửa đổi"))
            and target_fold in evidence_fold
            and "sửa đổi" in evidence_fold
        )
        if target_named_in_title or direct_heading:
            return {
                "source": source_id,
                "target": target_ref,
                "relationship_type": "SUA_DOI_BO_SUNG",
                "method": "rule",
                "confidence": 0.96 if target_named_in_title else 0.88,
                "evidence": evidence,
            }
        return None

    if trigger == "thay thế":
        evidence_fold = fold(evidence)
        anchored = (
            "thông tư này" in evidence_fold
            or "nghị định này" in evidence_fold
            or "luật này" in evidence_fold
            or "văn bản này" in evidence_fold
        )
        if anchored and fold(target_number) in evidence_fold:
            # Chiều bắt buộc: văn bản cũ -> văn bản mới.
            return {
                "source": target_ref,
                "target": source_id,
                "relationship_type": "THAY_THE_BOI",
                "method": "rule",
                "confidence": 0.95,
                "evidence": evidence,
            }
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    documents = pd.read_csv(DOCUMENTS_PATH, dtype={"id": "string"})
    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"source_id": "string"})
    entities = pd.read_csv(ENTITIES_PATH, dtype={"source_doc_id": "string"})
    enriched = pd.read_csv(ENRICHED_PATH, dtype={"id": "string"})
    if not enriched["gemini_status"].eq("success").all():
        raise ValueError("Bước 3 chưa PASS")

    title_by_id = documents.set_index("id")["title"].fillna("").to_dict()
    number_to_id = {
        fold(number): compact(doc_id)
        for doc_id, number in zip(documents["id"], documents["so_ky_hieu"])
    }
    relationships: list[dict[str, object]] = []
    omitted_candidates = 0
    for candidate in candidates.itertuples(index=False):
        relation = classify_document_candidate(
            candidate, title_by_id.get(compact(candidate.source_id), ""), number_to_id
        )
        if relation:
            relationships.append(relation)
        else:
            omitted_candidates += 1

    # Một edge Document→Entity cho mỗi canonical entity trong mỗi document.
    entity_mentions = entities.copy()
    entity_mentions["_method_rank"] = entity_mentions["method"].map(
        {"metadata": 0, "gemini": 1}
    ).fillna(2)
    entity_mentions = (
        entity_mentions.sort_values(
            ["source_doc_id", "entity_id", "_method_rank", "confidence"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["source_doc_id", "entity_id"], keep="first")
    )
    for entity in entity_mentions.itertuples(index=False):
        relation_type = ENTITY_RELATIONS.get(compact(entity.entity_type))
        if not relation_type:
            continue
        relationships.append(
            {
                "source": compact(entity.source_doc_id),
                "target": compact(entity.entity_id),
                "relationship_type": relation_type,
                "method": compact(entity.method),
                "confidence": float(entity.confidence),
                "evidence": compact(entity.evidence),
            }
        )

    result = pd.DataFrame(
        relationships,
        columns=["source", "target", "relationship_type", "method", "confidence", "evidence"],
    )
    before = len(result)
    result = result.sort_values(
        ["source", "target", "relationship_type", "confidence"],
        ascending=[True, True, True, False],
    ).drop_duplicates(["source", "target", "relationship_type"], keep="first")
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    empty_evidence = int(result["evidence"].fillna("").str.strip().eq("").sum())
    print(f"candidate_input={len(candidates)}")
    print(f"candidate_omitted_as_ambiguous={omitted_candidates}")
    print(f"relationships_raw={len(result)}")
    print(f"duplicates_removed={before - len(result)}")
    print(f"empty_evidence={empty_evidence}")
    print("relationships_by_type:")
    print(result["relationship_type"].value_counts().to_string())
    print("sample_relationships:")
    print(result.head(10).to_string(index=False))
    passed = OUTPUT_PATH.exists() and not result.empty and empty_evidence == 0
    print(f"STEP_5={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
