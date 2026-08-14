from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "extracted_entities_raw.csv"
ENRICHED_PATH = BASE_DIR / "enriched_metadata.csv"
OUTPUT_PATH = BASE_DIR / "entities.csv"

REQUIRED_COLUMNS = {
    "source_doc_id", "entity", "entity_type", "method", "confidence", "evidence"
}

# Chỉ gồm alias có ý nghĩa đồng nhất rõ ràng, không fuzzy merge.
CONTROLLED_ALIASES = {
    ("CoQuan", "nhnn"): "Ngân hàng Nhà nước Việt Nam",
    ("CoQuan", "ngân hàng nhà nước"): "Ngân hàng Nhà nước Việt Nam",
    ("CoQuan", "ngân hàng nhà nước việt nam"): "Ngân hàng Nhà nước Việt Nam",
    ("CoQuan", "chính phủ"): "Chính phủ",
    ("CoQuan", "quốc hội"): "Quốc hội",
    ("CoQuan", "bộ tài chính"): "Bộ Tài chính",
    ("DoiTuongApDung", "ngân hàng hợp tác xã"): "Ngân hàng hợp tác xã",
    ("DoiTuongApDung", "quỹ tín dụng nhân dân"): "Quỹ tín dụng nhân dân",
    ("LinhVuc", "chứng khoán và thị trường chứng khoán"): "Chứng khoán",
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def comparison_key(value: object) -> str:
    return normalize_text(value).casefold()


def entity_id(entity_type: str, canonical_name: str) -> str:
    key = f"{entity_type}\0{comparison_key(canonical_name)}".encode("utf-8")
    return "ent_" + hashlib.sha256(key).hexdigest()[:16]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    raw = pd.read_csv(RAW_PATH, dtype={"source_doc_id": "string"})
    enriched = pd.read_csv(ENRICHED_PATH, dtype={"id": "string"})
    if missing := REQUIRED_COLUMNS.difference(raw.columns):
        raise ValueError(f"extracted_entities_raw.csv thiếu cột: {sorted(missing)}")
    if not enriched["gemini_status"].eq("success").all():
        raise ValueError("Bước 3 chưa PASS cho toàn bộ document")

    rows = []
    alias_changes: list[tuple[str, str, str]] = []
    for record in raw.to_dict("records"):
        original = normalize_text(record["entity"])
        entity_type = normalize_text(record["entity_type"])
        if not original or not entity_type:
            continue
        canonical = CONTROLLED_ALIASES.get(
            (entity_type, comparison_key(original)), original
        )
        if comparison_key(canonical) != comparison_key(original) or canonical != original:
            alias_changes.append((entity_type, original, canonical))
        rows.append(
            {
                "entity_id": entity_id(entity_type, canonical),
                "entity_type": entity_type,
                "canonical_name": canonical,
                "original_name": original,
                "source_doc_id": normalize_text(record["source_doc_id"]),
                "method": normalize_text(record["method"]),
                "confidence": float(record["confidence"]),
                "evidence": normalize_text(record["evidence"]),
            }
        )

    normalized = pd.DataFrame(rows)
    before_dedup = len(normalized)
    normalized["_original_key"] = normalized["original_name"].map(comparison_key)
    normalized["_evidence_key"] = normalized["evidence"].map(comparison_key)
    normalized = normalized.drop_duplicates(
        [
            "entity_id", "_original_key", "source_doc_id", "method", "_evidence_key"
        ],
        keep="first",
    ).drop(columns=["_original_key", "_evidence_key"])
    normalized = normalized.sort_values(
        ["entity_type", "canonical_name", "source_doc_id"],
        key=lambda series: series.astype(str).str.casefold(),
    ).reset_index(drop=True)
    normalized.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    obvious_duplicates = int(
        normalized.assign(
            name_key=normalized["canonical_name"].map(comparison_key),
            evidence_key=normalized["evidence"].map(comparison_key),
        ).duplicated(
            ["entity_type", "name_key", "source_doc_id", "method", "evidence_key"]
        ).sum()
    )
    person_alias_merges = {
        (old, new)
        for entity_type, old, new in alias_changes
        if entity_type == "NguoiKy" and comparison_key(old) != comparison_key(new)
    }
    traceability_missing = int(
        normalized[["canonical_name", "original_name", "source_doc_id"]]
        .fillna("")
        .apply(lambda col: col.astype(str).str.strip().eq(""))
        .any(axis=1)
        .sum()
    )

    unique_before = raw.assign(
        key=raw["entity_type"].astype(str) + "\0" + raw["entity"].map(comparison_key)
    )["key"].nunique()
    unique_after = normalized["entity_id"].nunique()
    merged_aliases = sorted(set(alias_changes), key=lambda item: tuple(map(str.casefold, item)))
    print(f"entity_mentions_before={len(raw)}")
    print(f"entity_mentions_after={len(normalized)}")
    print(f"duplicate_mentions_removed={before_dedup - len(normalized)}")
    print(f"unique_entities_before={unique_before}")
    print(f"unique_entities_after={unique_after}")
    print(f"obvious_duplicates={obvious_duplicates}")
    print(f"person_alias_merges={len(person_alias_merges)}")
    print(f"traceability_missing={traceability_missing}")
    print("aliases_merged:")
    for entity_type, original, canonical in merged_aliases:
        print(f"{entity_type}: {original} -> {canonical}")
    print("sample_entities:")
    print(normalized.head(10).to_string(index=False))

    passed = all(
        (
            OUTPUT_PATH.exists(),
            not normalized.empty,
            obvious_duplicates == 0,
            not person_alias_merges,
            traceability_missing == 0,
        )
    )
    print(f"STEP_4={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
