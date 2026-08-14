from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "cleaned_documents.csv"
OUTPUT_PATH = BASE_DIR / "relation_candidates.csv"

# Số hiệu có dạng 22/2023/TT-NHNN, 32/2024/QH15, 73/2016/NĐ-CP, ...
DOCUMENT_NUMBER_RE = re.compile(
    r"(?<![\w/])\d{1,4}/\d{4}/[A-ZĐ][A-ZĐ0-9]*(?:-[A-ZĐ0-9]+)*(?![\w/-])",
    re.IGNORECASE,
)

TRIGGER_PATTERNS = (
    ("Sửa đổi, bổ sung", re.compile(r"sửa\s+đổi\s*,?\s*bổ\s+sung", re.IGNORECASE)),
    ("bãi bỏ", re.compile(r"bãi\s+bỏ", re.IGNORECASE)),
    ("thay thế", re.compile(r"thay\s+thế", re.IGNORECASE)),
    ("Căn cứ", re.compile(r"căn\s+cứ", re.IGNORECASE)),
    ("Thông tư số", re.compile(r"thông\s+tư\s+số", re.IGNORECASE)),
    ("Nghị định số", re.compile(r"nghị\s+định\s+số", re.IGNORECASE)),
    ("Luật số", re.compile(r"luật\s+số", re.IGNORECASE)),
    ("Quyết định số", re.compile(r"quyết\s+định\s+số", re.IGNORECASE)),
    ("Văn bản số", re.compile(r"văn\s+bản\s+số", re.IGNORECASE)),
)

TRIGGER_PRIORITY = {
    "Sửa đổi, bổ sung": 0,
    "bãi bỏ": 1,
    "thay thế": 2,
    "Căn cứ": 3,
    "Thông tư số": 4,
    "Nghị định số": 5,
    "Luật số": 6,
    "Quyết định số": 7,
    "Văn bản số": 8,
    "Số hiệu văn bản": 9,
}


def normalize_number(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def evidence_span(text: str, start: int, end: int, radius: int = 260) -> str:
    left_limit = max(0, start - radius)
    right_limit = min(len(text), end + radius)
    left_breaks = [text.rfind(mark, left_limit, start) for mark in (". ", "; ", "\n")]
    left = max(left_breaks)
    left = left + 1 if left >= 0 else left_limit

    right_candidates = []
    for mark in (". ", "; ", "\n"):
        position = text.find(mark, end, right_limit)
        if position >= 0:
            right_candidates.append(position + 1)
    right = min(right_candidates) if right_candidates else right_limit
    return re.sub(r"\s+", " ", text[left:right]).strip()


def choose_trigger(text: str, start: int, end: int) -> tuple[str, int]:
    window_start = max(0, start - 220)
    window_end = min(len(text), end + 100)
    window = text[window_start:window_end]
    reference_center = start - window_start
    choices: list[tuple[int, int, str]] = []
    for label, pattern in TRIGGER_PATTERNS:
        for match in pattern.finditer(window):
            distance = min(abs(reference_center - match.start()), abs(reference_center - match.end()))
            choices.append((TRIGGER_PRIORITY[label], distance, label))

    if not choices:
        return "Số hiệu văn bản", TRIGGER_PRIORITY["Số hiệu văn bản"]

    # Quan hệ pháp lý rõ được ưu tiên; trong cùng loại chọn trigger gần số hiệu nhất.
    priority, _, label = min(choices)
    return label, priority


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    documents = pd.read_csv(INPUT_PATH, dtype={"id": "string"})
    required = {"id", "so_ky_hieu", "content_clean"}
    if missing := required.difference(documents.columns):
        raise ValueError(f"cleaned_documents.csv thiếu cột: {sorted(missing)}")

    candidates: list[dict[str, object]] = []
    self_references_removed = 0
    for row in documents.itertuples(index=False):
        text = str(row.content_clean) if pd.notna(row.content_clean) else ""
        source_number = normalize_number(row.so_ky_hieu)
        for match in DOCUMENT_NUMBER_RE.finditer(text):
            target = normalize_number(match.group(0))
            if target == source_number:
                self_references_removed += 1
                continue
            evidence = evidence_span(text, match.start(), match.end())
            evidence_match = DOCUMENT_NUMBER_RE.search(
                evidence,
                pos=max(0, evidence.casefold().find(match.group(0).casefold())),
            )
            if evidence_match is None:
                trigger, priority = "Số hiệu văn bản", TRIGGER_PRIORITY["Số hiệu văn bản"]
            else:
                trigger, priority = choose_trigger(
                    evidence, evidence_match.start(), evidence_match.end()
                )
            candidates.append(
                {
                    "source_id": row.id,
                    "source_so_ky_hieu": source_number,
                    "target_so_ky_hieu": target,
                    "trigger": trigger,
                    "evidence": evidence,
                    "_priority": priority,
                }
            )

    raw_count = len(candidates)
    result = pd.DataFrame(candidates)
    if result.empty:
        result = pd.DataFrame(
            columns=[
                "source_id",
                "source_so_ky_hieu",
                "target_so_ky_hieu",
                "trigger",
                "evidence",
            ]
        )
    else:
        result["_evidence_length"] = result["evidence"].str.len()
        result = (
            result.sort_values(
                ["source_id", "target_so_ky_hieu", "_priority", "_evidence_length"],
                ascending=[True, True, True, False],
            )
            .drop_duplicates(["source_id", "target_so_ky_hieu"], keep="first")
            .drop(columns=["_priority", "_evidence_length"])
            .reset_index(drop=True)
        )

    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    duplicate_count = int(
        result.duplicated(["source_id", "target_so_ky_hieu"], keep=False).sum()
    )
    empty_evidence = int(result["evidence"].fillna("").str.strip().eq("").sum())
    target_absent = int(
        result.apply(
            lambda row: normalize_number(row["target_so_ky_hieu"])
            not in normalize_number(row["evidence"]),
            axis=1,
        ).sum()
    )

    print(f"documents={len(documents)}")
    print(f"raw_occurrences={raw_count}")
    print(f"self_references_removed={self_references_removed}")
    print(f"candidates={len(result)}")
    print(f"duplicates={duplicate_count}")
    print(f"empty_evidence={empty_evidence}")
    print(f"target_absent_from_evidence={target_absent}")
    print("candidates_by_trigger:")
    print(result["trigger"].value_counts().to_string())
    print("sample_candidates:")
    for row in result.head(10).itertuples(index=False):
        print(
            f"{row.source_so_ky_hieu} -> {row.target_so_ky_hieu} | "
            f"trigger={row.trigger} | evidence={row.evidence[:240]}"
        )

    passed = all(
        (
            OUTPUT_PATH.exists(),
            len(result) > 0,
            duplicate_count == 0,
            empty_evidence == 0,
            target_absent == 0,
        )
    )
    print(f"STEP_2={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
