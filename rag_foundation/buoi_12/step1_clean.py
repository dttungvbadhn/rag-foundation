from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "metadata.csv"
CONTENT_PATH = BASE_DIR / "content.csv"
OUTPUT_PATH = BASE_DIR / "cleaned_documents.csv"
LEGAL_PHRASES = ("Căn cứ", "Sửa đổi, bổ sung", "bãi bỏ", "thay thế")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_html(value: object) -> str:
    if pd.isna(value):
        return ""
    soup = BeautifulSoup(str(value), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def nonstandard_counts(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        values = frame[column]
        strings = values.astype("string")
        rows.append(
            {
                "column": column,
                "missing": int(values.isna().sum()),
                "empty": int(strings.str.strip().eq("").fillna(False).sum()),
                "literal_null": int(
                    strings.str.strip().str.casefold().eq("null").fillna(False).sum()
                ),
                "chua_phan_loai": int(
                    strings.str.strip().str.casefold().eq("chưa phân loại").fillna(False).sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    source_hashes_before = {
        METADATA_PATH: sha256(METADATA_PATH),
        CONTENT_PATH: sha256(CONTENT_PATH),
    }
    metadata = pd.read_csv(METADATA_PATH, dtype={"id": "string"})
    content = pd.read_csv(CONTENT_PATH, dtype={"id": "string"})

    required_metadata = {"id", "so_ky_hieu"}
    required_content = {"id", "content_html"}
    if missing := required_metadata.difference(metadata.columns):
        raise ValueError(f"metadata.csv thiếu cột: {sorted(missing)}")
    if missing := required_content.difference(content.columns):
        raise ValueError(f"content.csv thiếu cột: {sorted(missing)}")

    metadata_duplicate_rows = int(metadata["id"].duplicated(keep=False).sum())
    content_duplicate_rows = int(content["id"].duplicated(keep=False).sum())
    metadata_ids = set(metadata["id"].dropna())
    content_ids = set(content["id"].dropna())
    metadata_only = sorted(metadata_ids - content_ids)
    content_only = sorted(content_ids - metadata_ids)
    id_mismatch_count = len(metadata_only) + len(content_only)

    merged = metadata.merge(
        content,
        on="id",
        how="inner",
        validate="one_to_one",
        indicator=True,
    ).drop(columns="_merge")
    merged["content_clean"] = merged["content_html"].map(clean_html)
    quality = nonstandard_counts(metadata)

    empty_clean = int(merged["content_clean"].str.strip().eq("").sum())
    missing_ids = int(merged["id"].isna().sum())
    document_number_missing_in_html = int(
        merged.apply(
            lambda row: str(row["so_ky_hieu"]).strip().casefold()
            not in str(row["content_html"]).casefold(),
            axis=1,
        ).sum()
    )
    document_number_losses = int(
        merged.apply(
            lambda row: str(row["so_ky_hieu"]).strip().casefold()
            in str(row["content_html"]).casefold()
            and str(row["so_ky_hieu"]).strip().casefold()
            not in row["content_clean"].casefold(),
            axis=1,
        ).sum()
    )
    phrase_losses = {
        phrase: sum(
            phrase.casefold() in str(html).casefold()
            and phrase.casefold() not in clean.casefold()
            for html, clean in zip(merged["content_html"], merged["content_clean"])
        )
        for phrase in LEGAL_PHRASES
    }

    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    source_hashes_after = {
        METADATA_PATH: sha256(METADATA_PATH),
        CONTENT_PATH: sha256(CONTENT_PATH),
    }

    print(f"metadata_shape={metadata.shape}")
    print(f"content_shape={content.shape}")
    print(f"documents={len(merged)}")
    print(f"metadata_duplicate_id_rows={metadata_duplicate_rows}")
    print(f"content_duplicate_id_rows={content_duplicate_rows}")
    print(f"id_mismatch={id_mismatch_count}")
    print(f"metadata_only_ids={metadata_only}")
    print(f"content_only_ids={content_only}")
    print("metadata_quality:")
    print(quality.to_string())
    print(f"missing_ids_after_merge={missing_ids}")
    print(f"empty_content_clean={empty_clean}")
    print(f"document_number_missing_in_source_html={document_number_missing_in_html}")
    print(f"document_number_losses_during_clean={document_number_losses}")
    print(f"legal_phrase_losses={phrase_losses}")
    print(f"raw_files_unchanged={source_hashes_before == source_hashes_after}")
    print("samples:")
    for _, row in merged.head(2).iterrows():
        html_sample = re.sub(r"\s+", " ", str(row["content_html"]))[:300]
        clean_sample = row["content_clean"][:300]
        print(f"id={row['id']}")
        print(f"content_html={html_sample}")
        print(f"content_clean={clean_sample}")

    passed = all(
        (
            len(merged) == len(metadata) == len(content),
            metadata_duplicate_rows == 0,
            content_duplicate_rows == 0,
            id_mismatch_count == 0,
            missing_ids == 0,
            empty_clean == 0,
            document_number_losses == 0,
            not any(phrase_losses.values()),
            source_hashes_before == source_hashes_after,
            OUTPUT_PATH.exists(),
        )
    )
    print(f"STEP_1={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
