from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
INPUT_PATH = BASE_DIR / "cleaned_documents.csv"
RAW_ENTITY_PATH = BASE_DIR / "extracted_entities_raw.csv"
ENRICHED_PATH = BASE_DIR / "enriched_metadata.csv"
DEFAULT_MODEL = "gemini-flash-latest"
ENTITY_TYPES = {"CoQuan", "NguoiKy", "DoiTuongApDung", "LinhVuc"}
UNSTANDARD_VALUES = {"", "null", "chưa phân loại", "nan", "none"}
RAW_ENTITY_COLUMNS = [
    "source_doc_id",
    "entity",
    "entity_type",
    "source",
    "method",
    "confidence",
    "evidence",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "enum": sorted(ENTITY_TYPES),
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string"},
                },
                "required": ["entity", "entity_type", "confidence", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_unstandard(value: object) -> bool:
    return clean_scalar(value).casefold() in UNSTANDARD_VALUES


def build_excerpt(text: str) -> str:
    """Lấy các vùng giàu thông tin, không paraphrase nội dung nguồn."""
    if len(text) <= 18_000:
        return text
    spans = [(0, 7_000), (max(0, len(text) - 4_000), len(text))]
    for keyword in ("đối tượng áp dụng", "phạm vi điều chỉnh", "nơi nhận"):
        position = text.casefold().find(keyword)
        if position >= 0:
            spans.append((max(0, position - 1_000), min(len(text), position + 5_000)))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return "\n[...ĐOẠN KHÁC...]\n".join(text[start:end] for start, end in merged)


def prompt_for(row: Any) -> str:
    metadata = {
        "title": clean_scalar(row.title),
        "so_ky_hieu": clean_scalar(row.so_ky_hieu),
        "co_quan_ban_hanh": clean_scalar(row.co_quan_ban_hanh),
        "nguoi_ky": clean_scalar(row.nguoi_ky),
        "linh_vuc": clean_scalar(row.linh_vuc),
    }
    excerpt = build_excerpt(clean_scalar(row.content_clean))
    return f"""Trích xuất thực thể từ văn bản pháp luật Việt Nam.

Quy tắc bắt buộc:
- Chỉ trả entity có bằng chứng nguyên văn trong NỘI DUNG.
- evidence phải là một đoạn trích nguyên văn, ngắn, đủ kiểm chứng.
- Không suy đoán hoặc bổ sung kiến thức bên ngoài.
- Metadata gốc chỉ là ngữ cảnh; không được mâu thuẫn hoặc tự ý ghi đè giá trị rõ ràng.
- CoQuan: cơ quan ban hành; NguoiKy: người ký/có thẩm quyền ký;
  DoiTuongApDung: đối tượng chịu điều chỉnh; LinhVuc: lĩnh vực pháp lý.
- Nếu không có bằng chứng cho một loại thì không trả entity loại đó.
- confidence phản ánh độ rõ của bằng chứng, không mặc định bằng 1.

METADATA GỐC:
{json.dumps(metadata, ensure_ascii=False)}

NỘI DUNG:
{excerpt}
"""


def response_entities(response: Any) -> list[dict[str, Any]]:
    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise ValueError("empty_response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed_json: {exc.msg}") from exc
    entities = payload.get("entities")
    if not isinstance(entities, list):
        raise ValueError("missing_field: entities")
    return entities


def validate_entity(item: dict[str, Any], full_text: str) -> tuple[dict[str, Any] | None, str | None]:
    required = ("entity", "entity_type", "confidence", "evidence")
    if any(field not in item for field in required):
        return None, "missing_entity_field"
    entity = clean_scalar(item["entity"])
    entity_type = clean_scalar(item["entity_type"])
    evidence = clean_scalar(item["evidence"])
    try:
        confidence = float(item["confidence"])
    except (TypeError, ValueError):
        return None, "invalid_confidence"
    if not entity or entity_type not in ENTITY_TYPES or not evidence:
        return None, "invalid_or_empty_entity"
    if not 0 <= confidence <= 1:
        return None, "confidence_out_of_range"
    normalized_full = clean_scalar(full_text).casefold()
    if evidence.casefold() not in normalized_full:
        return None, "evidence_not_in_source"
    if entity.casefold() not in evidence.casefold():
        return None, "entity_not_in_evidence"
    return {
        "entity": entity,
        "entity_type": entity_type,
        "source": "content_clean",
        "method": "gemini",
        "confidence": confidence,
        "evidence": evidence,
    }, None


def metadata_entities(row: Any) -> list[dict[str, Any]]:
    mapping = {
        "co_quan_ban_hanh": "CoQuan",
        "nguoi_ky": "NguoiKy",
        "linh_vuc": "LinhVuc",
    }
    result = []
    for field, entity_type in mapping.items():
        value = getattr(row, field)
        if not is_unstandard(value):
            clean = clean_scalar(value)
            result.append(
                {
                    "entity": clean,
                    "entity_type": entity_type,
                    "source": f"metadata.{field}",
                    "method": "metadata",
                    "confidence": 0.99,
                    "evidence": clean,
                }
            )
    return result


def join_entities(items: list[dict[str, Any]], entity_type: str) -> str:
    values = []
    seen = set()
    for item in items:
        if item["entity_type"] != entity_type:
            continue
        key = item["entity"].casefold()
        if key not in seen:
            seen.add(key)
            values.append(item["entity"])
    return " | ".join(values)


def is_retryable_api_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in message
        for marker in (
            "429",
            "503",
            "resource_exhausted",
            "unavailable",
            "high demand",
            "timeout",
            "timed out",
            "connecterror",
            "connection",
        )
    )


def is_quota_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".casefold()
    return "429" in message or "resource_exhausted" in message or "quota" in message


def generate_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents: str,
    max_retries: int,
    base_delay: float,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_json_schema=RESPONSE_SCHEMA,
                ),
            )
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_api_error(exc):
                raise
            delay = min(base_delay * (2**attempt), 60.0) + random.uniform(0, 1)
            print(
                f"retry={attempt + 1}/{max_retries} "
                f"reason={type(exc).__name__} wait_seconds={delay:.1f}"
            )
            time.sleep(delay)
    raise RuntimeError("retry loop ended unexpectedly")


def load_checkpoint(
    documents: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    valid_ids = set(documents["id"].astype("string"))
    enriched_by_id: dict[str, dict[str, Any]] = {}
    entities_by_id: dict[str, list[dict[str, Any]]] = {}

    if ENRICHED_PATH.exists():
        previous = pd.read_csv(ENRICHED_PATH, dtype={"id": "string"})
        if "id" in previous.columns:
            for record in previous.to_dict("records"):
                doc_id = clean_scalar(record.get("id"))
                if doc_id in valid_ids:
                    enriched_by_id[doc_id] = record

    if RAW_ENTITY_PATH.exists():
        previous_entities = pd.read_csv(
            RAW_ENTITY_PATH, dtype={"source_doc_id": "string"}
        )
        if set(RAW_ENTITY_COLUMNS).issubset(previous_entities.columns):
            for record in previous_entities[RAW_ENTITY_COLUMNS].to_dict("records"):
                doc_id = clean_scalar(record.get("source_doc_id"))
                if doc_id in valid_ids:
                    entities_by_id.setdefault(doc_id, []).append(record)

    return enriched_by_id, entities_by_id


def save_checkpoint(
    documents: pd.DataFrame,
    enriched_by_id: dict[str, dict[str, Any]],
    entities_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    ordered_ids = [clean_scalar(value) for value in documents["id"]]
    enriched_records = [
        enriched_by_id[doc_id] for doc_id in ordered_ids if doc_id in enriched_by_id
    ]
    entity_records = [
        entity
        for doc_id in ordered_ids
        for entity in entities_by_id.get(doc_id, [])
    ]
    pd.DataFrame(entity_records, columns=RAW_ENTITY_COLUMNS).to_csv(
        RAW_ENTITY_PATH, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(enriched_records).to_csv(
        ENRICHED_PATH, index=False, encoding="utf-8-sig"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("STEP_3=FAIL")
        print("error=missing GEMINI_API_KEY in .env or process environment")
        raise SystemExit(2)

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    request_interval = max(0.0, float(os.getenv("GEMINI_REQUEST_INTERVAL", "12")))
    max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "5")))
    retry_base_delay = max(1.0, float(os.getenv("GEMINI_RETRY_BASE_DELAY", "2")))
    quota_circuit_breaker = max(
        1, int(os.getenv("GEMINI_QUOTA_CIRCUIT_BREAKER", "3"))
    )

    source_hash_before = sha256(INPUT_PATH)
    documents = pd.read_csv(INPUT_PATH, dtype={"id": "string"})
    required = {
        "id", "title", "so_ky_hieu", "co_quan_ban_hanh", "nguoi_ky",
        "linh_vuc", "thong_tin_ap_dung", "content_clean",
    }
    if missing := required.difference(documents.columns):
        raise ValueError(f"cleaned_documents.csv thiếu cột: {sorted(missing)}")

    client = genai.Client(api_key=api_key)
    enriched_by_id, entities_by_id = load_checkpoint(documents)
    successful_ids = {
        doc_id
        for doc_id, record in enriched_by_id.items()
        if clean_scalar(record.get("gemini_status")).casefold() == "success"
    }
    print(f"model={model}")
    print(f"checkpoint_successful={len(successful_ids)}")
    print(f"documents_to_process={len(documents) - len(successful_ids)}")
    errors: list[tuple[str, str]] = []
    consecutive_quota_failures = 0
    last_request_started: float | None = None

    for row in documents.itertuples(index=False):
        doc_id = clean_scalar(row.id)
        if doc_id in successful_ids:
            continue

        if last_request_started is not None:
            remaining = request_interval - (time.monotonic() - last_request_started)
            if remaining > 0:
                print(f"throttle_wait_seconds={remaining:.1f}")
                time.sleep(remaining)

        doc_entities = metadata_entities(row)
        rejected_reasons: list[str] = []
        status = "success"
        error_message = ""
        try:
            last_request_started = time.monotonic()
            response = generate_with_retry(
                client,
                model=model,
                contents=prompt_for(row),
                max_retries=max_retries,
                base_delay=retry_base_delay,
            )
            for item in response_entities(response):
                validated, reason = validate_entity(item, clean_scalar(row.content_clean))
                if validated:
                    doc_entities.append(validated)
                elif reason:
                    rejected_reasons.append(reason)
            consecutive_quota_failures = 0
        except Exception as exc:  # Lỗi một document không dừng toàn batch.
            status = "failed"
            error_message = f"{type(exc).__name__}: {str(exc)[:300]}"
            errors.append((str(row.id), error_message))
            consecutive_quota_failures = (
                consecutive_quota_failures + 1 if is_quota_error(exc) else 0
            )

        unique_entities: list[dict[str, Any]] = []
        seen = set()
        for item in doc_entities:
            key = (item["entity_type"], item["entity"].casefold(), item["method"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(item)

        enriched = row._asdict()
        original_co_quan = clean_scalar(row.co_quan_ban_hanh)
        original_nguoi_ky = clean_scalar(row.nguoi_ky)
        original_linh_vuc = clean_scalar(row.linh_vuc)
        original_doi_tuong = clean_scalar(row.thong_tin_ap_dung)
        enriched["co_quan_ban_hanh_enriched"] = (
            original_co_quan or join_entities(unique_entities, "CoQuan")
        )
        enriched["nguoi_ky_enriched"] = original_nguoi_ky or join_entities(
            unique_entities, "NguoiKy"
        )
        enriched["linh_vuc_enriched"] = (
            original_linh_vuc
            if not is_unstandard(original_linh_vuc)
            else join_entities(unique_entities, "LinhVuc")
        )
        enriched["doi_tuong_ap_dung_enriched"] = (
            original_doi_tuong or join_entities(unique_entities, "DoiTuongApDung")
        )
        enriched["gemini_status"] = status
        enriched["gemini_error"] = error_message
        enriched["rejected_entity_reasons"] = " | ".join(sorted(set(rejected_reasons)))
        enriched_by_id[doc_id] = enriched
        entities_by_id[doc_id] = [
            {"source_doc_id": row.id, **item} for item in unique_entities
        ]
        save_checkpoint(documents, enriched_by_id, entities_by_id)
        print(f"checkpoint id={doc_id} status={status}")

        if consecutive_quota_failures >= quota_circuit_breaker:
            print(
                "quota_circuit_breaker=OPEN; stopping safely so the next run can resume"
            )
            break

    try:
        client.close()
    except Exception:
        pass

    save_checkpoint(documents, enriched_by_id, entities_by_id)
    enriched_frame = pd.read_csv(ENRICHED_PATH, dtype={"id": "string"})
    entity_frame = pd.read_csv(
        RAW_ENTITY_PATH, dtype={"source_doc_id": "string"}
    )
    source_unchanged = source_hash_before == sha256(INPUT_PATH)

    successful = int(enriched_frame["gemini_status"].eq("success").sum())
    print(f"documents_successful={successful}")
    print(f"documents_failed={len(documents) - successful}")
    print("entities_by_type:")
    print(entity_frame["entity_type"].value_counts().to_string())
    enriched_lookup = enriched_frame.set_index("id").to_dict("index")
    added = sum(
        is_unstandard(row.linh_vuc)
        and bool(clean_scalar(enriched_lookup.get(clean_scalar(row.id), {}).get("linh_vuc_enriched")))
        for row in documents.itertuples(index=False)
    ) + sum(
        bool(clean_scalar(record.get("doi_tuong_ap_dung_enriched")))
        for record in enriched_lookup.values()
    )
    print(f"metadata_values_added={added}")
    print(f"cleaned_documents_unchanged={source_unchanged}")
    print("enrichment_samples:")
    for original in documents.head(5).itertuples(index=False):
        enriched = enriched_lookup.get(clean_scalar(original.id), {})
        print(
            f"id={original.id} | linh_vuc: {clean_scalar(original.linh_vuc)!r} -> "
            f"{clean_scalar(enriched.get('linh_vuc_enriched'))!r} | doi_tuong: "
            f"{clean_scalar(enriched.get('doi_tuong_ap_dung_enriched'))!r}"
        )
    print(f"errors={errors}")

    passed = all(
        (
            successful == len(documents),
            RAW_ENTITY_PATH.exists(),
            ENRICHED_PATH.exists(),
            source_unchanged,
            not entity_frame.empty,
            not entity_frame["evidence"].fillna("").str.strip().eq("").any(),
        )
    )
    print(f"STEP_3={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
