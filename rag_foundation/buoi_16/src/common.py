from __future__ import annotations

import csv
import math
import re
import unicodedata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


def source_dir() -> Path:
    candidates = [
        ROOT.parent / "kb+hops",
        ROOT.parent / "ner_kb",
        ROOT.parents[2] / "kb+hops",
        ROOT.parents[2] / "ner_kb",
    ]
    for path in candidates:
        if all((path / name).exists() for name in ("metadata.csv", "content.csv", "relationships.csv")):
            return path
    raise FileNotFoundError("Không tìm thấy bộ metadata.csv, content.csv, relationships.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def corpus_path() -> Path:
    return ROOT / "data" / "processed" / "chunks_normalized.csv"


def load_corpus() -> list[dict[str, str]]:
    path = corpus_path()
    if not path.exists():
        raise FileNotFoundError(f"Chưa có corpus: chạy scripts/prepare_corpus.py trước ({path})")
    return read_csv(path)


TOKEN_RE = re.compile(r"[\wÀ-ỹ]+(?:[-/.][\wÀ-ỹ]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def citation(row: dict[str, str]) -> str:
    parts = [row.get("title") or row.get("document_id", "")]
    if row.get("article"):
        parts.append(row["article"])
    parts.append(row.get("chunk_id", ""))
    return "[" + " | ".join(x for x in parts if x) + "]"
