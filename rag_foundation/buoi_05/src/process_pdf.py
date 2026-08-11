"""Demo OCR PDF tiếng Việt và so sánh ba cách chia chunk.

`--dry-run` không ghi file và không gọi dịch vụ mạng. `--write` chỉ gọi
LlamaParse khi text layer của ít nhất một trang không dùng được.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import fitz
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datademo"
OUTPUT_DIR = BASE_DIR / "output"
RAW_DIR = OUTPUT_DIR / "raw"
CHUNK_DIR = OUTPUT_DIR / "chunks"
ENV_PATH = BASE_DIR / "src" / ".env"


@dataclass
class PageData:
    source: str
    page: int
    text: str
    ocr_used: bool
    language: str = "vi"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkData:
    chunk_id: str
    strategy: str
    source: str
    page_start: int
    page_end: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str | None) -> str:
    """Chuẩn hóa văn bản về Unicode NFC, không thay đổi PDF nguồn."""
    return unicodedata.normalize("NFC", text or "").strip()


def needs_ocr(text: str) -> bool:
    """Nhận diện các dấu hiệu text layer rỗng hoặc bị lỗi rõ ràng."""
    value = text.strip()
    if not value or "\ufffd" in value:
        return True
    controls = sum(unicodedata.category(char) == "Cc" and char not in "\n\t\r" for char in value)
    # Các cụm này là dấu hiệu UTF-8 bị giải mã sai thường gặp, không phải tiếng Việt NFC hợp lệ.
    mojibake = re.search(r"(?:Ã.|Â.|Ä.|Æ.|áº.|á».)", value)
    return controls > max(2, len(value) // 20) or mojibake is not None


def extract_text_layer(pdf_path: Path) -> list[PageData]:
    """Đọc text layer từng trang. Lỗi trang được lưu metadata, không làm hỏng PDF."""
    pages: list[PageData] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            warning: str | None = None
            try:
                text = normalize_text(page.get_text("text"))
            except Exception as exc:
                text = ""
                warning = f"Không thể lấy text layer: {type(exc).__name__}."
            metadata: dict[str, Any] = {"text_layer_len": len(text), "ocr_provider": "pymupdf"}
            if warning:
                metadata["warning"] = warning
            pages.append(
                PageData(
                    source=pdf_path.name,
                    page=page_number,
                    text=text,
                    ocr_used=needs_ocr(text),
                    metadata=metadata,
                )
            )
    return pages


def render_pages_for_ocr(pdf_path: Path) -> list[dict[str, int]]:
    """Render trang sang ảnh trong bộ nhớ để kiểm tra khả năng OCR; không ghi ảnh ra đĩa."""
    rendered: list[dict[str, int]] = []
    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            rendered.append({"page": page_number, "width": pixmap.width, "height": pixmap.height})
    return rendered


def split_ocr_text_by_page(text: str, page_count: int) -> tuple[list[str], bool]:
    """Tách kết quả OCR theo marker trang; nếu không có marker thì chỉ tạo 1 bản ghi an toàn."""
    parts = [part.strip() for part in re.split(r"\n+(?:trang|page)\s*\d+\s*\n+", text, flags=re.I) if part.strip()]
    if len(parts) == page_count:
        return parts, True
    if page_count == 1:
        return [text.strip()], True
    # Không đoán ranh giới trang. Toàn bộ text được gắn trang 1 kèm cảnh báo.
    return [text.strip()] + [""] * (page_count - 1), False


async def ocr_with_llamaparse(pdf_path: Path) -> list[PageData]:
    """Gửi toàn bộ PDF cho LlamaParse khi fallback OCR là bắt buộc.

    dotenv chỉ nạp biến môi trường; code không lấy, in hay log giá trị API key.
    AsyncLlamaCloud tự lấy key từ biến môi trường.
    """
    from llama_cloud import AsyncLlamaCloud

    load_dotenv(dotenv_path=ENV_PATH)
    rendered = render_pages_for_ocr(pdf_path)
    client = AsyncLlamaCloud()
    file_obj = await client.files.create(file=str(pdf_path.resolve()), purpose="parse")
    result = await client.parsing.parse(
        file_id=file_obj.id,
        tier="agentic",
        version="latest",
        expand=["markdown_full"],
    )
    text = normalize_text(getattr(result, "markdown_full", ""))
    if not text:
        raise RuntimeError("LlamaParse không trả về văn bản OCR.")

    page_texts, exact_pages = split_ocr_text_by_page(text, len(rendered))
    pages: list[PageData] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        metadata: dict[str, Any] = {
            "ocr_render": rendered[page_number - 1],
            "ocr_provider": "llamaparse",
            "ocr_split_method": "page_marker" if exact_pages else "unmapped_full_text",
        }
        if not exact_pages:
            metadata["warning"] = "OCR không có marker trang rõ ràng; không suy đoán ranh giới trang."
        pages.append(PageData(pdf_path.name, page_number, normalize_text(page_text), True, metadata=metadata))
    return pages


def chunk_fixed_size(page: PageData, max_chars: int = 700, overlap: int = 100) -> list[ChunkData]:
    if max_chars <= overlap:
        raise ValueError("max_chars phải lớn hơn overlap.")
    if not page.text:
        return []
    chunks: list[ChunkData] = []
    start = 0
    while start < len(page.text):
        end = min(start + max_chars, len(page.text))
        number = len(chunks) + 1
        chunks.append(ChunkData(
            f"fixed_{page.page}_{number}", "fixed-size", page.source, page.page, page.page,
            normalize_text(page.text[start:end]), {"max_chars": max_chars, "overlap": overlap},
        ))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def chunk_semantic(page: PageData, max_chars: int = 1000) -> list[ChunkData]:
    """Cắt ở hết đoạn/cách dòng, chỉ tách câu khi một đoạn không có cách dòng."""
    # Chỉ cách dòng trống mới là ranh giới semantic. Xuống dòng do dàn trang được nối
    # bằng dấu cách để tránh cắt giữa một câu.
    paragraphs = [
        normalize_text(re.sub(r"[ \t]*\r?\n[ \t]*", " ", item))
        for item in re.split(r"\r?\n\s*\r?\n", page.text)
        if normalize_text(item)
    ]
    bounded_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            bounded_paragraphs.append(paragraph)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?…])\s+", paragraph):
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                bounded_paragraphs.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            bounded_paragraphs.append(current)
    return [
        ChunkData(
            f"semantic_{page.page}_{index}", "semantic", page.source, page.page, page.page, text,
            {"boundary": "paragraph_or_sentence", "max_chars": max_chars},
        )
        for index, text in enumerate(bounded_paragraphs, start=1)
    ]


STRUCTURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chapter", re.compile(r"^Chương\s+[IVXLCDM]+\b|^Chương\s+\d+\b", re.I)),
    ("section", re.compile(r"^Mục\s+[IVXLCDM]+\b|^Mục\s+\d+\b", re.I)),
    ("article", re.compile(r"^Điều\s+\d+\b", re.I)),
    ("clause", re.compile(r"^(?:Khoản\s+)?\d+\.\s+")),
    ("point", re.compile(r"^(?:Điểm\s+)?[a-zđ]\)\s*", re.I)),
)


def _heading(line: str) -> tuple[str, str] | None:
    value = line.strip()
    for kind, pattern in STRUCTURE_PATTERNS:
        if pattern.search(value):
            return kind, value
    return None


def chunk_hierarchical(page: PageData) -> list[ChunkData]:
    lines = page.text.splitlines()
    starts = []
    for index, line in enumerate(lines):
        found = _heading(line)
        if found:
            starts.append((index, found))
    if not starts:
        return [ChunkData(
            f"hierarchical_{page.page}_1", "hierarchical", page.source, page.page, page.page,
            normalize_text(page.text), {"warning": "Không tìm thấy cấu trúc rõ ràng; không bịa heading."},
        )] if page.text else []

    chunks: list[ChunkData] = []
    structure: dict[str, str] = {}
    first_start = starts[0][0]
    if first_start:
        chunks.append(ChunkData(
            f"hierarchical_{page.page}_0", "hierarchical", page.source, page.page, page.page,
            normalize_text("\n".join(lines[:first_start])),
            {"warning": "Đoạn mở đầu không có heading được nhận diện; giữ nguyên, không bịa cấu trúc."},
        ))
    for index, (start, found) in enumerate(starts):
        assert found is not None
        kind, heading = found
        structure[kind] = heading
        end = starts[index + 1][0] if index + 1 < len(starts) else len(lines)
        chunks.append(ChunkData(
            f"hierarchical_{page.page}_{index + 1}", "hierarchical", page.source, page.page, page.page,
            normalize_text("\n".join(lines[start:end])),
            {"structure": structure.copy(), "heading": heading, "heading_type": kind},
        ))
    return chunks


def summarize_chunks(chunks: list[ChunkData]) -> dict[str, float | int]:
    lengths = [len(chunk.text) for chunk in chunks if chunk.text]
    return {"count": len(lengths), "min": min(lengths, default=0), "max": max(lengths, default=0), "avg": round(mean(lengths), 2) if lengths else 0.0}


def make_chunks(pages: list[PageData]) -> tuple[list[ChunkData], dict[str, dict[str, float | int]]]:
    chunks = [chunk for page in pages for function in (chunk_fixed_size, chunk_semantic, chunk_hierarchical) for chunk in function(page)]
    summary = {strategy: summarize_chunks([item for item in chunks if item.strategy == strategy]) for strategy in ("fixed-size", "semantic", "hierarchical")}
    return chunks, summary


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def process_pdf(pdf_path: Path, write: bool = False, force_ocr: bool = False) -> dict[str, Any]:
    """Xử lý một PDF. Dry-run không gọi LlamaParse dù phát hiện cần OCR."""
    try:
        pages = extract_text_layer(pdf_path)
    except (fitz.FileDataError, OSError, RuntimeError) as exc:
        return {"source": pdf_path.name, "error": f"Không thể mở PDF: {type(exc).__name__}."}

    fallback_needed = force_ocr or any(page.ocr_used for page in pages)
    warnings: list[str] = []
    if fallback_needed and not write:
        warnings.append("Phát hiện trang cần OCR; dry-run không gọi LlamaParse và không ghi output.")
    elif fallback_needed:
        try:
            pages = asyncio.run(ocr_with_llamaparse(pdf_path))
        except Exception as exc:  # API/network error is reported safely without secrets.
            return {"source": pdf_path.name, "error": f"OCR fallback thất bại: {type(exc).__name__}.", "warnings": warnings}

    chunks, summary = make_chunks(pages)
    result: dict[str, Any] = {"source": pdf_path.name, "summary": summary, "warnings": warnings}
    if write:
        stem = pdf_path.stem
        write_json(RAW_DIR / f"{stem}.json", [asdict(page) for page in pages])
        write_json(CHUNK_DIR / f"{stem}_chunks.json", [asdict(chunk) for chunk in chunks])
        write_json(OUTPUT_DIR / f"{stem}_summary.json", result)
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo OCR PDF tiếng Việt và chunking (Buổi 5).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Không ghi file và không gọi LlamaParse (mặc định).")
    mode.add_argument("--write", action="store_true", help="Ghi output; chỉ gọi LlamaParse khi fallback OCR cần thiết.")
    parser.add_argument("--force-ocr", action="store_true", help="Chỉ hợp lệ với --write; yêu cầu fallback OCR toàn bộ file.")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.force_ocr and not args.write:
        raise SystemExit("--force-ocr cần dùng cùng --write vì dry-run không gọi dịch vụ mạng.")
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"Không tìm thấy PDF trong {DATA_DIR}.")
    results = [process_pdf(path, write=args.write, force_ocr=args.force_ocr) for path in pdf_files]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
