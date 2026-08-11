import sys
import unicodedata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from process_pdf import PageData, chunk_fixed_size, chunk_hierarchical, chunk_semantic, needs_ocr, normalize_text


def test_normalize_text_nfc():
    text = "Tiếng Việt".replace("ế", "e\u0302\u0301")
    assert normalize_text(text) == unicodedata.normalize("NFC", text)


def test_needs_ocr_for_empty_replacement_or_mojibake():
    assert needs_ocr("") is True
    assert needs_ocr("Văn bản \ufffd lỗi") is True
    assert needs_ocr("VÄƒn báº£n lỗi encoding") is True


def test_chunk_fixed_size_overlap():
    page = PageData(source="test.pdf", page=1, text="A" * 1500, ocr_used=False)
    chunks = chunk_fixed_size(page, max_chars=500, overlap=100)
    assert len(chunks) == 4
    assert chunks[1].text.startswith("A" * 100)
    assert all(chunk.page_start == chunk.page_end == 1 for chunk in chunks)


def test_chunk_semantic_prefers_paragraphs_and_lines():
    page = PageData(source="test.pdf", page=1, text="Đoạn một.\n\nĐoạn hai.\nCòn lại!", ocr_used=False)
    chunks = chunk_semantic(page)
    assert [item.text for item in chunks] == ["Đoạn một.", "Đoạn hai. Còn lại!"]


def test_chunk_semantic_does_not_cut_a_sentence_at_a_layout_line_break():
    page = PageData(source="test.pdf", page=1, text="Một câu bị xuống\ndòng do dàn trang.", ocr_used=False)
    assert [item.text for item in chunk_semantic(page)] == ["Một câu bị xuống dòng do dàn trang."]


def test_chunk_semantic_uses_sentence_boundary_for_a_long_paragraph():
    text = "A" * 40 + ". " + "B" * 40 + "."
    chunks = chunk_semantic(PageData("test.pdf", 1, text, False), max_chars=50)
    assert [item.text for item in chunks] == ["A" * 40 + ".", "B" * 40 + "."]


def test_chunk_hierarchical_tracks_only_detected_structure():
    text = "Lời nói đầu\nChương I\nMở đầu\nĐiều 1. Phạm vi\n1. Nội dung\na) Điểm nhỏ"
    chunks = chunk_hierarchical(PageData("test.pdf", 1, text, False))
    assert len(chunks) == 5
    assert chunks[0].metadata["warning"].startswith("Đoạn mở đầu")
    assert chunks[1].metadata["structure"]["chapter"] == "Chương I"
    assert chunks[-1].metadata["structure"]["point"] == "a) Điểm nhỏ"


def test_chunk_hierarchical_warns_without_structure():
    chunks = chunk_hierarchical(PageData("test.pdf", 1, "Văn bản thường.", False))
    assert chunks[0].metadata["warning"] == "Không tìm thấy cấu trúc rõ ràng; không bịa heading."
