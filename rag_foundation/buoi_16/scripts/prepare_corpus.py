from __future__ import annotations

import csv
import html
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import corpus_path, read_csv, source_dir


class BlockParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.blocks = []; self.current = None; self.buf = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"p", "li"}:
            self.current = attrs; self.buf = []
        elif tag == "br" and self.current is not None: self.buf.append(" ")
    def handle_data(self, data):
        if self.current is not None: self.buf.append(data)
    def handle_endtag(self, tag):
        if tag in {"p", "li"} and self.current is not None:
            text = re.sub(r"\s+", " ", html.unescape("".join(self.buf))).strip()
            if text: self.blocks.append((self.current, text))
            self.current = None; self.buf = []


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    src = source_dir(); metadata = {r["id"]: r for r in read_csv(src / "metadata.csv")}
    rows = []
    for doc in read_csv(src / "content.csv"):
        parser = BlockParser(); parser.feed(doc["content_html"])
        article = ""; clause_no = 0
        for pos, (attrs, text) in enumerate(parser.blocks, 1):
            cls = attrs.get("class", "")
            if "prov-article" in cls:
                match = re.match(r"(Điều\s+\d+[a-zA-Z]?)", text, re.I)
                article = match.group(1) if match else text[:100]; clause_no = 0
            if not article or cls not in {"prov-article", "prov-clause", "prov-content", "prov-item"}: continue
            clause_no += 1
            article_number = re.sub(r"\D", "", article) or "x"
            chunk_id = attrs.get("id") or f"{doc['id']}-a{article_number}-{clause_no}"
            meta = metadata.get(doc["id"], {})
            rows.append({"chunk_id": chunk_id, "document_id": doc["id"], "text": text,
                         "source_file": "content.csv", "title": meta.get("title", ""),
                         "document_type": meta.get("loai_van_ban", ""), "article": article,
                         "effective_date": meta.get("ngay_co_hieu_luc", ""), "status": meta.get("tinh_trang_hieu_luc", "")})
    seen = set()
    for row in rows:
        base = row["chunk_id"]; suffix = 2
        while row["chunk_id"] in seen: row["chunk_id"] = f"{base}-{suffix}"; suffix += 1
        seen.add(row["chunk_id"])
    out = corpus_path(); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
    print(f"Total chunks: {len(rows)}\nDocuments: {len(set(r['document_id'] for r in rows))}\nMissing text: {sum(not r['text'] for r in rows)}\nDuplicate chunk_id: {len(rows)-len(seen)}")
    for row in rows[:3]: print(row)


if __name__ == "__main__": main()
