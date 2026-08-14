# Graph RAG Lab 1 — Buổi 10

Module tái tạo pipeline đã dùng để xây dựng cơ sở dữ liệu Neo4j `kb-hops`:

1. Đọc `data/metadata.csv`, `data/content.csv`, `data/relationships.csv`.
2. Làm sạch HTML và chia văn bản thành chunk phân cấp.
3. Tạo embedding 384 chiều bằng `thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5`.
4. Import `Document`, `Chunk`, `PART_OF`, `PARENT_OF`, `NEXT` và quan hệ cấp văn bản.
5. Tạo vector index `chunk_embedding_index` và kiểm tra số lượng graph.

Không commit `.env`. Sao chép `.env.example` thành `.env` và nhập cấu hình Neo4j cục bộ.

```powershell
python -m pip install -r requirements.txt
python pipeline.py --dry-run
python pipeline.py --import-neo4j
python verify_graph.py
```

`--dry-run` chỉ chunk và in mẫu, không tải model và không ghi Neo4j. Import sử dụng constraint và `MERGE`, nên chạy lại không tạo node hoặc cạnh trùng.
